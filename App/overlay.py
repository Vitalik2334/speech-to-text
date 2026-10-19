"""
overlay.py — плаваюча картка-оверлей на PySide6 (Qt), яка показує статус
push-to-talk диктовки (запис / розпізнавання / готовий текст).

Картка з'являється ПО ЦЕНТРУ екрана і не забирає фокус клавіатури.

Анімація тексту:
  * "listening" без розпізнаного тексту (прийшло LISTENING_PLACEHOLDER) —
    замість тексту в циклі набираються три крапки (. → .. → ... → порожньо);
    повторні такі виклики цикл не перезапускають;
  * будь-який текст — «друкується» по літері; якщо новий текст продовжує
    вже показаний (проміжні результати під час запису), друкується лише
    нова частина.
Ненадрукована частина малюється прозорим кольором, тому розмір картки
рахується одразу під кінцевий текст і не змінюється під час анімації.

Потокобезпека: display() / dismiss() / copy_text() можна викликати з
будь-якого потоку — вони лише emit()-ять Qt-сигнал, а робота з віджетами
йде в _on_*-слотах у GUI-потоці (queued connection).

Сигнали життєвого циклу: display_started випускається на початку обробки
display(), display_finished — коли текст надруковано повністю (для
"listening" — одразу після запуску циклу крапок). Самому оверлею вони
не потрібні — це точки підключення для зовнішніх спостерігачів (наприклад,
profiling.py).
"""
import html
import threading

from PySide6.QtCore import (
    Property,
    QEasingCurve,
    QPropertyAnimation,
    Qt,
    QTimer,
    QVariantAnimation,
    Signal,
)
from PySide6.QtGui import QColor, QGuiApplication, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QWidget,
)

_TRANSPARENT_SPAN = '<span style="color: transparent;">{}</span>'


def _to_html(text: str) -> str:
    """Екранує текст для RichText і зберігає переноси рядків."""
    return html.escape(text).replace("\n", "<br>")


class StateDot(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(10, 10)
        self._color = QColor("#5b8dee")
        self._opacity = 1.0

        self._anim = QPropertyAnimation(self, b"dotOpacity", self)
        self._anim.setDuration(1100)
        self._anim.setLoopCount(-1)
        self._anim.setEasingCurve(QEasingCurve.Type.InOutSine)
        self._anim.setKeyValueAt(0.0, 1.0)
        self._anim.setKeyValueAt(0.5, 0.3)
        self._anim.setKeyValueAt(1.0, 1.0)

    def set_color(self, hex_color: str) -> None:
        self._color = QColor(hex_color)
        self.update()

    def pulse(self, enabled: bool) -> None:
        if enabled:
            if self._anim.state() != QPropertyAnimation.State.Running:
                self._anim.start()
        else:
            self._anim.stop()
            self._opacity = 1.0
            self.update()

    def _get_dot_opacity(self) -> float:
        return self._opacity

    def _set_dot_opacity(self, value: float) -> None:
        self._opacity = value
        self.update()

    dotOpacity = Property(float, _get_dot_opacity, _set_dot_opacity)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = QColor(self._color)
        color.setAlphaF(self._opacity)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawEllipse(self.rect())


class Overlay(QWidget):
    _display_signal = Signal(str, str, object)   # text, state, autohide_ms
    _dismiss_signal = Signal()
    _copy_signal = Signal(str)

    # Публічні сигнали для спостерігачів (див. докстрінг модуля).
    display_started = Signal()
    display_finished = Signal()

    STATE_COLORS = {
        "info": "#5b8dee",
        "listening": "#ef5b5b",
        "processing": "#e8b93d",
        "result": "#4caf82",
        "warning": "#e8b93d",
    }

    # Параметри анімації тексту.
    TYPE_MS_PER_CHAR = 20
    TYPE_MIN_MS = 150
    TYPE_MAX_MS = 800
    DOTS_MAX = 3
    DOTS_INTERVAL_MS = 200
    # Текст, який застосунок надсилає в "listening", поки нічого не розпізнано.
    LISTENING_PLACEHOLDER = "Слухаю..."

    def __init__(self) -> None:
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)

        # Сигналізує фоновому потоку, що текст реально потрапив у буфер обміну
        # (потрібно перед Ctrl+V, інакше вставиться старий вміст буфера).
        self._copied = threading.Event()

        self.dot = StateDot(self)
        self.label = QLabel("")
        self.label.setTextFormat(Qt.TextFormat.RichText)
        self.label.setWordWrap(True)
        self.label.setMaximumWidth(420)
        self.label.setStyleSheet(
            "color: #f0f0f0; font-size: 14px; font-family: 'Segoe UI', sans-serif;"
        )

        row = QHBoxLayout(self)
        row.setContentsMargins(18, 14, 20, 14)
        row.setSpacing(12)
        row.addWidget(self.dot, 0, Qt.AlignmentFlag.AlignTop)
        row.addWidget(self.label, 1)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(36)
        shadow.setOffset(0, 8)
        shadow.setColor(QColor(0, 0, 0, 180))
        self.setGraphicsEffect(shadow)

        self._fade_anim = QPropertyAnimation(self, b"windowOpacity", self)
        self._fade_anim.setDuration(160)
        self._pending_hide = False
        self._fade_anim.finished.connect(self._on_fade_finished)

        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self._fade_out)

        # Друк тексту по літері: анімація рахує кількість показаних символів.
        self._full_text = ""
        self._shown_count = 0
        self._mode: str | None = None   # None | "dots" | "text"
        self._pending_autohide: object = None
        self._type_anim = QVariantAnimation(self)
        self._type_anim.setEasingCurve(QEasingCurve.Type.Linear)
        self._type_anim.valueChanged.connect(self._on_type_step)
        self._type_anim.finished.connect(self._on_type_finished)

        # Цикл трьох крапок для стану "listening".
        self._dots_count = 0
        self._dots_suffix = ""
        self._dots_timer = QTimer(self)
        self._dots_timer.setInterval(self.DOTS_INTERVAL_MS)
        self._dots_timer.timeout.connect(self._on_dots_tick)

        self._display_signal.connect(self._on_display)
        self._dismiss_signal.connect(self._on_dismiss)
        self._copy_signal.connect(self._on_copy)

        self.setWindowOpacity(0.0)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(self.rect().adjusted(1, 1, -1, -1), 16, 16)
        painter.fillPath(path, QColor(24, 24, 26, 235))
        pen = QPen(QColor(255, 255, 255, 28))
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawPath(path)

    # ---- рендеринг анімованого тексту ---------------------------------

    def _render_partial(self, count: int) -> None:
        """Показує перші count символів, решту — прозорими (тримають розмір)."""
        self._shown_count = count
        visible = _to_html(self._full_text[:count])
        hidden = self._full_text[count:]
        if hidden:
            visible += _TRANSPARENT_SPAN.format(_to_html(hidden))
        self.label.setText(visible)

    def _render_dots(self) -> None:
        """Показує _dots_count крапок, решту до трьох — прозорими."""
        visible = "." * self._dots_count
        hidden = "." * (self.DOTS_MAX - self._dots_count)
        if hidden:
            visible += _TRANSPARENT_SPAN.format(hidden)
        self.label.setText(visible + _to_html(self._dots_suffix))

    def _stop_text_animations(self) -> None:
        self._type_anim.stop()
        self._dots_timer.stop()

    # ---- слоти: виконуються ТІЛЬКИ в GUI-потоці --------------------------

    def _on_display(self, text: str, state: str, autohide_ms: object) -> None:
        self.display_started.emit()
        self._hide_timer.stop()

        color = self.STATE_COLORS.get(state, self.STATE_COLORS["info"])
        self.dot.set_color(color)
        self.dot.pulse(state == "listening")
        self._pending_autohide = autohide_ms

        if state == "listening" and text.startswith(self.LISTENING_PLACEHOLDER):
            self._show_dots(text[len(self.LISTENING_PLACEHOLDER):])
        else:
            self._start_typing(text)

    def _show_dots(self, suffix: str) -> None:
        """Цикл крапок замість тексту. Повторний виклик його не перезапускає."""
        already_running = (
            self._mode == "dots"
            and self._dots_timer.isActive()
            and suffix == self._dots_suffix
        )
        if not already_running:
            self._type_anim.stop()
            self._mode = "dots"
            self._full_text = ""
            self._shown_count = 0
            self._dots_suffix = suffix
            self._dots_count = 0
            self._render_dots()
            self._show_card()
            self._dots_timer.start()
        self._start_autohide()
        self.display_finished.emit()

    def _start_typing(self, text: str) -> None:
        """Друкує текст. Якщо він продовжує вже показаний — лише нову частину."""
        self._dots_timer.stop()
        self._type_anim.stop()

        start = 0
        if self._mode == "text":
            prefix = 0
            limit = min(len(self._full_text), len(text))
            while prefix < limit and self._full_text[prefix] == text[prefix]:
                prefix += 1
            start = min(self._shown_count, prefix)

        self._mode = "text"
        self._full_text = text
        self._render_partial(start)
        self._show_card()

        remaining = len(text) - start
        if remaining <= 0:
            self._on_type_finished()
            return

        duration = remaining * self.TYPE_MS_PER_CHAR
        duration = max(self.TYPE_MIN_MS, min(self.TYPE_MAX_MS, duration))
        self._type_anim.setDuration(duration)
        self._type_anim.setStartValue(start)
        self._type_anim.setEndValue(len(text))
        self._type_anim.start()

    def _on_type_step(self, value) -> None:
        count = int(value)
        if count != self._shown_count:   # анімація тікає частіше, ніж міняється літера
            self._render_partial(count)

    def _on_type_finished(self) -> None:
        self._render_partial(len(self._full_text))
        self._start_autohide()
        self.display_finished.emit()

    def _on_dots_tick(self) -> None:
        self._dots_count = (self._dots_count + 1) % (self.DOTS_MAX + 1)
        self._render_dots()

    def _on_dismiss(self) -> None:
        self._hide_timer.stop()
        self._stop_text_animations()
        self._mode = None
        self._fade_out()

    def _on_copy(self, text: str) -> None:
        QGuiApplication.clipboard().setText(text)
        self._copied.set()

    # ---- допоміжне ------------------------------------------------------

    def _show_card(self) -> None:
        self.adjustSize()
        self._reposition()
        self._fade_to(1.0)
        self.show()

    def _start_autohide(self) -> None:
        if self._pending_autohide:
            self._hide_timer.start(int(self._pending_autohide))

    def _reposition(self) -> None:
        screen = QGuiApplication.primaryScreen().availableGeometry()
        x = screen.center().x() - self.width() // 2
        y = screen.center().y() - self.height() // 2
        self.move(x, y)

    def _on_fade_finished(self) -> None:
        if self._pending_hide:
            self.hide()

    def _fade_to(self, value: float) -> None:
        self._fade_anim.stop()
        self._pending_hide = value <= 0.0
        self._fade_anim.setStartValue(self.windowOpacity())
        self._fade_anim.setEndValue(value)
        self._fade_anim.start()

    def _fade_out(self) -> None:
        self._fade_to(0.0)

    # ---- публічний API: безпечно викликати з будь-якого потоку ---------

    def display(self, text: str, state: str = "info", autohide_ms: int | None = None) -> None:
        self._display_signal.emit(text, state, autohide_ms)

    def dismiss(self) -> None:
        self._dismiss_signal.emit()

    def copy_text(self, text: str) -> None:
        self._copied.clear()
        self._copy_signal.emit(text)

    def copy_text_and_wait(self, text: str, timeout: float = 1.0) -> bool:
        """Як copy_text(), але чекає, поки GUI-потік справді запише текст у
        буфер обміну. Викликати ТІЛЬКИ з фонового потоку (з GUI-потоку —
        дедлок до timeout). Повертає False, якщо не встигло."""
        self.copy_text(text)
        return self._copied.wait(timeout)
