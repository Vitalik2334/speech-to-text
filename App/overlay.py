"""
overlay.py — плаваюча картка-оверлей на PySide6 (Qt), яка показує статус
push-to-talk диктовки (запис / розпізнавання / готовий текст).

Картка з'являється ПО ЦЕНТРУ екрана і не забирає фокус клавіатури.

Вигляд (дизайн з канвасу «Оверлей · шаблони станів»):
  * ширина картки від 320 до 640 px: розширює її лише текст, далі — перенос;
  * кути 6 px, темний фон із сіткою (крок 12 / 60 px);
  * рамка 2 px у кольорі стану, колір між станами змінюється плавно (200 мс);
  * знизу по центру — сірий лоток-вкладка 28 px: верх заокруглений, низ
    плавно розходиться в боки й зливається з краєм картки. У лотку:
    глобус (мова) зліва, риски голосу по центру, «надіслати» справа;
  * риски голосу: у "listening" — 9 червоних рисок дзеркальною хвилею від
    центру, що реагують на гучність мікрофона (feed_audio); в інших станах —
    нерухомі сірі крапки;
  * «надіслати» світла, коли є текст, і контурна, коли тексту немає.
  Іконки малюються QPainter-ом — жодних файлів-картинок для PyInstaller.

Анімація тексту:
  * "listening" без розпізнаного тексту (прийшло LISTENING_PLACEHOLDER) —
    замість тексту мигає курсор «_» (крок CARET_INTERVAL_MS); повторні
    такі виклики мигання не перезапускають;
  * будь-який текст — «друкується» по літері; якщо новий текст продовжує
    вже показаний (проміжні результати під час запису), друкується лише
    нова частина.
Ненадрукована частина малюється прозорим кольором, тому розмір картки
рахується одразу під кінцевий текст і не змінюється під час анімації.

Потокобезпека: display() / dismiss() / copy_text() / set_language() /
feed_audio() можна викликати з будь-якого потоку — вони лише emit()-ять Qt-сигнал, а робота з
віджетами йде в _on_*-слотах у GUI-потоці (queued connection).

Сигнали життєвого циклу: display_started випускається на початку обробки
display(), display_finished — коли текст надруковано повністю (для
"listening" без тексту — одразу після запуску курсора). Самому оверлею вони
не потрібні — це точки підключення для зовнішніх спостерігачів (наприклад,
profiling.py).

Рівень голосу (feed_audio, з потоку мікрофона): RMS чанка → dBFS (як
frame_levels_dbfs у UAFastConformer.py) → 0..1 між адаптивним порогом шуму
(10-й перцентиль останніх ~3 с + LEVEL_NOISE_MARGIN_DB, не нижче
LEVEL_FLOOR_MIN_DB) і LEVEL_CEIL_DB. У GUI летить не частіше LEVEL_EMIT_MS
і лише під час "listening". Риски: швидкий підйом, повільний спад.

Сигнали кнопок: language_clicked, send_clicked — випускаються
в GUI-потоці при кліку. Оверлей сам нічого на них не робить: дії
підключає застосунок.
"""
import collections
import html
import threading
import time

import numpy as np

from PySide6.QtCore import (
    QEasingCurve,
    QPointF,
    QPropertyAnimation,
    QRectF,
    Qt,
    QTimer,
    QVariantAnimation,
    Signal,
)
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetricsF,
    QGuiApplication,
    QPainter,
    QPainterPath,
    QPen,
    QRadialGradient,
)
from PySide6.QtWidgets import (
    QAbstractButton,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

_TRANSPARENT_SPAN = '<span style="color: transparent;">{}</span>'


def _to_html(text: str) -> str:
    """Екранує текст для RichText і зберігає переноси рядків."""
    return html.escape(text).replace("\n", "<br>")


# ---- дизайн-токени (див. канвас «Оверлей · усі стани») ------------------

CARD_MIN_WIDTH = 320
CARD_MAX_WIDTH = 640
CARD_PAD_X = 20
CARD_PAD_Y = 16
CARD_RADIUS = 6
CARD_BG = QColor(24, 24, 26, 240)          # #18181a, α 0.94
BORDER_WIDTH = 2
TEXT_COLOR = "#f0f0f0"

GRID_MINOR_STEP = 12
GRID_MAJOR_STEP = 60
GRID_MINOR = QColor(255, 255, 255, 8)      # α 0.03
GRID_MAJOR = QColor(255, 255, 255, 19)     # α 0.075
VIGNETTE = QColor(255, 255, 255, 11)       # α 0.045

TRAY_BG = QColor("#2c2c30")
TRAY_HEIGHT = 28
TRAY_WIDTH = 200
TRAY_RADIUS = 14                           # верхні кути й нижні «крила»

BUTTON_SIZE = 20
RED = QColor("#ef5b5b")
GLOBE = QColor("#a8abb2")
GLOBE_HOVER_BG = QColor(255, 255, 255, 20)
BAR_IDLE = QColor(255, 255, 255, 71)       # α 0.28
SEND_IDLE_FG = QColor("#c9cbd0")
SEND_IDLE_BORDER = QColor(255, 255, 255, 51)   # α 0.2
SEND_ACTIVE_BG = QColor("#f0f0f0")
SEND_ACTIVE_FG = QColor("#18181a")

BAR_COUNT = 9
BAR_WIDTH = 2
BAR_GAP = 2
BAR_MIN_H = 2
BAR_MAX_H = 16
BAR_TICK_MS = 50

# Рівень голосу → 0..1 (див. докстрінг модуля).
LEVEL_FLOOR_MIN_DB = -45.0     # = SILENCE_DBFS у UAFastConformer.py
LEVEL_NOISE_MARGIN_DB = 4.0    # (було 8 = PAUSE_MARGIN_DB) менше — риски оживають від тихішого голосу
LEVEL_NOISE_PERCENTILE = 10    # = PAUSE_NOISE_PERCENTILE
LEVEL_NOISE_WINDOW_SEC = 3.0
LEVEL_CEIL_DB = -28.0          # (було -15) рівень, на якому риски вже на повну висоту
LEVEL_MIN_SPAN_DB = 12.0
LEVEL_CURVE = 0.6              # <1 підсилює тихі звуки (рівень ** LEVEL_CURVE); 1.0 — лінійно
LEVEL_EMIT_MS = 33
LEVEL_ATTACK = 0.5             # частка за тік 50 мс ≈ 30 мс підйому
LEVEL_RELEASE = 0.2            # ≈ 200–250 мс спаду


def _icon_painter(painter: QPainter, rect: QRectF, size: float) -> None:
    """Переводить систему координат у viewBox 24×24 іконки розміром size,
    відцентрованої в rect (так іконки малюються з тих самих координат, що й SVG
    у макеті)."""
    painter.translate(rect.center())
    painter.scale(size / 24.0, size / 24.0)
    painter.translate(-12, -12)


def _stroke_pen(color: QColor, width: float) -> QPen:
    pen = QPen(color, width)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    return pen


# ---- кнопки лотка ---------------------------------------------------------

class _GlobeButton(QAbstractButton):
    """Мова розпізнавання: лише іконка глобуса, без рамки й фону."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(BUTTON_SIZE, BUTTON_SIZE)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.set_language("українська")

    def set_language(self, name: str) -> None:
        self.setToolTip(f"Мова: {name}")
        self.setAccessibleName(f"Мова розпізнавання: {name}")

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        hovered = self.underMouse()
        if hovered:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(GLOBE_HOVER_BG)
            p.drawEllipse(QRectF(self.rect()))
        _icon_painter(p, QRectF(self.rect()), 13)
        p.setPen(_stroke_pen(QColor(TEXT_COLOR) if hovered else GLOBE, 2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QPointF(12, 12), 9, 9)
        p.drawLine(QPointF(3, 12), QPointF(21, 12))
        meridian = QPainterPath()
        meridian.moveTo(12, 3)
        meridian.quadTo(19.5, 12, 12, 21)
        meridian.quadTo(4.5, 12, 12, 3)
        p.drawPath(meridian)


class _VoiceLevelMeter:
    """dBFS чанків → рівень 0..1 з адаптивним порогом шуму.
    Живе в GUI-потоці; на вхід — уже пораховані dBFS."""

    def __init__(self) -> None:
        maxlen = int(LEVEL_NOISE_WINDOW_SEC * 1000 / LEVEL_EMIT_MS)
        self._history: collections.deque[float] = collections.deque(maxlen=maxlen)

    def normalize(self, db: float) -> float:
        self._history.append(db)
        floor = LEVEL_FLOOR_MIN_DB
        if len(self._history) >= 10:
            noise = float(np.percentile(np.fromiter(self._history, float), LEVEL_NOISE_PERCENTILE))
            floor = max(noise + LEVEL_NOISE_MARGIN_DB, LEVEL_FLOOR_MIN_DB)
        floor = min(floor, LEVEL_CEIL_DB - LEVEL_MIN_SPAN_DB)
        level = min(1.0, max(0.0, (db - floor) / (LEVEL_CEIL_DB - floor)))
        return level ** LEVEL_CURVE


class _VoiceBars(QWidget):
    """9 рисок дзеркальною хвилею від центру: голосніше — вищі й ширше."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        width = BAR_COUNT * BAR_WIDTH + (BAR_COUNT - 1) * BAR_GAP
        self.setFixedSize(width, BUTTON_SIZE)
        self.setAccessibleName("Рівень голосу")
        self._active = False
        self._target = 0.0            # рівень 0..1, що прийшов з мікрофона
        self._level = 0.0             # згладжений (баластика)
        self._bars = [0.0] * BAR_COUNT
        self._timer = QTimer(self)
        self._timer.setInterval(BAR_TICK_MS)
        self._timer.timeout.connect(self._tick)

    def set_active(self, active: bool) -> None:
        if active == self._active:
            return
        self._active = active
        self._target = self._level = 0.0
        self._bars = [0.0] * BAR_COUNT
        if active:
            self._timer.start()
        else:
            self._timer.stop()
        self.update()

    def set_level(self, level: float) -> None:
        self._target = level

    def _tick(self) -> None:
        k = LEVEL_ATTACK if self._target > self._level else LEVEL_RELEASE
        self._level += (self._target - self._level) * k
        lv = self._level
        centre = (BAR_COUNT - 1) / 2
        reach_span = 0.25 + 0.85 * lv
        rnd = np.random.random(BAR_COUNT)
        for i in range(BAR_COUNT):
            d = abs(i - centre) / centre
            reach = max(0.0, 1.0 - d / reach_span)
            v = lv * reach * (0.75 + 0.25 * rnd[i])
            b = self._bars[i]
            self._bars[i] = b + (v - b) * (0.6 if v > b else 0.25)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        mid = self.height() / 2
        for i, v in enumerate(self._bars):
            if self._active:
                h = BAR_MIN_H + (BAR_MAX_H - BAR_MIN_H) * min(1.0, v)
                color = QColor(RED)
                color.setAlphaF(0.55 + 0.45 * min(1.0, v))
            else:
                h = BAR_MIN_H
                color = BAR_IDLE
            p.setBrush(color)
            x = i * (BAR_WIDTH + BAR_GAP)
            p.drawRoundedRect(QRectF(x, mid - h / 2, BAR_WIDTH, h), BAR_WIDTH / 2, BAR_WIDTH / 2)


class _SendButton(QAbstractButton):
    """«Надіслати»: світла, коли є текст; контурна, коли тексту немає."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(BUTTON_SIZE, BUTTON_SIZE)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setAccessibleName("Надіслати текст")
        self._active = False

    def set_active(self, active: bool) -> None:
        if active != self._active:
            self._active = active
            self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        if self._active:
            p.setPen(QPen(SEND_ACTIVE_BG, 1))
            p.setBrush(SEND_ACTIVE_BG)
            fg = SEND_ACTIVE_FG
        else:
            p.setPen(QPen(SEND_IDLE_BORDER, 1))
            p.setBrush(Qt.BrushStyle.NoBrush)
            fg = SEND_IDLE_FG
        p.drawEllipse(r)

        _icon_painter(p, QRectF(self.rect()), 11)
        p.setPen(_stroke_pen(fg, 2.4))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawLine(QPointF(12, 19), QPointF(12, 5))
        p.drawPolyline([QPointF(6, 11), QPointF(12, 5), QPointF(18, 11)])


class _Tray(QWidget):
    """Сірий лоток-вкладка під текстом: глобус, риски голосу, «надіслати». «Крила» внизу (увігнуті заокруглення,
    що зливають лоток із краєм картки) малює сама картка — вони виходять за
    межі лотка."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(TRAY_WIDTH, TRAY_HEIGHT)
        self.globe = _GlobeButton(self)
        self.bars = _VoiceBars(self)
        self.send = _SendButton(self)

        # Глобус і «надіслати» однакової ширини → риски точно по центру.
        row = QHBoxLayout(self)
        row.setContentsMargins(10, 4, 10, 4)
        row.setSpacing(0)
        row.addWidget(self.globe)
        row.addStretch(1)
        row.addWidget(self.bars)
        row.addStretch(1)
        row.addWidget(self.send)

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect())
        path = QPainterPath()
        path.addRoundedRect(r, TRAY_RADIUS, TRAY_RADIUS)
        path.addRect(r.adjusted(0, r.height() / 2, 0, 0))     # низ прямий
        path.setFillRule(Qt.FillRule.WindingFill)
        p.fillPath(path.simplified(), TRAY_BG)


class _BorderLayer(QWidget):
    """Кольорова рамка стану поверх усього вмісту (у т.ч. поверх лотка)."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.color = QColor("#5b8dee")

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        half = BORDER_WIDTH / 2
        p.setPen(QPen(self.color, BORDER_WIDTH))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(QRectF(self.rect()).adjusted(half, half, -half, -half),
                          CARD_RADIUS - half, CARD_RADIUS - half)


# ---- сама картка ------------------------------------------------------------

class Overlay(QWidget):
    _display_signal = Signal(str, str, object)   # text, state, autohide_ms
    _dismiss_signal = Signal()
    _copy_signal = Signal(str)
    _language_signal = Signal(str)
    _level_signal = Signal(float)              # dBFS чанка мікрофона

    # Публічні сигнали для спостерігачів (див. докстрінг модуля).
    display_started = Signal()
    display_finished = Signal()

    # Кліки по кнопках лотка (GUI-потік).
    language_clicked = Signal()
    send_clicked = Signal()

    STATE_COLORS = {
        "info": "#5b8dee",
        "listening": "#ef5b5b",
        "processing": "#e8b93d",
        "result": "#4caf82",
        "warning": "#e8b93d",
    }

    # Параметри анімації.
    TYPE_MS_PER_CHAR = 20
    TYPE_MIN_MS = 150
    TYPE_MAX_MS = 800
    CARET = "_"
    CARET_INTERVAL_MS = 500
    BORDER_FADE_MS = 200
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
        self.setFixedWidth(CARD_MIN_WIDTH)

        # Сигналізує фоновому потоку, що текст реально потрапив у буфер обміну
        # (потрібно перед Ctrl+V, інакше вставиться старий вміст буфера).
        self._copied = threading.Event()

        self.label = QLabel("")
        self.label.setTextFormat(Qt.TextFormat.RichText)
        self.label.setWordWrap(True)
        text_font = QFont("Segoe UI")
        text_font.setPixelSize(14)
        self.label.setFont(text_font)
        self.label.setStyleSheet(f"color: {TEXT_COLOR};")
        self.label.setFixedWidth(CARD_MIN_WIDTH - 2 * CARD_PAD_X)

        self.tray = _Tray(self)
        self.tray.globe.clicked.connect(self.language_clicked)
        self.tray.send.clicked.connect(self.send_clicked)

        # Рівень голосу: потік мікрофона → (не частіше LEVEL_EMIT_MS) → GUI.
        self._meter = _VoiceLevelMeter()
        self._listening = False                # читає потік мікрофона
        self._last_level_emit = 0.0
        self._level_peak_rms = 0.0

        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        text_box = QVBoxLayout()
        text_box.setContentsMargins(CARD_PAD_X, CARD_PAD_Y, CARD_PAD_X, CARD_PAD_Y)
        text_box.addWidget(self.label)
        column.addLayout(text_box)
        column.addWidget(self.tray, 0, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom)

        self._border = _BorderLayer(self)
        self._border.raise_()

        # Плавна зміна кольору рамки між станами.
        self._border_anim = QVariantAnimation(self)
        self._border_anim.setDuration(self.BORDER_FADE_MS)
        self._border_anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self._border_anim.valueChanged.connect(self._on_border_color)

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
        self._mode: str | None = None   # None | "caret" | "text"
        self._pending_autohide: object = None
        self._type_anim = QVariantAnimation(self)
        self._type_anim.setEasingCurve(QEasingCurve.Type.Linear)
        self._type_anim.valueChanged.connect(self._on_type_step)
        self._type_anim.finished.connect(self._on_type_finished)

        # Мигаючий курсор «_» для стану "listening" без тексту.
        self._caret_visible = True
        self._caret_suffix = ""
        self._caret_timer = QTimer(self)
        self._caret_timer.setInterval(self.CARET_INTERVAL_MS)
        self._caret_timer.timeout.connect(self._on_caret_tick)

        self._display_signal.connect(self._on_display)
        self._dismiss_signal.connect(self._on_dismiss)
        self._copy_signal.connect(self._on_copy)
        self._language_signal.connect(self.tray.globe.set_language)
        self._level_signal.connect(self._on_level)

        self.setWindowOpacity(0.0)

    # ---- малювання картки -------------------------------------------------

    def resizeEvent(self, event) -> None:  # noqa: N802
        self._border.setGeometry(self.rect())
        super().resizeEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect())
        card = QPainterPath()
        card.addRoundedRect(rect, CARD_RADIUS, CARD_RADIUS)
        painter.fillPath(card, CARD_BG)
        painter.setClipPath(card)

        # Легка віньєтка: світліше по центру.
        glow = QRadialGradient(QPointF(rect.width() * 0.5, rect.height() * 0.4), rect.width() * 0.5)
        glow.setColorAt(0.0, VIGNETTE)
        glow.setColorAt(0.7, QColor(255, 255, 255, 0))
        painter.fillRect(rect, glow)

        # Сітка: дрібна (12 px) і помітніша кожна п'ята (60 px).
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        w, h = self.width(), self.height()
        for step, color in ((GRID_MINOR_STEP, GRID_MINOR), (GRID_MAJOR_STEP, GRID_MAJOR)):
            painter.setPen(QPen(color, 1))
            for x in range(step - 1, w, step):
                painter.drawLine(x, 0, x, h)
            for y in range(step - 1, h, step):
                painter.drawLine(0, y, w, y)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # «Крила» лотка: увігнуті заокруглення зліва й справа від нього.
        g = QRectF(self.tray.geometry())
        r = TRAY_RADIUS
        for corner_x, square_x in ((g.left() - r, g.left() - r), (g.right() + r, g.right())):
            wing = QPainterPath()
            wing.addRect(QRectF(square_x, h - r, r, r))
            hole = QPainterPath()
            hole.addEllipse(QPointF(corner_x, h - r), r, r)
            painter.fillPath(wing.subtracted(hole), TRAY_BG)

    def _on_border_color(self, value) -> None:
        self._border.color = QColor(value)
        self._border.update()

    def _set_border_color(self, hex_color: str) -> None:
        target = QColor(hex_color)
        if target == self._border.color and self._border_anim.state() != QVariantAnimation.State.Running:
            return
        self._border_anim.stop()
        if not self.isVisible() or self.windowOpacity() == 0.0:
            self._on_border_color(target)      # картка схована — без переходу
            return
        self._border_anim.setStartValue(QColor(self._border.color))
        self._border_anim.setEndValue(target)
        self._border_anim.start()

    # ---- рендеринг анімованого тексту ---------------------------------

    def _render_partial(self, count: int) -> None:
        """Показує перші count символів, решту — прозорими (тримають розмір)."""
        self._shown_count = count
        visible = _to_html(self._full_text[:count])
        hidden = self._full_text[count:]
        if hidden:
            visible += _TRANSPARENT_SPAN.format(_to_html(hidden))
        self.label.setText(visible)

    def _render_caret(self) -> None:
        """Показує «_» або прозорий «_» (місце під нього не зникає)."""
        caret = self.CARET if self._caret_visible else _TRANSPARENT_SPAN.format(self.CARET)
        self.label.setText(caret + _to_html(self._caret_suffix))

    def _stop_text_animations(self) -> None:
        self._type_anim.stop()
        self._caret_timer.stop()

    # ---- слоти: виконуються ТІЛЬКИ в GUI-потоці --------------------------

    def _on_display(self, text: str, state: str, autohide_ms: object) -> None:
        self.display_started.emit()
        self._hide_timer.stop()

        color = self.STATE_COLORS.get(state, self.STATE_COLORS["info"])
        self._set_border_color(color)
        self._listening = state == "listening"
        self.tray.bars.set_active(self._listening)
        self._pending_autohide = autohide_ms

        if state == "listening" and text.startswith(self.LISTENING_PLACEHOLDER):
            self.tray.send.set_active(False)
            self._show_caret(text[len(self.LISTENING_PLACEHOLDER):])
        else:
            self.tray.send.set_active(state in ("listening", "result") and bool(text.strip()))
            self._start_typing(text)

    def _show_caret(self, suffix: str) -> None:
        """Мигаючий «_» замість тексту. Повторний виклик його не перезапускає."""
        already_running = (
            self._mode == "caret"
            and self._caret_timer.isActive()
            and suffix == self._caret_suffix
        )
        if not already_running:
            self._type_anim.stop()
            self._mode = "caret"
            self._full_text = ""
            self._shown_count = 0
            self._caret_suffix = suffix
            self._caret_visible = True
            self._fit_width(self.CARET + suffix)
            self._render_caret()
            self._show_card()
            self._caret_timer.start()
        self._start_autohide()
        self.display_finished.emit()

    def _start_typing(self, text: str) -> None:
        """Друкує текст. Якщо він продовжує вже показаний — лише нову частину."""
        self._caret_timer.stop()
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
        self._fit_width(text)
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

    def _on_caret_tick(self) -> None:
        self._caret_visible = not self._caret_visible
        self._render_caret()

    def _on_level(self, db: float) -> None:
        if self._listening:
            self.tray.bars.set_level(self._meter.normalize(db))

    def _fit_width(self, text: str) -> None:
        """Ширина картки під КІНЦЕВИЙ текст: від CARD_MIN_WIDTH до
        CARD_MAX_WIDTH; довший текст переноситься. Рахується до анімації
        друку, тож під час друку ширина не змінюється."""
        metrics = QFontMetricsF(self.label.font())
        natural = max((metrics.horizontalAdvance(line) for line in text.split("\n")), default=0.0)
        width = int(min(CARD_MAX_WIDTH, max(CARD_MIN_WIDTH, natural + 2 * CARD_PAD_X + 2)))
        self.label.setFixedWidth(width - 2 * CARD_PAD_X)
        self.setFixedWidth(width)

    def _on_dismiss(self) -> None:
        self._listening = False
        self.tray.bars.set_active(False)
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

    def set_language(self, name: str) -> None:
        """Назва мови для підказки й екранних читачів, напр. "українська"."""
        self._language_signal.emit(name)

    def feed_audio(self, chunk: np.ndarray) -> None:
        """Чанк з мікрофона (float32, моно). Викликати з потоку запису —
        рахує dBFS тут же (дешево) і шле в GUI не частіше LEVEL_EMIT_MS,
        лише поки оверлей у стані "listening"."""
        if not self._listening or len(chunk) == 0:
            return
        # Між відправками беремо найгучніший чанк, щоб не губити короткі склади.
        rms = float(np.sqrt(np.mean(np.square(chunk, dtype=np.float32))))
        self._level_peak_rms = max(self._level_peak_rms, rms)
        now = time.perf_counter()
        if (now - self._last_level_emit) * 1000 < LEVEL_EMIT_MS:
            return
        self._last_level_emit = now
        peak, self._level_peak_rms = self._level_peak_rms, 0.0
        self._level_signal.emit(20.0 * np.log10(peak + 1e-12))

    def copy_text(self, text: str) -> None:
        self._copied.clear()
        self._copy_signal.emit(text)

    def copy_text_and_wait(self, text: str, timeout: float = 1.0) -> bool:
        """Як copy_text(), але чекає, поки GUI-потік справді запише текст у
        буфер обміну. Викликати ТІЛЬКИ з фонового потоку (з GUI-потоку —
        дедлок до timeout). Повертає False, якщо не встигло."""
        self.copy_text(text)
        return self._copied.wait(timeout)
