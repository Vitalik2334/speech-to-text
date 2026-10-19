"""
profiling.py — профілювання push-to-talk диктовки, яке під'єднується ЗЗОВНІ.

Робочий код (UAFastConformer.py) про профайлер нічого не знає. Він лише:
  * розбитий на окремі методи DictationApp, які можна обгорнути;
  * випускає кілька семантичних подій через app.events (там, де обгорнути
    нічого не вийде: "з'явився перший проміжний текст", "фінал пропущено").
Overlay віддає сигнали display_started / display_finished.

Усе інше робить attach_profiler(app):
  1. Обгортає методи ІНСТАНСУ (app.asr_pass, app.recorder.stop, ...) —
     setattr на інстанс, клас не змінюється.
  2. Обгортає функції модуля застосунку (apply_voice_gate, find_commit_point,
     should_skip_recognition) — вони викликаються за глобальним ім'ям, тому
     підміна в модулі підхоплюється одразу.
  3. Обгортає onnxruntime.InferenceSession.run (час кожної ONNX-моделі).
  4. Підписується на app.events і сигнали оверлея.
  5. Запускає сторож зависання Qt event loop.

Важливо щодо порядку: attach_profiler() викликати ПІСЛЯ створення
DictationApp, але ДО app.load() (onnx-обгортка і заміри завантаження) і
ДО app.run() (хоткеї та Qt-сигнали підключаються в run() і мусять
побачити вже обгорнуті методи).

    app = DictationApp()
    attach_profiler(app)      # або нічого — і жодних накладних витрат
    app.load()
    app.run()
"""
from __future__ import annotations

import collections
import os
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


@dataclass
class ProfilerOptions:
    per_step: bool = True        # окремий рядок на кожен прохід ASR
    onnx: bool = True            # міряти кожен InferenceSession.run
    gui: bool = True             # затримки оверлея + сторож зависання event loop
    gui_stall_ms: float = 40     # "зависання" event loop довше за це — фіксуємо
    gui_tick_ms: int = 10


# ==========================================================================
# Profiler — накопичувач замірів
# ==========================================================================

class Profiler:
    """Накопичує (кількість, сума, максимум) по іменованих етапах для ОДНОГО
    натискання хоткею і друкує таблицю в кінці. Потокобезпечний."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stats: dict[str, list] = {}
        self._marks: dict[str, float] = {}

    def reset(self) -> None:
        with self._lock:
            self._stats = {}
            self._marks = {}

    def add(self, name: str, seconds: float | None) -> None:
        if seconds is None:
            return
        with self._lock:
            s = self._stats.get(name)
            if s is None:
                s = self._stats[name] = [0, 0.0, 0.0]
            s[0] += 1
            s[1] += seconds
            s[2] = max(s[2], seconds)

    @contextmanager
    def timed(self, name: str):
        t = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - t)

    def mark(self, name: str) -> None:
        with self._lock:
            self._marks[name] = time.perf_counter()

    def since(self, name: str) -> float | None:
        with self._lock:
            t = self._marks.get(name)
        return None if t is None else time.perf_counter() - t

    def has_mark(self, name: str) -> bool:
        with self._lock:
            return name in self._marks

    def total(self, prefix: str) -> float:
        with self._lock:
            return sum(v[1] for k, v in self._stats.items() if k.startswith(prefix))

    def stats(self) -> dict[str, tuple[int, float, float]]:
        """Копія статистики: name -> (n, всього_с, макс_с). Зручно для тестів."""
        with self._lock:
            return {k: tuple(v) for k, v in self._stats.items()}

    def report(self, title: str) -> None:
        with self._lock:
            rows = sorted(self._stats.items(), key=lambda kv: kv[1][1], reverse=True)
        if not rows:
            return
        w = 44
        out = [f"\n┌── {title} " + "─" * 30,
               f"│ {'етап':<{w}}{'n':>5}{'всього мс':>11}{'сер. мс':>9}{'макс мс':>9}"]
        for name, (n, total, mx) in rows:
            out.append(f"│ {name:<{w}}{n:>5}{total * 1000:>11.1f}{total / n * 1000:>9.1f}{mx * 1000:>9.1f}")
        out.append("└" + "─" * (w + 36))
        print("\n".join(out), flush=True)


# ==========================================================================
# Інструменти обгортання
# ==========================================================================

Wrapper = Callable[[Callable], Callable]


def wrap_attr(obj: Any, name: str, make_wrapper: Wrapper) -> bool:
    """obj.name = make_wrapper(obj.name). Повторне обгортання тим самим
    профайлером ігнорується. Повертає True, якщо обгорнули."""
    original = getattr(obj, name)
    if getattr(original, "_profiled", False):
        return False
    wrapper = make_wrapper(original)
    wrapper._profiled = True
    wrapper.__wrapped__ = original
    setattr(obj, name, wrapper)
    return True


def timed(prof: Profiler, label: str | Callable[..., str]) -> Wrapper:
    """Обгортка, що міряє виклик. label — рядок або функція від аргументів
    виклику (наприклад, щоб додати довжину тексту)."""
    def make(original):
        def wrapper(*args, **kwargs):
            t = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                name = label(*args, **kwargs) if callable(label) else label
                prof.add(name, time.perf_counter() - t)
        return wrapper
    return make


def before(hook: Callable[..., None]) -> Wrapper:
    """Обгортка, що викликає hook(*args) перед оригіналом."""
    def make(original):
        def wrapper(*args, **kwargs):
            hook(*args, **kwargs)
            return original(*args, **kwargs)
        return wrapper
    return make


def print_duration(fmt: Callable[[float], str]) -> Wrapper:
    """Обгортка для разових операцій (завантаження моделі, відкриття
    мікрофона): друкує час у консоль, у таблицю натискання не потрапляє."""
    def make(original):
        def wrapper(*args, **kwargs):
            t = time.perf_counter()
            result = original(*args, **kwargs)
            print(f"[prof] {fmt(time.perf_counter() - t)}", flush=True)
            return result
        return wrapper
    return make


# ==========================================================================
# ONNX
# ==========================================================================

def install_onnx_profiling(prof: Profiler) -> None:
    """Обгортає onnxruntime.InferenceSession.run, щоб бачити, скільки часу
    займає кожна ONNX-модель (preprocessor / encoder / decoder_joint) і
    скільки разів її викликають. Ставиться ДО onnx_asr.load_model()."""
    import onnxruntime as ort  # noqa: PLC0415

    labels: dict[int, str] = {}

    def label_of(sess) -> str:
        key = id(sess)
        label = labels.get(key)
        if label is None:
            path = getattr(sess, "_model_path", None)
            if isinstance(path, (str, os.PathLike)):
                label = Path(path).name
            else:
                try:
                    label = "in-memory:" + sess.get_inputs()[0].name
                except Exception:  # noqa: BLE001
                    label = "in-memory:?"
            labels[key] = label
        return f"onnx:{label}"

    wrap_attr(ort.InferenceSession, "run", timed(prof, lambda sess, *a, **kw: label_of(sess)))


# ==========================================================================
# Сторож зависання Qt event loop
# ==========================================================================

class GuiStallMonitor:
    """QTimer кожні tick_ms у GUI-потоці. Якщо між тиками минуло помітно
    більше (GIL зайнятий ASR, важкий код у GUI-потоці) — фіксує паузу,
    але лише поки is_active() (йде обробка натискання)."""

    def __init__(self, prof: Profiler, is_active: Callable[[], bool],
                 threshold_ms: float = 40, tick_ms: int = 10) -> None:
        self.prof = prof
        self.is_active = is_active
        self.threshold_ms = threshold_ms
        self.tick_ms = tick_ms
        self._last = time.perf_counter()
        self._timer = None

    def start(self) -> None:
        from PySide6.QtCore import QTimer  # noqa: PLC0415

        self._last = time.perf_counter()
        self._timer = QTimer()
        self._timer.timeout.connect(self.tick)
        self._timer.start(self.tick_ms)

    def tick(self) -> None:
        now = time.perf_counter()
        gap = now - self._last - self.tick_ms / 1000
        self._last = now
        if gap * 1000 >= self.threshold_ms and self.is_active():
            self.prof.add("gui: event loop завис (GIL/блокування)", gap)


# ==========================================================================
# Діагностика рівнів гучності
# ==========================================================================

def describe_levels(cfg, audio: np.ndarray) -> str:
    """Рядок діагностики гучності: шум, мовлення, поріг паузи, найдовша
    пауза. Потрібен, щоб підібрати PAUSE_* під свій мікрофон. cfg — модуль
    застосунку (константи й функції VAD беруться звідти)."""
    dbfs = cfg.frame_dbfs(audio)
    if len(dbfs) == 0:
        return "рівні: (порожньо)"
    thr = cfg.pause_threshold(dbfs)
    longest = run = 0
    for quiet in dbfs < thr:
        run = run + 1 if quiet else 0
        longest = max(longest, run)
    p = cfg.PAUSE_NOISE_PERCENTILE
    return (f"рівні: шум(p{p}) {np.percentile(dbfs, p):.0f} dBFS, "
            f"медіана {np.median(dbfs):.0f}, мовлення(p90) {np.percentile(dbfs, 90):.0f}, "
            f"поріг паузи {thr:.0f} | найдовша пауза {longest * cfg.FRAME_SEC:.2f} с "
            f"(потрібно ≥ {cfg.COMMIT_PAUSE_SEC:.2f})"
            + (f" | гейт голосу {cfg.voice_gate_threshold(dbfs):.0f} dBFS" if cfg.VOICE_GATE else ""))


# ==========================================================================
# DictationProfiler — під'єднання до DictationApp
# ==========================================================================

# Функції модуля застосунку, які міряються підміною глобального імені.
MODULE_FUNCTIONS = {
    "apply_voice_gate": "vad: гейт голосу",
    "find_commit_point": "vad: пошук паузи",
    "should_skip_recognition": "vad: should_skip_recognition",
}


class DictationProfiler:
    def __init__(self, app, prof: Profiler | None = None,
                 options: ProfilerOptions | None = None) -> None:
        self.app = app
        self.prof = prof or Profiler()
        self.opt = options or ProfilerOptions()
        # Модуль, де визначено клас застосунку (__main__ при запуску скриптом).
        self.cfg = sys.modules[type(app).__module__]
        self._step_no = 0
        self._display_emitted: collections.deque[float] = collections.deque()
        self._slot_started: float | None = None
        self.stall_monitor: GuiStallMonitor | None = None

    def attach(self) -> "DictationProfiler":
        self._instrument_loading()
        if self.opt.onnx:
            install_onnx_profiling(self.prof)
        self._instrument_hotkeys()
        self._instrument_loop()
        self._instrument_audio()
        self._instrument_module_functions()
        self._instrument_asr()
        self._instrument_output()
        self._subscribe_events()
        if self.opt.gui:
            self._instrument_overlay()
            self.stall_monitor = GuiStallMonitor(
                self.prof, self.app.is_busy.is_set, self.opt.gui_stall_ms, self.opt.gui_tick_ms)
            self.stall_monitor.start()
        print("[prof] Профілювання УВІМКНЕНО — таблиця після кожного натискання.")
        return self

    # ---- разові операції ----------------------------------------------------

    def _instrument_loading(self) -> None:
        app = self.app
        wrap_attr(app, "_load_asr_model", print_duration(lambda s: f"ASR-модель завантажено за {s:.2f} с"))
        wrap_attr(app, "_load_translator", print_duration(lambda s: f"модель перекладу завантажено за {s:.2f} с"))
        wrap_attr(app.recorder, "open", print_duration(lambda s: f"мікрофон відкрито за {s * 1000:.0f} мс"))

    # ---- хоткеї й латентності ----------------------------------------------

    def _on_hook_press(self, *_a, **_kw) -> None:
        # Потік хуку. Не стираємо статистику натискання, що ще обробляється.
        if not self.app.is_busy.is_set():
            self.prof.reset()
            self._step_no = 0
        self.prof.mark("press_hook")

    def _on_hook_release(self, *_a, **_kw) -> None:
        self.prof.mark("release_hook")

    def _on_press(self, *_a, **_kw) -> None:
        self.prof.add("latency: хук → on_press (черга Qt)", self.prof.since("press_hook"))

    def _on_release(self, *_a, **_kw) -> None:
        self.prof.mark("on_release")
        self.prof.add("latency: хук → on_release (черга Qt)", self.prof.since("release_hook"))

    def _instrument_hotkeys(self) -> None:
        app = self.app
        wrap_attr(app, "_hook_press", before(self._on_hook_press))
        wrap_attr(app, "_hook_release", before(self._on_hook_release))
        wrap_attr(app, "on_press", before(self._on_press))
        wrap_attr(app, "on_release", before(self._on_release))

    # ---- цикл розпізнавання -------------------------------------------------

    def _instrument_loop(self) -> None:
        prof = self.prof

        def wait_step(original):
            def wrapper(*args, **kwargs):
                t = time.perf_counter()
                released = original(*args, **kwargs)
                prof.add("loop: очікування (простій)", time.perf_counter() - t)
                if released and not prof.has_mark("loop_saw_release"):
                    prof.mark("loop_saw_release")
                    prof.add("latency: on_release → цикл прокинувся", prof.since("on_release"))
                return released
            return wrapper

        def loop(original):
            def wrapper(*args, **kwargs):
                try:
                    return original(*args, **kwargs)
                finally:
                    self.finish_report()
            return wrapper

        wrap_attr(self.app, "_wait_step", wait_step)
        wrap_attr(self.app, "transcribe_loop", loop)

    def finish_report(self) -> None:
        """Похідний рядок: скільки часу ASR пішло НЕ на ONNX (Python-цикл
        декодера, numpy, обгортки onnx_asr). Потім друк таблиці."""
        asr_total = self.prof.total("asr: прохід")
        onnx_total = self.prof.total("onnx:")
        if asr_total > 0 and onnx_total > 0:
            self.prof.add("asr: поза ONNX (Python, numpy, декодер-цикл)", max(0.0, asr_total - onnx_total))
        self.prof.report("ПРОФІЛЬ НАТИСКАННЯ")

    # ---- аудіо та VAD -------------------------------------------------------

    def _instrument_audio(self) -> None:
        rec = self.app.recorder
        wrap_attr(rec, "start", timed(self.prof, "audio: recorder.start()"))
        wrap_attr(rec, "snapshot", timed(self.prof, "audio: snapshot()"))
        wrap_attr(rec, "stop", timed(self.prof, "audio: stop()"))

    def _instrument_module_functions(self) -> None:
        for func, label in MODULE_FUNCTIONS.items():
            if hasattr(self.cfg, func):
                wrap_attr(self.cfg, func, timed(self.prof, label))

    # ---- ASR, переклад, вивід ----------------------------------------------

    def _instrument_asr(self) -> None:
        prof, opt = self.prof, self.opt
        step_interval = getattr(self.cfg, "STEP_INTERVAL_SEC", None)
        sample_rate = getattr(self.cfg, "SAMPLE_RATE", 16_000)

        def asr_pass(original):
            def wrapper(segment, kind="крок", total_sec=None):
                # Значення за замовчуванням — ті самі, що в DictationApp.asr_pass:
                # короткий виклик asr_pass(window) не падає на обгортці.
                if total_sec is None:
                    total_sec = len(segment) / sample_rate
                t = time.perf_counter()
                result = original(segment, kind, total_sec)
                dt = time.perf_counter() - t
                prof.add(f"asr: прохід ({kind})", dt)
                self._step_no += 1
                if opt.per_step:
                    seg_sec = len(segment) / sample_rate
                    rtfx = seg_sec / dt if dt > 0 else 0.0
                    over = ""
                    if kind != "фінал" and step_interval and dt > step_interval:
                        over = f"  ⚠ довше за STEP_INTERVAL ({step_interval * 1000:.0f} мс)"
                    print(f"\n[prof] {kind} #{self._step_no}: вікно {seg_sec:5.2f} с (запис {total_sec:5.2f} с) → "
                          f"ASR {dt * 1000:6.0f} мс (RTFx {rtfx:5.1f}){over}", flush=True)
                return result
            return wrapper

        def confidence(original):
            def wrapper(*args, **kwargs):
                with prof.timed("asr: фільтр впевненості"):
                    noise, threshold = original(*args, **kwargs)
                if noise:
                    prof.add("asr: відкинуто фільтром впевненості", 0.0)
                return noise, threshold
            return wrapper

        wrap_attr(self.app, "asr_pass", asr_pass)
        wrap_attr(self.app, "_confidence_check", confidence)
        wrap_attr(self.app, "translate_text", timed(prof, "переклад OPUS-MT (torch)"))

    def _instrument_output(self) -> None:
        wrap_attr(self.app, "paste_text",
                  timed(self.prof, lambda text, *a, **kw: f"вивід: буфер + Ctrl+V ({len(text)} симв.)"))
        wrap_attr(self.app, "type_text",
                  timed(self.prof, lambda text, *a, **kw: f"вивід: typer.type ({len(text)} симв.)"))

    # ---- семантичні події застосунку ---------------------------------------

    def _subscribe_events(self) -> None:
        prof, ev = self.prof, self.app.events
        ev.subscribe("commit_fallback", lambda **_: prof.add("asr: фіксація взяла текст кроку", 0.0))
        ev.subscribe("final_skipped", lambda **_: prof.add("asr: фінал ПРОПУЩЕНО (взято текст кроку)", 0.0))
        ev.subscribe("partial_text", self._on_partial_text)
        ev.subscribe("final_audio", self._on_final_audio)
        ev.subscribe("final_done", lambda **_: prof.add("E2E: відпустили клавішу → текст надруковано",
                                                         prof.since("release_hook")))

    def _on_partial_text(self, text: str, **_) -> None:
        if text and not self.prof.has_mark("first_text"):
            self.prof.mark("first_text")
            self.prof.add("latency: натиск → перший проміжний текст", self.prof.since("press_hook"))

    def _on_final_audio(self, raw_audio, muted_share: float, **_) -> None:
        gate_info = f", заглушено фону {muted_share * 100:.0f}% запису" if getattr(self.cfg, "VOICE_GATE", False) else ""
        print(f"\n[prof] {describe_levels(self.cfg, raw_audio)}{gate_info}", flush=True)

    # ---- оверлей ------------------------------------------------------------

    def _instrument_overlay(self) -> None:
        ov = self.app.overlay
        emitted = self._display_emitted
        # Час emit() запам'ятовуємо ДО виклику: з GUI-потоку з'єднання пряме,
        # і display_started спрацює ще всередині display().
        wrap_attr(ov, "display", before(lambda *a, **kw: emitted.append(time.perf_counter())))
        ov.display_started.connect(self._on_display_started)
        ov.display_finished.connect(self._on_display_finished)

    def _on_display_started(self) -> None:
        now = time.perf_counter()
        if self._display_emitted:
            self.prof.add("gui: overlay сигнал у черзі", now - self._display_emitted.popleft())
        self._slot_started = now

    def _on_display_finished(self) -> None:
        if self._slot_started is not None:
            self.prof.add("gui: overlay _on_display (слот)", time.perf_counter() - self._slot_started)
            self._slot_started = None


def attach_profiler(app, prof: Profiler | None = None,
                    options: ProfilerOptions | None = None) -> DictationProfiler:
    """Під'єднує профайлер до DictationApp. Див. докстрінг модуля про порядок."""
    return DictationProfiler(app, prof, options).attach()
