"""
Тести для profiling.py без Qt, мікрофона й моделей.

FakeApp має той самий "контракт", що й DictationApp (імена методів,
recorder, events, is_busy), і живе у фейковому модулі fake_dictation —
туди ж attach_profiler підставляє обгортки функцій VAD.

Запуск:  uv run pytest -q test_profiling.py
"""
from __future__ import annotations

import sys
import threading
import types
from collections import defaultdict

import numpy as np
import pytest

import profiling
from profiling import ProfilerOptions, Profiler, attach_profiler, wrap_attr, timed


# --------------------------------------------------------------------------
# Фейковий модуль застосунку
# --------------------------------------------------------------------------

fake_mod = types.ModuleType("fake_dictation")
fake_mod.STEP_INTERVAL_SEC = 0.3
fake_mod.SAMPLE_RATE = 16_000
fake_mod.VOICE_GATE = False
fake_mod.apply_voice_gate = lambda audio: (audio, 0.0, 0.0)
fake_mod.choose_commit_cut = lambda window: None
fake_mod.has_no_new_speech_since = lambda audio, last: False
sys.modules["fake_dictation"] = fake_mod


class Events:
    def __init__(self):
        self._subs = defaultdict(list)

    def subscribe(self, name, handler):
        self._subs[name].append(handler)

    def emit(self, name, **payload):
        for h in self._subs.get(name, ()):
            h(**payload)


class FakeRecorder:
    def open(self): pass
    def start(self): pass
    def snapshot(self): return np.zeros(10, dtype=np.float32)
    def stop(self): return np.zeros(10, dtype=np.float32)


class FakeApp:
    def __init__(self):
        self.events = Events()
        self.recorder = FakeRecorder()
        self.is_busy = threading.Event()
        self.noise = False
        self.released = False

    def _load_asr_model(self): return "asr"
    def _load_translator(self): return "tr"
    def _hook_press(self): pass
    def _hook_release(self): pass
    def on_press(self): pass
    def on_release(self): pass
    def _wait_for_next_step(self, started): return self.released
    def run_dictation_session(self): self.events.emit("final_done")
    def asr_pass(self, segment, kind, total_sec): return "текст"
    def _classify_by_confidence(self, text, logprobs): return self.noise, -0.4
    def translate_text(self, text): return text
    def paste_via_clipboard(self, text): pass
    def type_via_keystrokes(self, text): pass


FakeApp.__module__ = "fake_dictation"

OPTS = ProfilerOptions(onnx=False, gui=False, per_step=False)


@pytest.fixture
def app():
    return FakeApp()


@pytest.fixture(autouse=True)
def restore_module_functions(monkeypatch):
    # Обгортки функцій модуля не мають "протікати" між тестами.
    for name in profiling.MODULE_FUNCTIONS:
        monkeypatch.setattr(fake_mod, name, getattr(fake_mod, name))


@pytest.fixture
def prof(app):
    p = Profiler()
    attach_profiler(app, p, OPTS)
    return p


# --------------------------------------------------------------------------
# Profiler
# --------------------------------------------------------------------------

def test_profiler_add_and_stats():
    p = Profiler()
    p.add("a", 0.1)
    p.add("a", 0.3)
    p.add("b", None)          # None ігнорується
    n, total, mx = p.stats()["a"]
    assert n == 2 and total == pytest.approx(0.4) and mx == pytest.approx(0.3)
    assert "b" not in p.stats()


def test_profiler_marks_and_reset():
    p = Profiler()
    assert p.since("x") is None and not p.has_mark("x")
    p.mark("x")
    assert p.has_mark("x") and p.since("x") >= 0
    p.add("a", 1.0)
    p.reset()
    assert p.stats() == {} and not p.has_mark("x")


def test_profiler_report_prints_table(capsys):
    p = Profiler()
    p.add("етап", 0.012)
    p.report("ТЕСТ")
    out = capsys.readouterr().out
    assert "ТЕСТ" in out and "етап" in out


def test_report_empty_prints_nothing(capsys):
    Profiler().report("ТЕСТ")
    assert capsys.readouterr().out == ""


def test_wrap_attr_is_idempotent():
    p = Profiler()
    obj = types.SimpleNamespace(f=lambda: 1)
    assert wrap_attr(obj, "f", timed(p, "f")) is True
    assert wrap_attr(obj, "f", timed(p, "f")) is False
    assert obj.f() == 1
    assert p.stats()["f"][0] == 1   # не подвійний замір


def test_timed_counts_even_on_exception():
    p = Profiler()

    def boom():
        raise RuntimeError

    obj = types.SimpleNamespace(f=boom)
    wrap_attr(obj, "f", timed(p, "f"))
    with pytest.raises(RuntimeError):
        obj.f()
    assert p.stats()["f"][0] == 1


# --------------------------------------------------------------------------
# attach_profiler
# --------------------------------------------------------------------------

def test_class_untouched_only_instance_wrapped(app, prof):
    assert hasattr(app.asr_pass, "__wrapped__")
    assert not hasattr(FakeApp.asr_pass, "__wrapped__")
    assert not hasattr(FakeApp().asr_pass, "__wrapped__")


def test_press_latency_and_reset(app, prof):
    prof.add("старе", 1.0)
    app._hook_press()                 # не busy → скидання
    app.on_press()
    stats = prof.stats()
    assert "старе" not in stats
    assert "latency: хук → on_press (черга Qt)" in stats


def test_no_reset_while_busy(app, prof):
    prof.add("старе", 1.0)
    app.is_busy.set()
    app._hook_press()
    assert "старе" in prof.stats()


def test_release_latency_and_loop_wakeup(app, prof):
    app._hook_release()
    app.on_release()
    app.released = True
    app._wait_for_next_step(0.0)
    app._wait_for_next_step(0.0)               # другий раз латентність не додається
    stats = prof.stats()
    assert stats["latency: хук → on_release (черга Qt)"][0] == 1
    assert stats["latency: on_release → цикл прокинувся"][0] == 1
    assert stats["loop: очікування (простій)"][0] == 2


def test_asr_pass_by_kind(app, prof):
    seg = np.zeros(16_000, dtype=np.float32)
    assert app.asr_pass(seg, "крок", 1.0) == "текст"
    app.asr_pass(seg, "крок", 1.5)
    app.asr_pass(seg, "фінал", 2.0)
    stats = prof.stats()
    assert stats["asr: прохід (крок)"][0] == 2
    assert stats["asr: прохід (фінал)"][0] == 1


def test_per_step_line_printed(app, capsys):
    p = Profiler()
    attach_profiler(app, p, ProfilerOptions(onnx=False, gui=False, per_step=True))
    app.asr_pass(np.zeros(8_000, dtype=np.float32), "крок", 0.5)
    out = capsys.readouterr().out
    assert "[prof] крок #1: вікно  0.50 с" in out


def test_confidence_rejected_counted(app, prof):
    app.noise = True
    assert app._classify_by_confidence("а", None) == (True, -0.4)
    stats = prof.stats()
    assert stats["asr: відкинуто фільтром впевненості"][0] == 1
    assert stats["asr: фільтр впевненості"][0] == 1


def test_output_label_has_length(app, prof):
    app.paste_via_clipboard("привіт")
    app.type_via_keystrokes("ok")
    stats = prof.stats()
    assert "вивід: буфер + Ctrl+V (6 симв.)" in stats
    assert "вивід: typer.type (2 симв.)" in stats


def test_module_functions_wrapped(app, prof):
    fake_mod.choose_commit_cut(np.zeros(1))
    fake_mod.apply_voice_gate(np.zeros(1))
    stats = prof.stats()
    assert "vad: пошук паузи" in stats and "vad: гейт голосу" in stats


def test_events(app, prof):
    app._hook_press()
    app.events.emit("partial_text", text="")
    app.events.emit("partial_text", text="перший")
    app.events.emit("partial_text", text="другий")
    app.events.emit("commit_fallback")
    app.events.emit("final_skipped")
    stats = prof.stats()
    assert stats["latency: натиск → перший проміжний текст"][0] == 1
    assert stats["asr: фіксація взяла текст кроку"][0] == 1
    assert stats["asr: фінал ПРОПУЩЕНО (взято текст кроку)"][0] == 1


def test_run_dictation_session_prints_report(app, prof, capsys):
    app._hook_release()
    app.run_dictation_session()
    out = capsys.readouterr().out
    assert "ПРОФІЛЬ НАТИСКАННЯ" in out
    assert "E2E: відпустили клавішу → текст надруковано" in out


def test_report_derives_outside_onnx(app, prof):
    prof.add("asr: прохід (крок)", 0.5)
    prof.add("onnx:encoder-model.onnx", 0.3)
    app.run_dictation_session()
    # reset() у report не робиться — рядок лишається у статистиці
    assert prof.stats()["asr: поза ONNX (Python, numpy, декодер-цикл)"][1] == pytest.approx(0.2)


def test_loading_prints_duration(app, capsys):
    attach_profiler(app, Profiler(), OPTS)
    assert app._load_asr_model() == "asr"
    app.recorder.open()
    out = capsys.readouterr().out
    assert "ASR-модель завантажено" in out and "мікрофон відкрито" in out


def test_stall_monitor_only_when_active():
    p = Profiler()
    active = {"v": False}
    m = profiling.GuiStallMonitor(p, lambda: active["v"], threshold_ms=40, tick_ms=10)
    m._last -= 0.2
    m.tick()
    assert p.stats() == {}
    active["v"] = True
    m._last -= 0.2
    m.tick()
    assert "gui: event loop завис (GIL/блокування)" in p.stats()
