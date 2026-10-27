"""Проганяє справжній transcribe_loop (через dictate()) з підробленими
мікрофоном, моделлю та оверлеєм і перевіряє, що duration = len(audio) /
SAMPLE_RATE завжди додатне для аудіо, яке доходить до розпізнавання."""
import os
import signal
import sys
import threading
import time
import types
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from App import UAFastConformer as uaf


class _Signal:
    """Мінімальна заміна Qt Signal: emit() одразу викликає підключені слоти."""

    def __init__(self):
        self.slots = []

    def connect(self, slot, *_args):
        self.slots.append(slot)

    def emit(self):
        for slot in self.slots:
            slot()


class _FakeBridge:
    instances = []

    def __init__(self):
        self.dictate_pressed = _Signal()
        self.dictate_released = _Signal()
        self.translate_toggled = _Signal()
        _FakeBridge.instances.append(self)


class _FakeTimer:
    def __init__(self):
        self.timeout = _Signal()

    def start(self, *_args):
        pass


class _FakeApp:
    @classmethod
    def instance(cls):
        return cls()

    def exec(self):
        return 0

    def quit(self):
        pass


class _FakeOverlay:
    calls = []

    def display(self, text, state=None, autohide_ms=None):
        _FakeOverlay.calls.append((text, state))

    def dismiss(self):
        pass

    def copy_text(self, text):
        pass


def run_dictation(monkeypatch, snapshot_audio, final_audio, hold_sec=0.15):
    """Імітує затискання й відпускання хоткея. Повертає (довжини аудіо, що
    дійшло до розпізнавання, набраний текст, виклики overlay, винятки потоку)."""
    recognized_lengths, typed, errors, threads = [], [], [], []
    _FakeBridge.instances.clear()
    _FakeOverlay.calls.clear()

    class FakeRecorder:
        def __init__(self, samplerate):
            pass

        def start(self):
            pass

        def snapshot(self):
            return snapshot_audio

        def stop(self):
            return final_audio

    class FakeModel:
        def with_timestamps(self):
            return self

        def recognize(self, audio, **_kwargs):
            recognized_lengths.append(len(audio))
            return SimpleNamespace(text="привіт", logprobs=[-0.1], tokens=None, timestamps=None)

    fake_keyboard = types.ModuleType("keyboard")
    fake_keyboard.write = typed.append
    fake_onnx = types.ModuleType("onnx_asr")
    fake_onnx.load_model = lambda *a, **k: FakeModel()
    fake_tf = types.ModuleType("transformers")
    fake_tf.MarianTokenizer = SimpleNamespace(from_pretrained=lambda *_: object())
    fake_tf.MarianMTModel = SimpleNamespace(from_pretrained=lambda *_: SimpleNamespace(eval=lambda: None))

    monkeypatch.setitem(sys.modules, "keyboard", fake_keyboard)
    monkeypatch.setitem(sys.modules, "onnx_asr", fake_onnx)
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)
    monkeypatch.setattr(signal, "signal", lambda *a, **k: None)
    monkeypatch.setattr(uaf, "QApplication", _FakeApp)
    monkeypatch.setattr(uaf, "QTimer", _FakeTimer)
    monkeypatch.setattr(uaf, "Qt", SimpleNamespace(QueuedConnection=0))
    monkeypatch.setattr(uaf, "Overlay", _FakeOverlay)
    monkeypatch.setattr(uaf, "HotkeyBridge", _FakeBridge)
    monkeypatch.setattr(uaf, "Recorder", FakeRecorder)
    monkeypatch.setattr(uaf, "register_hotkeys", lambda *a, **k: None)
    monkeypatch.setattr(uaf, "STEP_INTERVAL_SEC", 0.01)
    monkeypatch.setattr(threading, "excepthook", lambda args: errors.append(args.exc_value))

    real_thread = threading.Thread

    def tracking_thread(*args, **kwargs):
        thread = real_thread(*args, **kwargs)
        threads.append(thread)
        return thread

    monkeypatch.setattr(uaf.threading, "Thread", tracking_thread)

    assert uaf.dictate() == 0
    bridge = _FakeBridge.instances[-1]
    bridge.dictate_pressed.emit()
    time.sleep(hold_sec)
    bridge.dictate_released.emit()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive(), "transcribe_loop не завершився"

    return recognized_lengths, typed, list(_FakeOverlay.calls), errors


def _audio(seconds):
    return np.full(int(seconds * uaf.SAMPLE_RATE), 0.1, dtype=np.float32)


def test_duration_is_positive_for_every_recognized_chunk(monkeypatch):
    lengths, typed, _, errors = run_dictation(monkeypatch, _audio(1.0), _audio(1.0))
    assert not errors
    assert lengths, "розпізнавання жодного разу не викликалось"
    for n in lengths:
        duration = n / uaf.SAMPLE_RATE
        assert duration > 0
    assert typed == ["привіт"]


def test_zero_duration_audio_is_never_recognized(monkeypatch):
    empty = np.zeros(0, dtype=np.float32)
    lengths, typed, overlay_calls, errors = run_dictation(monkeypatch, empty, empty)
    assert not errors
    assert lengths == []  # нульова тривалість не доходить до моделі
    assert typed == []
    assert ("(занадто коротко)", "warning") in overlay_calls


def test_audio_shorter_than_min_seconds_is_not_recognized(monkeypatch):
    tiny = _audio(uaf.MIN_SECONDS / 4)
    lengths, typed, overlay_calls, errors = run_dictation(monkeypatch, tiny, tiny)
    assert not errors
    assert lengths == []
    assert typed == []
    assert ("(занадто коротко)", "warning") in overlay_calls


@pytest.mark.xfail(strict=True, reason="transcribe_loop не перевіряє audio is None: len(None) кидає TypeError")
def test_none_audio_does_not_crash_the_loop(monkeypatch):
    lengths, typed, _, errors = run_dictation(monkeypatch, None, None)
    assert not errors
    assert lengths == []
