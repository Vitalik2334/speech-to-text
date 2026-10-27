"""Справжній цикл диктування (run_dictation_session) з USE_VAD = True і False.

Модель розпізнавання — фейкова: записує, яке аудіо їй прийшло, і повертає
"текст", якщо в ньому є енергія. Мікрофон — повтор сигналу кроками по
STEP_INTERVAL_SEC. Потрібні пакети silero-vad і pedalboard; тести з
вашими записами пропускаються, якщо немає Tests/clips/0004.wav.
"""
from __future__ import annotations

import contextlib
import io
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("silero_vad")
pytest.importorskip("pedalboard")

import App.UAFastConformer as m  # noqa: E402
from App.CaptureDataset_ForTests.dataset_capture import read_wav  # noqa: E402

SR = m.SAMPLE_RATE
CLIP = Path(__file__).parent / "clips" / "0004.wav"


class FakeAsr:
    def __init__(self):
        self.inputs: list[np.ndarray] = []

    def recognize(self, audio, **_):
        self.inputs.append(np.asarray(audio).copy())
        rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0
        text = "" if rms < 1e-4 else f"текст{len(self.inputs)}."
        return type("R", (), {"text": text, "logprobs": [-0.1]})()


class Replay:
    def __init__(self, audio):
        self.audio, self.pos = audio, 0

    def start(self): ...
    def open(self): ...
    def close(self): ...

    def snapshot(self):
        return self.audio[: self.pos].copy()

    def stop(self):
        return self.audio[: self.pos].copy()


class Overlay:
    def __getattr__(self, _name):
        return lambda *a, **k: True


def run_session(audio: np.ndarray, use_vad: bool, monkeypatch) -> tuple[FakeAsr, list[str]]:
    monkeypatch.setattr(m, "USE_VAD", use_vad)
    app = m.DictationApp()
    asr, delivered = FakeAsr(), []
    rec = Replay(audio)
    step = int(m.STEP_INTERVAL_SEC * SR)

    def wait(_started):
        rec.pos = min(len(audio), rec.pos + step)
        if rec.pos >= len(audio):
            app.is_recording.clear()
            return True
        return False

    app.recorder, app.overlay, app.ts_model = rec, Overlay(), asr
    app._wait_for_next_step = wait
    app.insert_text_into_active_window = delivered.append
    app.is_recording.set()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        app.run_dictation_session()
    return asr, delivered


def loud_noise(seconds, db=-20.0):
    rng = np.random.default_rng(0)
    return (10 ** (db / 20) * rng.standard_normal(int(seconds * SR))).astype(np.float32)


def test_loud_noise_never_reaches_asr_with_vad(monkeypatch):
    asr, delivered = run_session(loud_noise(3.0), use_vad=True, monkeypatch=monkeypatch)
    assert asr.inputs == []
    assert delivered == []


def test_loud_noise_reaches_asr_without_vad(monkeypatch):
    """Для порівняння: стара логіка за гучністю пропускає гучний шум у модель."""
    asr, _ = run_session(loud_noise(3.0), use_vad=False, monkeypatch=monkeypatch)
    assert asr.inputs


@pytest.fixture
def recording():
    if not CLIP.exists():
        pytest.skip(f"немає запису {CLIP}")
    audio, rate = read_wav(CLIP)
    assert rate == SR
    return audio


def test_real_speech_is_recognized_with_vad(recording, monkeypatch):
    asr, delivered = run_session(recording, use_vad=True, monkeypatch=monkeypatch)
    assert asr.inputs
    assert delivered and delivered[0]


def test_noise_between_phrases_is_zeroed_before_asr(recording, monkeypatch):
    noise = loud_noise(1.5, db=-25.0)
    audio = np.concatenate([recording[: 4 * SR], noise, recording[4 * SR:]])
    asr, delivered = run_session(audio, use_vad=True, monkeypatch=monkeypatch)
    assert delivered
    # Усе, що отримала модель, разом: частка семплів шуму, що пройшли, мізерна.
    seen = np.concatenate(asr.inputs)
    assert np.mean(np.abs(seen) > 0) < 0.95     # обнулені ділянки справді є
    # Silero після мовлення ще ~0.4-0.6 с тримає стан "мовлення" (відома
    # затримка кінця фрази), тому шум одразу після слова частково проходить.
    # Далі, після цієї затримки, шум має бути відкинутий.
    gated, _raw, _share, mask = run_gate_only(audio, monkeypatch)
    hangover = int(0.8 * SR)
    noise_after_hangover = mask[4 * SR + hangover: 4 * SR + len(noise) - int(0.2 * SR)]
    assert noise_after_hangover.mean() < 0.05


def run_gate_only(audio, monkeypatch):
    monkeypatch.setattr(m, "USE_VAD", True)
    app = m.DictationApp()
    app.recorder = Replay(audio)
    app.recorder.pos = len(audio)
    app._start_speech_tracking()
    return app._capture_gated_audio(released=True)
