"""Тести App/speech_detection.py.

Швидкі тести — на фейковій моделі. Тести зі справжнім Silero VAD беруть
записи з Tests/clips/*.wav і пропускаються, якщо немає пакета silero-vad
або самих записів.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from App import speech_detection as sd
from App.CaptureDataset_ForTests.dataset_capture import read_wav

CLIPS = Path(__file__).parent / "clips"


class FakeSilero:
    """Ймовірність = середній модуль шматка × 10, і лічильник викликів."""

    def __init__(self):
        self.calls = 0
        self.resets = 0

    def __call__(self, chunk, sample_rate):
        self.calls += 1
        return min(1.0, float(chunk.abs().mean()) * 10)

    def reset_states(self):
        self.resets += 1


# ---- маски ---------------------------------------------------------------

def test_hysteresis_keeps_speech_between_thresholds():
    probs = np.array([0.1, 0.6, 0.4, 0.4, 0.2, 0.45])
    mask = sd.speech_mask_from_probabilities(probs, speech_threshold=0.5, silence_threshold=0.35)
    # 0.4 після мовлення ще мовлення; 0.45 після тиші — ще ні
    assert mask.tolist() == [False, True, True, True, False, False]


def test_widen_mask_pads_both_sides():
    mask = np.array([False, False, False, True, False, False, False])
    assert sd.widen_mask(mask, 1).tolist() == [False, False, True, True, True, False, False]


def test_widen_mask_shorter_than_kernel_keeps_length():
    mask = np.array([True, False])
    assert len(sd.widen_mask(mask, 5)) == 2


def test_chunk_mask_to_samples_tail_inherits_last_chunk():
    mask = np.array([False, True])
    samples = sd.chunk_mask_to_samples(mask, 2 * sd.VAD_CHUNK + 100)
    assert len(samples) == 2 * sd.VAD_CHUNK + 100
    assert not samples[: sd.VAD_CHUNK].any()
    assert samples[sd.VAD_CHUNK:].all()


def test_chunk_mask_to_samples_empty():
    assert not sd.chunk_mask_to_samples(np.zeros(0, dtype=bool), 300).any()


# ---- трекер ---------------------------------------------------------------

def test_tracker_processes_only_new_chunks():
    pytest.importorskip("torch")
    model = FakeSilero()
    tracker = sd.SpeechProbabilityTracker(model=model)
    audio = np.full(5 * sd.VAD_CHUNK + 10, 0.2, dtype=np.float32)
    assert len(tracker.update(audio[: 2 * sd.VAD_CHUNK])) == 2
    assert len(tracker.update(audio)) == 5          # неповний хвіст не рахується
    assert model.calls == 5                          # кожен шматок — рівно один раз


def test_tracker_resets_when_new_recording_is_shorter():
    pytest.importorskip("torch")
    model = FakeSilero()
    tracker = sd.SpeechProbabilityTracker(model=model)
    tracker.update(np.zeros(4 * sd.VAD_CHUNK, dtype=np.float32))
    assert len(tracker.update(np.zeros(sd.VAD_CHUNK, dtype=np.float32))) == 1
    assert model.resets == 1


def test_tracker_rejects_other_sample_rates():
    with pytest.raises(ValueError):
        sd.SpeechProbabilityTracker(sample_rate=8000, model=FakeSilero())


# ---- гейт гучності ---------------------------------------------------------

def _tone(db, seconds=2.0):
    t = np.arange(int(seconds * sd.VAD_SAMPLE_RATE)) / sd.VAD_SAMPLE_RATE
    return (10 ** (db / 20) * np.sqrt(2) * np.sin(2 * np.pi * 200 * t)).astype(np.float32)


def _rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)


def test_noise_gate_keeps_loud_and_mutes_quiet():
    pytest.importorskip("pedalboard")
    sr = sd.VAD_SAMPLE_RATE
    loud = sd.apply_noise_gate(_tone(-30), sr, threshold_db=-40, ratio=10, attack_ms=1, release_ms=100)
    quiet = sd.apply_noise_gate(_tone(-60), sr, threshold_db=-40, ratio=10, attack_ms=1, release_ms=100)
    assert abs(_rms_db(loud[sr:]) - _rms_db(_tone(-30)[sr:])) < 1.0
    assert _rms_db(quiet[sr:]) < -100


# ---- справжній Silero на записах ------------------------------------------

def _real_tracker():
    pytest.importorskip("silero_vad")
    return sd.SpeechProbabilityTracker()


def _speech_share(tracker, audio):
    probs = tracker.update(audio)
    return float(sd.speech_mask_from_probabilities(probs, 0.5, 0.35).mean())


@pytest.mark.parametrize("name", ["0001.wav", "0004.wav"])
def test_silero_finds_speech_in_recordings(name):
    path = CLIPS / name
    if not path.exists():
        pytest.skip(f"немає запису {path}")
    tracker = _real_tracker()
    audio, rate = read_wav(path)
    assert rate == sd.VAD_SAMPLE_RATE
    assert _speech_share(tracker, audio) > 0.4


@pytest.mark.parametrize("signal", ["silence", "white_noise", "hum_50hz"])
def test_silero_rejects_non_speech(signal):
    tracker = _real_tracker()
    n = 3 * sd.VAD_SAMPLE_RATE
    rng = np.random.default_rng(0)
    audio = {
        "silence": np.zeros(n, dtype=np.float32),
        "white_noise": (0.05 * rng.standard_normal(n)).astype(np.float32),
        "hum_50hz": (0.2 * np.sin(2 * np.pi * 50 * np.arange(n) / sd.VAD_SAMPLE_RATE)).astype(np.float32),
    }[signal]
    assert _speech_share(tracker, audio) < 0.05
