import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from App.UAFastConformer import (
    COMMIT_MAX_WINDOW_SEC,
    SAMPLE_RATE,
    SessionTranscript,
    choose_commit_cut,
    join_text_segments,
)


def _tone(sec: float, amp: float = 0.1) -> np.ndarray:
    t = np.arange(int(sec * SAMPLE_RATE)) / SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * 200 * t)).astype(np.float32)


def _silence(sec: float) -> np.ndarray:
    return np.zeros(int(sec * SAMPLE_RATE), dtype=np.float32)


def test_join_keeps_model_punctuation_and_case():
    parts = ["Привіт, світе.", "Що нового?", "Добре!"]
    assert join_text_segments(parts) == "Привіт, світе. Що нового? Добре!"


def test_join_does_not_remove_dot_or_change_case():
    assert join_text_segments(["рішення.", "Що прийняти"]) == "рішення. Що прийняти"


def test_join_skips_empty_parts():
    assert join_text_segments(["", " a ", "  ", "b"]) == "a b"
    assert join_text_segments(["", ""]) == ""


def test_transcript_full_text_and_commit():
    state = SessionTranscript()
    audio = np.concatenate([_tone(2), _silence(1)])
    state.commit_segment("Перше.", 16000, audio)
    state.commit_segment("", 8000, audio)   # порожній шматок нічого не додає
    state.commit_segment("Друге.", 8000, audio)
    assert state.committed_texts == ["Перше.", "Друге."]
    assert state.full_text("Хвіст") == "Перше. Друге. Хвіст"
    assert state.committed_samples == 32000


def test_cut_is_none_for_short_window():
    assert choose_commit_cut(_tone(0.5)) is None


def test_cut_returns_int_on_pause():
    cut = choose_commit_cut(np.concatenate([_tone(2), _silence(0.8), _tone(1)]))
    assert isinstance(cut, int)
    assert 2 * SAMPLE_RATE < cut < 2.8 * SAMPLE_RATE


def test_cut_is_none_without_pause_before_max_window():
    assert choose_commit_cut(_tone(5)) is None


def test_forced_cut_returns_int():
    assert isinstance(choose_commit_cut(_tone(COMMIT_MAX_WINDOW_SEC + 1)), int)
