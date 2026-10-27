# test_recognize_committed_head.py
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

import App.UAFastConformer as m          # ← твій модуль
from App.UAFastConformer import DictationApp  # ← твій клас

CUT = 16000
HEAD = np.zeros(CUT, dtype=np.float32)


def make_self(asr_result="", asr_error=None):
    """Фейковий self: підмінюємо asr_pass і events."""
    fake = MagicMock()
    if asr_error:
        fake.asr_pass.side_effect = asr_error
    else:
        fake.asr_pass.return_value = asr_result
    return fake


def make_state(tail_text="", covered=CUT):
    return SimpleNamespace(tail_text=tail_text, samples_seen_by_last_step=lambda: covered)


def pick(fake_self, state):
    return DictationApp._recognize_committed_head(fake_self, HEAD, CUT, state, 1.0)


@pytest.fixture
def speech(monkeypatch):
    monkeypatch.setattr(m, "has_voiced_frames", lambda audio: True)


def test_returns_asr_text(speech):
    fake = make_self(asr_result="привіт")
    assert pick(fake, make_state(tail_text="інше")) == "привіт"
    fake.events.emit.assert_not_called()


def test_falls_back_to_step_text(speech):
    fake = make_self(asr_result="")
    assert pick(fake, make_state(tail_text="привіт", covered=CUT)) == "привіт"
    fake.events.emit.assert_called_once_with("commit_fallback")


def test_no_fallback_when_step_overran_pause(speech):
    fake = make_self(asr_result="")
    state = make_state(tail_text="привіт", covered=CUT + m.PAUSE_SAMPLES + 1)
    assert pick(fake, state) == ""


def test_no_fallback_without_step_text(speech):
    assert pick(make_self(asr_result=""), make_state(tail_text="")) == ""


def test_skips_asr_on_silence(monkeypatch):
    monkeypatch.setattr(m, "has_voiced_frames", lambda audio: False)
    fake = make_self()
    pick(fake, make_state())
    fake.asr_pass.assert_not_called()


def test_asr_error_propagates(speech):
    fake = make_self(asr_error=RuntimeError("ASR впав"))
    with pytest.raises(RuntimeError):
        pick(fake, make_state(tail_text="привіт"))