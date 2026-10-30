# Повторне розпізнавання ВСЬОГО запису після відпускання клавіші.
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

import App.UAFastConformer as m
from App.UAFastConformer import DictationApp

AUDIO = np.zeros(16000, dtype=np.float32)


def make_state(tail_text="", committed=("Перше.", "Друге."), tail_valid=False):
    return SimpleNamespace(
        tail_text=tail_text, tail_valid=tail_valid, last_audio_len=len(AUDIO), speech_mask=None,
        committed_samples=0,
        speech_mask_slice=lambda *a: None,
        full_text=lambda tail="": " ".join([*committed, tail] if tail else committed),
    )


def make_app(asr_results=None, asr_error=None):
    """Фейковий self зі справжніми допоміжними методами DictationApp."""
    app = MagicMock()
    if asr_error:
        app.asr_pass.side_effect = asr_error
    else:
        app.asr_pass.side_effect = list(asr_results or [])
    app.translate_if_enabled.side_effect = lambda text: text
    app._recognize_full_recording = lambda a, d: DictationApp._recognize_full_recording(app, a, d)
    app._recognize_final_tail = lambda a, w, s, d: DictationApp._recognize_final_tail(app, a, w, s, d)
    return app


def finish(app, state):
    DictationApp._finish_session(app, AUDIO, AUDIO, 0.0, AUDIO, state, 1.0)


@pytest.fixture
def speech(monkeypatch):
    monkeypatch.setattr(m, "has_voiced_frames", lambda *a, **k: True)
    monkeypatch.setattr(m, "has_no_new_speech_since", lambda *a, **k: False)


def test_full_pass_replaces_stitched_text(speech):
    app = make_app(["Повний текст, виправлений."])
    finish(app, make_state(tail_text="хвіст"))
    app.asr_pass.assert_called_once()
    assert app.asr_pass.call_args.args[0] is AUDIO
    assert app.asr_pass.call_args.args[1] == "фінал-повний"
    app.insert_text_into_active_window.assert_called_once_with("Повний текст, виправлений.")
    app.events.emit.assert_any_call("full_pass_done", text="Повний текст, виправлений.")


def test_full_pass_has_no_length_limit(speech):
    long_audio = np.zeros(16000 * 90, dtype=np.float32)   # 90 с
    app = make_app(["довгий текст"])
    DictationApp._finish_session(app, long_audio, long_audio, 0.0, long_audio, make_state(), 90.0)
    assert app.asr_pass.call_args.args[0] is long_audio
    app.insert_text_into_active_window.assert_called_once_with("довгий текст")


def test_empty_full_pass_falls_back_to_tail(speech):
    app = make_app(["", "хвіст"])           # повний прохід порожній, потім хвіст
    finish(app, make_state())
    assert app.asr_pass.call_count == 2
    assert app.asr_pass.call_args_list[1].args[1] == "фінал"
    app.insert_text_into_active_window.assert_called_once_with("Перше. Друге. хвіст")


def test_full_pass_error_falls_back_to_committed_text(speech):
    app = make_app(asr_error=RuntimeError("boom"))
    finish(app, make_state(tail_text="крок"))
    app.insert_text_into_active_window.assert_called_once_with("Перше. Друге. крок")
    assert all(c.args[0] != "full_pass_done" for c in app.events.emit.call_args_list)


def test_no_speech_skips_full_pass(monkeypatch):
    monkeypatch.setattr(m, "has_voiced_frames", lambda *a, **k: False)
    monkeypatch.setattr(m, "has_no_new_speech_since", lambda *a, **k: True)
    app = make_app([])
    finish(app, make_state(tail_text="крок", tail_valid=True))
    app.asr_pass.assert_not_called()
    app.insert_text_into_active_window.assert_called_once_with("Перше. Друге. крок")
