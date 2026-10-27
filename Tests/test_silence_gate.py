import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from App.UAFastConformer import has_no_new_speech_since


def test_silence_and_no_new_audio_skips():
    assert has_no_new_speech_since(True, 16000, 16000) is True


def test_silence_but_audio_grew_recognizes():
    assert has_no_new_speech_since(True, 24000, 16000) is False


def test_previous_text_not_empty_recognizes():
    assert has_no_new_speech_since(False, 16000, 16000) is False


def test_first_step_recognizes():
    assert has_no_new_speech_since(False, 8000, 0) is False
