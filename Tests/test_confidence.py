import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from App.UAFastConformer import average_logprob, is_low_confidence


def test_high_confidence_is_not_noise():
    assert is_low_confidence([-0.1, -0.2, -0.3], -1.0) is False


def test_low_confidence_is_noise():
    assert is_low_confidence([-2.0, -1.5, -3.0], -1.0) is True


def test_exactly_threshold_is_not_noise():
    assert is_low_confidence([-1.0, -1.0], -1.0) is False


def test_mixed_tokens_use_average():
    # одне невпевнене слово серед впевнених не робить результат шумом
    assert is_low_confidence([-0.1, -0.1, -0.1, -2.5], -1.0) is False


def test_none_or_empty_is_not_noise():
    assert is_low_confidence(None, -1.0) is False
    assert is_low_confidence([], -1.0) is False


def test_average_logprob():
    assert average_logprob([-1.0, -3.0]) == -2.0
    assert average_logprob(None) is None
    assert average_logprob([]) is None
