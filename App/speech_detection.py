"""
speech_detection.py — нейромережевий детектор мовлення і гейт гучності.

  * SpeechProbabilityTracker — Silero VAD v6 (пакет silero-vad, режим ONNX):
    ймовірність мовлення для кожних 32 мс запису. Рахується інкрементно:
    на кожному кроці лише нові шматки, а не весь запис наново.
  * speech_mask_from_probabilities — рішення "мовлення / ні" з гістерезисом
    (як у самого Silero): мовлення починається з порогу speech_threshold,
    а закінчується лише нижче silence_threshold.
  * apply_noise_gate — гейт гучності з pedalboard (Spotify, на JUCE).

Важкі бібліотеки (torch, silero_vad, pedalboard) імпортуються лише під
час першого використання: з USE_VAD = False вони не потрібні взагалі.

PyInstaller: модель Silero лежить усередині пакета, тому при збірці
потрібні --collect-data silero_vad і --collect-all pedalboard.
"""
from __future__ import annotations

import numpy as np

VAD_SAMPLE_RATE = 16_000
VAD_CHUNK = 512          # рідний розмір вікна Silero при 16 кГц (32 мс)


def _load_silero_onnx():
    from silero_vad import load_silero_vad  # noqa: PLC0415

    return load_silero_vad(onnx=True)


class SpeechProbabilityTracker:
    """Ймовірності мовлення для запису, що росте (одне натискання хоткея).

    update(audio) отримує ВЕСЬ запис від початку натискання і дораховує
    лише нові повні шматки по VAD_CHUNK семплів. Silero — рекурентна
    модель, тому перед новим записом обов'язково reset()."""

    def __init__(self, sample_rate: int = VAD_SAMPLE_RATE, model=None) -> None:
        if sample_rate != VAD_SAMPLE_RATE:
            raise ValueError(f"Silero VAD тут працює лише з {VAD_SAMPLE_RATE} Гц, а не {sample_rate}")
        self.sample_rate = sample_rate
        self._model = model if model is not None else _load_silero_onnx()
        self._probs: list[float] = []
        self._processed = 0      # скільки семплів запису вже пройшло через модель

    def reset(self) -> None:
        self._model.reset_states()
        self._probs = []
        self._processed = 0

    def update(self, audio: np.ndarray) -> np.ndarray:
        """Дораховує нові шматки й повертає ймовірності для всіх повних шматків."""
        import torch  # noqa: PLC0415

        if len(audio) < self._processed:    # запис коротший за оброблене — це вже інший запис
            self.reset()
        while self._processed + VAD_CHUNK <= len(audio):
            chunk = np.ascontiguousarray(audio[self._processed:self._processed + VAD_CHUNK], dtype=np.float32)
            self._probs.append(float(self._model(torch.from_numpy(chunk), self.sample_rate)))
            self._processed += VAD_CHUNK
        return self.probabilities()

    def probabilities(self) -> np.ndarray:
        return np.asarray(self._probs, dtype=np.float32)


def speech_mask_from_probabilities(probs: np.ndarray, speech_threshold: float,
                                   silence_threshold: float) -> np.ndarray:
    """Мовлення чи ні для кожного шматка, з гістерезисом: починається при
    p >= speech_threshold, триває, доки p >= silence_threshold."""
    mask = np.zeros(len(probs), dtype=bool)
    in_speech = False
    for i, p in enumerate(probs):
        in_speech = p >= silence_threshold if in_speech else p >= speech_threshold
        mask[i] = in_speech
    return mask


def widen_mask(mask: np.ndarray, pad_chunks: int) -> np.ndarray:
    """Розширює кожну ділянку мовлення на pad_chunks шматків в обидва боки,
    щоб не обрізати тихі приголосні на краях слів."""
    if pad_chunks <= 0 or not mask.any():
        return mask.copy()
    kernel = np.ones(2 * pad_chunks + 1, dtype=np.int32)
    # mode="full" + зріз, а не "same": коли ядро довше за маску, "same"
    # повертає масив довжини ядра (та сама пастка, що в apply_voice_gate).
    full = np.convolve(mask.astype(np.int32), kernel, mode="full")
    return full[pad_chunks:pad_chunks + len(mask)] > 0


def chunk_mask_to_samples(mask: np.ndarray, n_samples: int) -> np.ndarray:
    """Маска по шматках → маска по семплах довжини n_samples. Хвіст, який
    VAD ще не бачив (< VAD_CHUNK семплів), успадковує стан останнього шматка."""
    samples = np.repeat(mask, VAD_CHUNK)[:n_samples]
    if len(samples) < n_samples:
        tail_value = bool(mask[-1]) if len(mask) else False
        samples = np.concatenate([samples, np.full(n_samples - len(samples), tail_value)])
    return samples


def apply_noise_gate(audio: np.ndarray, sample_rate: int, threshold_db: float, ratio: float,
                     attack_ms: float, release_ms: float) -> np.ndarray:
    """Гейт гучності pedalboard.NoiseGate: глушить усе тихіше за threshold_db."""
    import pedalboard  # noqa: PLC0415

    gate = pedalboard.NoiseGate(threshold_db=threshold_db, ratio=ratio,
                                attack_ms=attack_ms, release_ms=release_ms)
    return gate(np.ascontiguousarray(audio, dtype=np.float32), sample_rate)
