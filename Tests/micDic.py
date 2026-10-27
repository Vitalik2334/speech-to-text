"""
Parakeet-TDT-0.6b-v3: безперервна диктовка з мікрофона.

Слухає мікрофон, сам визначає кінець репліки за паузою
і одразу друкує розпізнаний текст. Ctrl+C — вихід.

Запуск:
    python 02_mic_dictation.py
"""

import queue
import sys

import numpy as np
import sounddevice as sd

import onnx_asr

MODEL = "nemo-parakeet-tdt-0.6b-v3"
QUANTIZATION = None

SAMPLE_RATE = 16_000
BLOCK = 1600                  # 100 мс
SILENCE_RMS = 0.008           # поріг тиші; підберіть під свій мікрофон
SILENCE_BLOCKS = 8            # 0.8 с тиші = кінець репліки
MIN_SPEECH_BLOCKS = 3         # коротше 0.3 с — ігноруємо
MAX_SPEECH_BLOCKS = 200       # 20 с — межа моделі, ріжемо примусово

audio_q: "queue.Queue[np.ndarray]" = queue.Queue()


def on_audio(indata, frames, time_info, status):
    if status:
        print(status, file=sys.stderr)
    audio_q.put(indata[:, 0].copy())


def main() -> int:
    print("Пристрої вводу:")
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            print(f"  [{i}] {dev['name']}")
    print(f"\nЗавантаження моделі {MODEL} ...")
    model = onnx_asr.load_model(MODEL, quantization=QUANTIZATION)

    buffer: list[np.ndarray] = []
    silence_run = 0

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=BLOCK,
        callback=on_audio,
    )

    print("Готово. Говоріть. Ctrl+C — вихід.\n")
    with stream:
        try:
            while True:
                block = audio_q.get()
                rms = float(np.sqrt(np.mean(block**2)))
                speaking = rms > SILENCE_RMS

                if speaking:
                    buffer.append(block)
                    silence_run = 0
                elif buffer:
                    buffer.append(block)
                    silence_run += 1

                too_long = len(buffer) >= MAX_SPEECH_BLOCKS
                ended = buffer and (silence_run >= SILENCE_BLOCKS or too_long)

                if ended:
                    if len(buffer) >= MIN_SPEECH_BLOCKS:
                        waveform = np.concatenate(buffer)
                        text = model.recognize(waveform, sample_rate=SAMPLE_RATE)
                        if text and text.strip():
                            print(text.strip())
                    buffer = []
                    silence_run = 0
        except KeyboardInterrupt:
            print("\nЗупинено.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())