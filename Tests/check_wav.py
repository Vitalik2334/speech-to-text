"""
Перевіряє, чи придатний WAV-файл для onnx-asr / Parakeet.
Нічого не завантажує і не запускає модель — тільки аналіз файлу.

Запуск:
    python check_wav.py test.wav
"""

import sys
import wave
from pathlib import Path

import numpy as np

OK = "✔"
WARN = "⚠"
BAD = "✘"
file = "test.wav"
SUPPORTED_WIDTHS = {1: "PCM_U8", 2: "PCM_16", 3: "PCM_24", 4: "PCM_32"}


def main() -> int:


    path = Path(file)
    if not path.exists():
        print(f"{BAD} Файл не знайдено: {path}")
        return 1

    try:
        w = wave.open(str(path), "rb")
    except wave.Error as e:
        print(f"{BAD} Не WAV або стиснений кодек ({e}).")
        print("   Конвертуйте: ffmpeg -i вхід -ac 1 -ar 16000 -sample_fmt s16 test.wav")
        return 1

    with w:
        channels = w.getnchannels()
        width = w.getsampwidth()
        rate = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)

    duration = n / rate if rate else 0.0
    problems = 0

    print(f"Файл: {path.name}  ({path.stat().st_size / 1024:.0f} КБ)\n")

    subtype = SUPPORTED_WIDTHS.get(width)
    if subtype:
        print(f"{OK} Формат семплів: {subtype}")
    else:
        print(f"{BAD} Непідтримувана розрядність: {width * 8} біт")
        problems += 1

    if rate == 16000:
        print(f"{OK} Частота: {rate} Гц")
    else:
        print(f"{WARN} Частота: {rate} Гц — onnx-asr передискретизує сам, але 16000 краще")

    if channels == 1:
        print(f"{OK} Канали: моно")
    else:
        print(f"{WARN} Канали: {channels} — передайте channel='mean' або зведіть у моно")

    if duration == 0:
        print(f"{BAD} Тривалість: 0 с — файл порожній")
        problems += 1
    elif duration <= 30:
        print(f"{OK} Тривалість: {duration:.2f} с")
    else:
        print(f"{WARN} Тривалість: {duration:.2f} с — довше 30 с, потрібен .with_vad(...)")

    if width == 2 and raw:
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)
        peak = float(np.abs(samples).max())
        rms = float(np.sqrt(np.mean(samples**2)))
        print(f"\n   пік {peak:.3f}   RMS {rms:.4f}")

        if peak < 0.02:
            print(f"{BAD} Практично тиша — мікрофон вимкнений або не той пристрій")
            problems += 1
        elif peak < 0.05:
            print(f"{WARN} Дуже тихо — розпізнавання буде поганим")
        elif peak > 0.995:
            print(f"{WARN} Кліпінг — знизьте гучність мікрофона")
        else:
            print(f"{OK} Рівень сигналу нормальний")

    print()
    if problems:
        print(f"{BAD} Файл не готовий: проблем — {problems}")
        return 1

    print(f"{OK} Файл придатний. Далі: python 01_transcribe_file.py {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
