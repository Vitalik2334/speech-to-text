"""
Parakeet-TDT-0.6b-v3: розпізнавання WAV-файлу.

Запуск:
    python 01_transcribe_file.py test.wav

Модель зберігається в локальній папці MODEL_DIR (див. нижче). Перший запуск
завантажить ваги туди (кілька хвилин), кожен наступний бере файли звідти
і взагалі не звертається в мережу — жодного "Fetching 5 files..." при
повторному старті скрипта.
"""

import sys
import wave
from pathlib import Path

import onnx_asr

DEFAULT_FILE = "test.wav"
MODEL = "nemo-parakeet-tdt-0.6b-v3"

# Локальна копія ваг моделі. Якщо папка вже існує — onnx_asr.load_model
# автоматично вмикає офлайн-режим: працює тільки з цими файлами, без
# жодного звернення до Hugging Face. Якщо змінити QUANTIZATION на значення,
# файлів під яке ще немає в цій папці, видаліть папку — вона перезавантажиться.
MODEL_DIR = Path(__file__).parent / "models" / MODEL

# "int8" — менший розмір і швидше, ціною ~1-2% WER.
# None — повна fp32-точність.
QUANTIZATION = None

# Пропускаємо VAD-етап, якщо файл коротший за це значення (секунди).
# Модель однаково приймає такий шматок за один прохід, тож VAD
# тут лише зайве навантаження без користі.
VAD_THRESHOLD_SEC = 25.0

# ["CUDAExecutionProvider", "CPUExecutionProvider"] — GPU без спроби TensorRT.
# None — стандартний список onnx-asr (TensorRT -> CUDA -> CPU).
PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        return w.getnframes() / rate if rate else 0.0


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) >= 2 else Path(DEFAULT_FILE)
    if not path.exists():
        print(f"Файл не знайдено: {path}")
        return 1

    first_run = not MODEL_DIR.exists()
    if first_run:
        print(f"Перший запуск: завантажую {MODEL} у {MODEL_DIR} ...")
    else:
        print(f"Завантаження моделі {MODEL} з локальної папки (офлайн) ...")

    # providers тут задає провайдер і для основної моделі, і для препроцесора —
    # onnx-asr сам підхоплює його в default_onnx_config. Якщо стоїть
    # CUDAExecutionProvider, пакет автоматично вмикає conv-based STFT
    # (use_conv_preprocessors=None -> auto) замість NumPy-версії на CPU,
    # щоб уникнути зайвого копіювання даних між CPU і GPU.
    # 1-2 "Memcpy nodes" у логах ONNX Runtime — це залишкове копіювання
    # довкола вузла STFT (у CUDA немає власного ядра для нього), не помилка
    # і не помітний вплив на швидкість.
    #
    # Другий позиційний аргумент (MODEL_DIR) — саме те, що вмикає локальний
    # кеш замість Hugging Face: якщо папка вже є, пакет ставить offline=True
    # і більше не ходить у мережу.
    model = onnx_asr.load_model(
        MODEL, MODEL_DIR, quantization=QUANTIZATION, providers=PROVIDERS
    )

    duration = wav_duration(path)
    print(f"Розпізнавання {path.name} ({duration:.1f}с) ...\n")

    if duration <= VAD_THRESHOLD_SEC:
        # Короткий файл — без VAD, менше накладних витрат.
        text = model.recognize(str(path))
        print(text)
    else:
        # Довгий файл — ріжемо на репліки через Silero VAD.
        vad = onnx_asr.load_vad("silero")
        for segment in model.with_vad(vad).recognize(str(path)):
            print(f"[{segment.start:7.2f}s - {segment.end:7.2f}s]  {segment.text}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())