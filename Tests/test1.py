"""
Parakeet-TDT-0.6b-v3: push-to-talk диктовка на затиснуту клавішу.

Модель завантажується ОДИН раз при старті й лишається в пам'яті, поки
скрипт працює (це і є "фоновий режим" з попереднього питання). Далі:

    затиснули HOTKEY -> йде запис з мікрофона
    відпустили      -> аудіо розпізнається й текст:
                          - друкується в консоль
                          - копіюється в буфер обміну
                          - вводиться в те вікно, яке зараз у фокусі
                            (AUTO_TYPE, як звичайна клавіатура)

Встановлення:
    pip install keyboard

Запуск:
    python 03_hotkey_dictation.py

Ctrl+C - вихід. Клавішу міняйте через HOTKEY нижче
(наприклад "f9", "right ctrl", "caps lock", "pause").

Примітка Windows: якщо гарячі клавіші взагалі не реагують - спробуйте
запустити консоль/IDE від імені адміністратора. Зазвичай не потрібно,
але деякі системи блокують глобальні хуки клавіатури для звичайного
користувача.
"""

import sys
import threading
import tkinter as tk
from pathlib import Path

import keyboard
import numpy as np
import sounddevice as sd

import onnx_asr

MODEL = "nemo-parakeet-tdt-0.6b-v3"
MODEL_DIR = Path(__file__).parent / "models" / MODEL

QUANTIZATION = None
PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]


# "shift" ловить і лівий, і правий Shift разом. Якщо Shift активно
# використовується для звичайного набору тексту (великі літери) і
# трапляються випадкові спрацювання — поставте "right shift" або
# "left shift", щоб слухати тільки одну конкретну клавішу.
HOTKEY = "shift"

SAMPLE_RATE = 16_000
MAX_SECONDS = 28.0          # запобіжник під ліміт моделі ~20-30с за прохід
MIN_SECONDS = 0.2           # коротше — вважаємо випадковим натисканням

AUTO_TYPE = True            # вводити текст у активне вікно
COPY_TO_CLIPBOARD = True    # дублювати текст у буфер обміну


class Recorder:
    """Запис з мікрофона між start() і stop() у власному потоці callback'ів."""

    def __init__(self, samplerate: int):
        self.samplerate = samplerate
        self._frames: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._stream is not None:
                return
            self._frames = []
            self._stream = sd.InputStream(
                samplerate=self.samplerate,
                channels=1,
                dtype="float32",
                callback=self._callback,
            )
            self._stream.start()

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            print(status, file=sys.stderr)
        self._frames.append(indata[:, 0].copy())

    def stop(self) -> np.ndarray:
        with self._lock:
            if self._stream is None:
                return np.zeros(0, dtype=np.float32)
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if not self._frames:
            return np.zeros(0, dtype=np.float32)

        audio = np.concatenate(self._frames)
        max_len = int(MAX_SECONDS * self.samplerate)
        return audio[:max_len]


def copy_to_clipboard(text: str) -> None:
    root = tk.Tk()
    root.withdraw()
    root.clipboard_clear()
    root.clipboard_append(text)
    root.update()
    root.destroy()


def main() -> int:
    first_run = not MODEL_DIR.exists()
    if first_run:
        print(f"Перший запуск: завантажую {MODEL} у {MODEL_DIR} ...")
    else:
        print(f"Завантаження моделі {MODEL} з локальної папки (офлайн) ...")

    model = onnx_asr.load_model(
        MODEL, MODEL_DIR, quantization=QUANTIZATION, providers=PROVIDERS
    )

    recorder = Recorder(SAMPLE_RATE)
    is_recording = threading.Event()
    is_busy = threading.Event()  # блокує повторний старт, поки йде розпізнавання

    def on_press(_event) -> None:
        if is_recording.is_set() or is_busy.is_set():
            return
        is_recording.set()
        recorder.start()
        print("🎙 запис...", end="", flush=True)

    def on_release(_event) -> None:
        if not is_recording.is_set():
            return
        is_recording.clear()
        is_busy.set()
        try:
            audio = recorder.stop()
            duration = len(audio) / SAMPLE_RATE
            if duration < MIN_SECONDS:
                print("\r" + " " * 40 + "\r(занадто коротко, пропущено)")
                return

            print(" розпізнаю...", end="", flush=True)
            text = model.recognize(audio, sample_rate=SAMPLE_RATE).strip()
            print("\r" + " " * 40 + "\r" + (text or "(тиша)"))

            if text:
                if COPY_TO_CLIPBOARD:
                    copy_to_clipboard(text)
                if AUTO_TYPE:
                    keyboard.write(text)
        finally:
            is_busy.clear()

    keyboard.on_press_key(HOTKEY, on_press, suppress=False)
    keyboard.on_release_key(HOTKEY, on_release, suppress=False)

    print(f"\nГотово. Тримайте [{HOTKEY.upper()}] — говоріть, відпустіть — отримаєте текст.")
    print("Ctrl+C — вихід.\n")

    try:
        keyboard.wait()
    except KeyboardInterrupt:
        print("\nЗупинено.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())