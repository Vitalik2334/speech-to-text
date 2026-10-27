"""
dataset_capture.py — збір тестових кліпів для інтеграційних тестів
розпізнавання. Під'єднується ЗЗОВНІ, як profiling.py: DictationApp про нього
нічого не знає.

Що робить attach_capture(app):
  * підписується на подію app.events "final_audio" і запам'ятовує копію
    raw_audio — сирий звук до гейту голосу (саме те, що дав мікрофон);
  * обгортає app.show_final_text на ІНСТАНСІ (клас не змінюється): викликає
    оригінал без змін, а потім бере recognized (текст ДО перекладу);
  * зберігає пару файлів у <VTT_DATASET>/clips/:
        NNNN.wav   — 16 кГц, моно, PCM16
        NNNN.json  — службове: що розпізнав застосунок, модель, налаштування

Еталон NNNN.txt створюєте ви вручну — див. README.md у цій теці.

Порядок у main():
    app = DictationApp()
    attach_capture(app)       # до load() і run()
    app.load()
    app.run()

Натискання коротші за MIN_SECONDS не зберігаються: run_dictation_session виходить
раніше, ніж випускає final_audio.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import sys
import threading
import wave
from pathlib import Path

import numpy as np

ENV_VAR = "VTT_DATASET"
CLIPS_DIR = "clips"
# Єдине місце, де визначено папку датасету за замовчуванням: <корінь проєкту>/Tests
# (файл лежить у App/CaptureDataset_ForTests/, тому parents[2] — це корінь VoiceToText).
# І застосунок (--capture), і інтеграційні тести беруть шлях через dataset_dir_from_env().
DEFAULT_DATASET_DIR = Path(__file__).resolve().parents[2] / "Tests"   # .../VoiceToText/Tests
_CLIP_NAME = re.compile(r"^(\d+)\.wav$", re.IGNORECASE)


# --------------------------------------------------------------------------
# WAV: запис і читання (читання використовують тести)
# --------------------------------------------------------------------------

def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """float32 [-1, 1] → моно PCM16."""
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Моно PCM16 → (float32 [-1, 1], частота). Інші формати — зрозуміла помилка."""
    with wave.open(str(path), "rb") as wav:
        channels, width, rate = wav.getnchannels(), wav.getsampwidth(), wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    if channels != 1 or width != 2:
        raise ValueError(f"{path.name}: потрібен моно PCM16, а тут каналів={channels}, байт на семпл={width}")
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0, rate


def dataset_dir_from_env() -> Path:
    """Папка датасету: змінна VTT_DATASET, якщо задана, інакше DEFAULT_DATASET_DIR."""
    value = os.environ.get(ENV_VAR, "").strip()
    return Path(value) if value else DEFAULT_DATASET_DIR


def next_clip_number(clips_dir: Path) -> int:
    numbers = [int(m.group(1)) for p in clips_dir.iterdir() if (m := _CLIP_NAME.match(p.name))]
    return max(numbers, default=0) + 1


# --------------------------------------------------------------------------
# DatasetCapture
# --------------------------------------------------------------------------

class DatasetCapture:
    def __init__(self, app, dataset_dir: Path) -> None:
        self.app = app
        self.clips_dir = Path(dataset_dir) / CLIPS_DIR
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        # Модуль, де визначено DictationApp (__main__ при запуску скриптом).
        self.cfg = sys.modules[type(app).__module__]
        self._pending_audio: np.ndarray | None = None
        self._number_lock = threading.Lock()
        self._next_number: int | None = None

    def attach(self) -> "DatasetCapture":
        self.app.events.subscribe("final_audio", self._on_final_audio)
        original_show_final_text = self.app.show_final_text

        def show_final_text(recognized: str, output: str) -> None:
            original_show_final_text(recognized, output)
            self._on_result(recognized)

        self.app.show_final_text = show_final_text
        print(f"[capture] Збір кліпів УВІМКНЕНО → {self.clips_dir}", flush=True)
        return self

    # ---- події ---------------------------------------------------------------

    def _on_final_audio(self, raw_audio, **_) -> None:
        self._pending_audio = np.array(raw_audio, dtype=np.float32, copy=True)

    def _on_result(self, recognized: str) -> None:
        audio, self._pending_audio = self._pending_audio, None
        if audio is None or len(audio) == 0:
            return
        number = self._reserve_number()
        meta = self._metadata(recognized, audio)
        # Запис на диск — у фоні, щоб не затримувати вивід тексту.
        threading.Thread(target=self._save, args=(number, audio, meta), name="dataset-capture").start()

    # ---- збереження ----------------------------------------------------------

    def _reserve_number(self) -> int:
        with self._number_lock:
            if self._next_number is None:
                self._next_number = next_clip_number(self.clips_dir)
            number = self._next_number
            self._next_number += 1
            return number

    def _metadata(self, recognized: str, audio: np.ndarray) -> dict:
        cfg = self.cfg
        sample_rate = getattr(cfg, "SAMPLE_RATE", 16_000)
        model = getattr(cfg, "ACTIVE_MODEL", None)
        return {
            "recognized": recognized,
            "model": getattr(model, "value", model),
            "quantization": getattr(cfg, "QUANTIZATION", None),
            "voice_gate": getattr(cfg, "VOICE_GATE", None),
            "duration_sec": round(len(audio) / sample_rate, 3),
            "sample_rate": sample_rate,
            "captured_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        }

    def _save(self, number: int, audio: np.ndarray, meta: dict) -> None:
        stem = f"{number:04d}"
        try:
            write_wav(self.clips_dir / f"{stem}.wav", audio, meta["sample_rate"])
            (self.clips_dir / f"{stem}.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\n[capture] збережено {stem}.wav ({meta['duration_sec']:.1f} с)", flush=True)
        except OSError as exc:
            print(f"\n[capture] не вдалося зберегти {stem}: {exc}", file=sys.stderr)


def attach_capture(app) -> DatasetCapture | None:
    """Під'єднує збір кліпів. Без dataset_dir шлях — dataset_dir_from_env()
    (VTT_DATASET або DEFAULT_DATASET_DIR), той самий, що читають тести."""
    path = dataset_dir_from_env()
    return DatasetCapture(app, path).attach()
