"""
harness.py — запуск СПРАВЖНЬОГО DictationApp у тесті.

Справжнє: модель (ваш _load_asr_model), run_dictation_session, VAD, гейт голосу,
склейка, фільтр впевненості й УСІ константи UAFastConformer.py. Тут вони
лише ЧИТАЮТЬСЯ — жодних присвоєнь UAFastConformer.X = ... немає.

Підмінено тільки ввід/вивід (на інстансі, клас не змінюється):
  app.recorder     → ReplayRecorder: звук з .wav замість мікрофона
  app._wait_for_next_step → без реального очікування, крок = ваш STEP_INTERVAL_SEC
  app.overlay      → NullOverlay: без вікон
  app.insert_text_into_active_window → нічого не робить: текст не вставляється у ваші вікна
  app.show_final_text → обгортка: викликає ваш оригінал і запам'ятовує recognized

app.load() не викликається: він вантажить ще й модель перекладу (torch) і
відкриває мікрофон.
"""
from __future__ import annotations

import enum
import importlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import numpy as np

from App.CaptureDataset_ForTests.dataset_capture import CLIPS_DIR, read_wav

APP_MODULE = "UAFastConformer"
EVENT_NAMES = ("partial_text", "commit_fallback", "final_audio", "final_skipped", "final_done")

# Константи, які потрапляють у звіт і baseline. Якщо якоїсь у модулі вже
# немає (перейменували) — вона просто пропускається.
CONFIG_KEYS = (
    "ACTIVE_MODEL", "QUANTIZATION", "FORCE_CPU",
    "SAMPLE_RATE", "MAX_SECONDS", "MIN_SECONDS",
    "STEP_INTERVAL_SEC", "STEP_MIN_AUDIO_SEC",
    "MIN_AVG_LOGPROB", "MIN_AVG_LOGPROB_LONG", "CONF_SHORT_WORDS",
    "SILENCE_DBFS", "FRAME_SEC", "MIN_VOICED_FRAMES",
    "COMMIT_MIN_SEC", "COMMIT_PAUSE_SEC", "SENTENCE_PAUSE_SEC", "COMMIT_MAX_WINDOW_SEC",
    "FORCED_CUT_SEARCH_SEC", "FORCED_CUT_GUARD_SEC",
    "PAUSE_NOISE_PERCENTILE", "PAUSE_MARGIN_DB", "PAUSE_SPEECH_GAP_DB",
    "VOICE_GATE", "VOICE_GATE_DBFS", "VOICE_GATE_HOLD_SEC",
)

LENGTH_TAGS = ((5.0, "short"), (20.0, "medium"))   # понад 20 с — "long"


def load_app_module() -> ModuleType:
    """Імпорт UAFastConformer без вікон: Qt працює в режимі offscreen."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return importlib.import_module(APP_MODULE)


def config_snapshot(module: ModuleType) -> dict:
    snap = {}
    for key in CONFIG_KEYS:
        if hasattr(module, key):
            value = getattr(module, key)
            snap[key] = value.value if isinstance(value, enum.Enum) else value
    return snap


# --------------------------------------------------------------------------
# Датасет
# --------------------------------------------------------------------------

@dataclass
class Clip:
    id: str
    wav: Path
    reference: str
    tags: list[str]
    duration_sec: float

    @property
    def expect_empty(self) -> bool:
        return not self.reference.strip()


def length_tag(duration_sec: float) -> str:
    for limit, tag in LENGTH_TAGS:
        if duration_sec <= limit:
            return tag
    return "long"


def load_dataset(dataset_dir: Path) -> list[Clip]:
    """Розмічені кліпи: є і NNNN.wav, і NNNN.txt. Порожній .txt — негативний
    кліп. Тег довжини — автоматично; інші теги — з manifest.json (необов'язково)."""
    clips_dir = Path(dataset_dir) / CLIPS_DIR
    if not clips_dir.is_dir():
        return []
    manifest_path = Path(dataset_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig")) if manifest_path.exists() else {}

    clips = []
    for wav in sorted(clips_dir.glob("*.wav")):
        txt = wav.with_suffix(".txt")
        if not txt.exists():
            continue
        audio, rate = read_wav(wav)
        duration = len(audio) / rate
        extra = [t for t in manifest.get(wav.stem, []) if t]
        tags = [length_tag(duration)] + [t for t in extra if t not in ("short", "medium", "long")]
        clips.append(Clip(wav.stem, wav, txt.read_text(encoding="utf-8-sig").strip(), tags, duration))
    return clips


# --------------------------------------------------------------------------
# Заміни вводу/виводу
# --------------------------------------------------------------------------

class NullOverlay:
    """Оверлей без вікна: ті самі методи, нічого не показують."""

    def display(self, *_a, **_kw) -> None: ...
    def dismiss(self, *_a, **_kw) -> None: ...
    def copy_text(self, *_a, **_kw) -> None: ...
    def copy_text_and_wait(self, *_a, **_kw) -> bool: return True

    def __getattr__(self, _name):          # нові методи оверлея не ламають тест
        return lambda *_a, **_kw: None


class ReplayRecorder:
    """Звук з файлу замість мікрофона. snapshot() росте на step_samples
    за кожен advance(); stop() віддає весь файл. Обрізання до MAX_SECONDS —
    як у справжнього Recorder."""

    def __init__(self, audio: np.ndarray, step_samples: int, max_samples: int) -> None:
        self.audio = audio[:max_samples]
        self.step = max(1, step_samples)
        self.pos = 0

    def open(self) -> None: ...
    def close(self) -> None: ...

    def start(self) -> None:
        self.pos = 0

    def advance(self) -> bool:
        """Ще один крок звуку. True — файл закінчився (клавішу відпустили)."""
        self.pos = min(len(self.audio), self.pos + self.step)
        return self.pos >= len(self.audio)

    def snapshot(self) -> np.ndarray:
        return self.audio[:self.pos].copy()

    def stop(self) -> np.ndarray:
        return self.audio.copy()


@dataclass
class Probe:
    """Що побачив тест за один прогін run_dictation_session."""
    recognized: str | None = None          # None — show_final_text не викликали
    events: list[str] = field(default_factory=list)

    def reset(self) -> None:
        self.recognized = None
        self.events = []


@dataclass
class ClipRun:
    clip: Clip
    recognized: str                        # "" якщо тиша / занадто коротко
    events: list[str]
    too_short: bool                        # цикл вийшов до фіналу (< MIN_SECONDS)
    single_pass: str | None = None         # один asr_pass на всьому файлі


class TestApp:
    """Справжній DictationApp, підготовлений до прогону кліпів."""

    __test__ = False                       # pytest: це не клас з тестами

    def __init__(self) -> None:
        self.module = load_app_module()
        self.app = self.module.DictationApp()
        self.probe = Probe()
        self._prepare()

    def _prepare(self) -> None:
        app, probe = self.app, self.probe
        app.overlay = NullOverlay()
        app.model = app._load_asr_model()
        app.ts_model = app.model.with_timestamps()

        original_show_final_text = app.show_final_text

        def show_final_text(recognized: str, output: str) -> None:
            original_show_final_text(recognized, output)
            probe.recognized = recognized

        app.show_final_text = show_final_text
        app.insert_text_into_active_window = lambda _text: None

        for name in EVENT_NAMES:
            app.events.subscribe(name, lambda _n=name, **_: probe.events.append(_n))

    @property
    def sample_rate(self) -> int:
        return self.module.SAMPLE_RATE

    def config(self) -> dict:
        return config_snapshot(self.module)

    def _load_audio(self, clip: Clip) -> np.ndarray:
        audio, rate = read_wav(clip.wav)
        if rate != self.sample_rate:
            raise ValueError(f"{clip.wav.name}: частота {rate} Гц, а застосунок працює з {self.sample_rate} Гц")
        return audio

    def run_clip(self, clip: Clip) -> ClipRun:
        """Один прогін вашого run_dictation_session() на кліпі."""
        m, app = self.module, self.app
        audio = self._load_audio(clip)
        recorder = ReplayRecorder(audio, int(m.STEP_INTERVAL_SEC * m.SAMPLE_RATE),
                                  int(m.MAX_SECONDS * m.SAMPLE_RATE))

        def wait_step(_step_started_at: float) -> bool:
            finished = recorder.advance()
            if finished:
                app.is_recording.clear()   # як у on_release()
            return finished

        app.recorder = recorder
        app._wait_for_next_step = wait_step
        self.probe.reset()
        recorder.start()
        app.is_recording.set()
        app.run_dictation_session()

        too_short = "final_audio" not in self.probe.events
        return ClipRun(clip, self.probe.recognized or "", list(self.probe.events), too_short)

    def single_pass(self, clip: Clip) -> str:
        """Один прохід asr_pass на всьому файлі — як фінал без розрізів
        (з вашим гейтом голосу, якщо VOICE_GATE)."""
        m = self.module
        audio = self._load_audio(clip)[: int(m.MAX_SECONDS * m.SAMPLE_RATE)]
        if len(audio) < m.MIN_SECONDS * m.SAMPLE_RATE:
            return ""
        if m.VOICE_GATE:
            audio, _thr, _muted = m.apply_voice_gate(audio)
        return self.app.asr_pass(audio, "фінал", len(audio) / m.SAMPLE_RATE)
