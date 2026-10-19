"""
UAFastConformer.py — push-to-talk диктовка (запис, розпізнавання, вивід).

Профілювання сюди НЕ вшите. Воно під'єднується ззовні в main() через
profiling.attach_profiler(app), який обгортає методи DictationApp і
слухає app.events. Вимкнути: PROFILE = False або аргумент --no-profile.

Компіляція:
   uv run python -m PyInstaller --noconfirm --clean --onedir --windowed --name UAFastConformerNew1 --contents-directory libs --collect-all onnx_asr --collect-all onnxruntime --copy-metadata onnx-asr --hidden-import pynput.keyboard._win32 --hidden-import pynput.mouse._win32 --hidden-import profiling App/UAFastConformer.py

"""
import collections
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from PySide6.QtCore import QObject, Signal, Qt, QTimer
from PySide6.QtWidgets import QApplication
from hotkeys_manager import HotkeyManager
from pynput.keyboard import Controller as KbController
from models_config import MODEL_TYPES, RECOMMENDED_CPU_QUANTIZATION, WHISPER_LANGUAGE, WHISPER_MODELS, ModelName
from overlay import Overlay

#TODO Після відправки цілого повідомлення. І ще раз пропускати все через модель.
ACTIVE_MODEL = ModelName.NEMO_PARAKEET
# --- Апаратний профіль ---------------------------------------------------
FORCE_CPU = True
print(f"{FORCE_CPU=}")
PROVIDERS = ["CPUExecutionProvider"] if FORCE_CPU else ["CUDAExecutionProvider", "CPUExecutionProvider"]

QUANTIZATION = RECOMMENDED_CPU_QUANTIZATION[ACTIVE_MODEL] if FORCE_CPU else None

print(f"{QUANTIZATION=},{RECOMMENDED_CPU_QUANTIZATION[ACTIVE_MODEL]=}")
# --------------------------------------------------------------------------
MODEL_TYPE = MODEL_TYPES[ACTIVE_MODEL]
EXPECTED_FILES = ("encoder-model.onnx", "decoder_joint-model.onnx", "vocab.txt", "config.json")

HOTKEY = "shift+1"
TRANSLATE_TOGGLE_HOTKEY = "shift+2"
TRANSLATE_TARGET_LANG = "en"

TRANSLATION_MODELS: dict[str, str] = {
    "en": "Helsinki-NLP/opus-mt-uk-en",
    "pl": "Helsinki-NLP/opus-mt-uk-pl",
    "de": "Helsinki-NLP/opus-mt-uk-de",
    "fr": "Helsinki-NLP/opus-mt-uk-fr",
    "es": "Helsinki-NLP/opus-mt-uk-es",
}

SAMPLE_RATE = 16_000
MAX_SECONDS = 120.0         # запобіжник на весь запис; модель бачить лише вікно ≤ COMMIT_MAX_WINDOW_SEC
MIN_SECONDS = 0.2

AUTO_TYPE = True
COPY_TO_CLIPBOARD = True

# Як вводити готовий текст у активне вікно:
#   "paste" — буфер обміну + Ctrl+V: миттєво незалежно від довжини
#             (замінює вміст буфера — як і COPY_TO_CLIPBOARD).
#   "type"  — посимвольно через pynput: ~1 мс/символ (159 симв. = 144 мс),
#             зате не чіпає буфер і працює там, де Ctrl+V не вставляє.
OUTPUT_MODE = "paste"

# Мікрофонний стрім відкривається один раз при старті й лишається відкритим:
# відкриття пристрою займало ~210 мс на КОЖНЕ натискання (і блокувало GUI),
# закриття — ще ~75 мс після відпускання. Мінус: Windows весь час показує
# індикатор "мікрофон використовується". False — старая поведінка.
KEEP_MIC_OPEN = True
# Скільки аудіо ДО натискання додавати на початок запису (тільки при
# KEEP_MIC_OPEN) — не обрізається перший склад, якщо почали говорити
# одночасно з натисканням.
PREROLL_SEC = 0.3

OVERLAY_AUTOHIDE_MS = 4000
OVERLAY_TOO_SHORT_MS = 1200
STEP_INTERVAL_SEC = 0.3        # період між ПОЧАТКАМИ проміжних кроків (а не пауза після кроку)
STEP_MIN_GAP_SEC = 0.03        # мінімальна пауза між кроками — щоб GUI/хук встигали дихати
STEP_MIN_AUDIO_SEC = 0.3

# Фільтр впевненості відсікає "галюцинації" моделі на шумі (типово 1-2
# випадкових слова). Раніше один поріг -0.40 застосовувався до БУДЬ-якого
# тексту — і при фоновому шумі викидав справжнє мовлення цілком (модель
# менш упевнена → середній logprob падає нижче порогу). Тепер:
#   * коротка фраза (≤ CONF_SHORT_WORDS слів) — суворий поріг MIN_AVG_LOGPROB;
#   * довша — м'який MIN_AVG_LOGPROB_LONG (відкидаємо лише явне сміття).
# Знайдіть свої значення: DEBUG_CONFIDENCE=True друкує avg_logprob кожного
# результату — подивіться, які цифри дає мовлення в шумі, а які чистий шум.
MIN_AVG_LOGPROB = -0.40
MIN_AVG_LOGPROB_LONG = -1.00
CONF_SHORT_WORDS = 2
DEBUG_CONFIDENCE = False

# Профілювання (profiling.py) за замовчуванням; перекривається аргументами
# командного рядка --profile / --no-profile.

PROFILE = False
CAPTURE = False

@dataclass
class StreamState:
    committed: list[str] = field(default_factory=list)  # зафіксовані сегменти тексту
    committed_len: int = 0      # скільки семплів уже зафіксовано
    tail_text: str = ""         # текст останнього кроку по незафіксованому хвосту
    tail_valid: bool = True     # чи хвіст без мовлення (текст кроку не потрібен)
    last_audio_len: int = 0     # довжина audio на момент останнього кроку
    # soft[i] — тип межі ПІСЛЯ committed[i]: True = середина речення
    # (коротка пауза / примусовий розріз), False = кінець речення.
    soft: list[bool] = field(default_factory=list)

    def step_covered(self) -> int:
        """Скільки семплів після committed_len бачив останній крок."""
        return self.last_audio_len - self.committed_len

    def joined(self, tail: str = "") -> str:
        """Увесь текст: зафіксовані шматки + хвіст, склеєні з урахуванням меж."""
        return join_segments(self.committed + [tail], self.soft)

    def commit(self, segment: str, cut: int, audio, soft_cut: bool = False) -> None:
        """Фіксує шматок довжиною cut і скидає стан хвоста.
        soft_cut — межа в кінці шматка м'яка (див. find_commit_point)."""
        if segment:
            self.committed.append(segment)
            self.soft.append(soft_cut)
        elif self.soft:
            # Шматок без тексту = тиша між фразами: це кінець речення.
            self.soft[-1] = False
        self.committed_len += cut
        self.tail_text = ""
        self.tail_valid = not has_speech(audio[self.committed_len:])
        self.last_audio_len = len(audio)


class Events:
    """Мінімальна шина подій: робочий код повідомляє, ЩО сталося, а
    спостерігачі (профайлер, логер, тести) самі вирішують, що з цим робити.
    Без підписників emit() — лише пошук у словнику."""

    def __init__(self) -> None:
        self._subs: dict[str, list[Callable[..., None]]] = defaultdict(list)

    def subscribe(self, name: str, handler: Callable[..., None]) -> None:
        self._subs[name].append(handler)

    def emit(self, name: str, **payload) -> None:
        for handler in self._subs.get(name, ()):
            handler(**payload)


class HotkeyBridge(QObject):
    dictate_pressed = Signal()
    dictate_released = Signal()
    translate_toggled = Signal()


class Recorder:
    """Запис з мікрофона.

    KEEP_MIC_OPEN=True: стрім відкривається один раз (open()) і працює
    постійно; start()/stop() лише вмикають/вимикають накопичення кадрів —
    це миттєво. Поки запис вимкнено, останні PREROLL_SEC тримаються в
    кільцевому буфері й додаються на початок наступного запису.

    KEEP_MIC_OPEN=False: стара поведінка — стрім відкривається в start()
    і закривається в stop()."""

    def __init__(self, samplerate: int):
        self.samplerate = samplerate
        self._frames: list[np.ndarray] = []
        self._stream = None
        self._recording = False
        self._lock = threading.Lock()
        self._preroll: collections.deque[np.ndarray] = collections.deque()
        self._preroll_len = 0
        self._preroll_max = int(PREROLL_SEC * samplerate) if KEEP_MIC_OPEN else 0

    def open(self) -> None:
        import sounddevice as sd  # noqa: PLC0415

        with self._lock:
            if self._stream is not None:
                return
            self._stream = sd.InputStream(
                samplerate=self.samplerate,
                channels=1,
                dtype="float32",
                callback=self._callback,
            )
            self._stream.start()

    def close(self) -> None:
        with self._lock:
            stream, self._stream = self._stream, None
        if stream is not None:
            stream.stop()
            stream.close()

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            print(status, file=sys.stderr)
        chunk = indata[:, 0].copy()
        with self._lock:
            if self._recording:
                self._frames.append(chunk)
            elif self._preroll_max > 0:
                self._preroll.append(chunk)
                self._preroll_len += len(chunk)
                while self._preroll and self._preroll_len - len(self._preroll[0]) >= self._preroll_max:
                    self._preroll_len -= len(self._preroll.popleft())

    def start(self) -> None:
        if self._stream is None:
            self.open()
        with self._lock:
            self._frames = list(self._preroll)
            self._preroll.clear()
            self._preroll_len = 0
            self._recording = True

    def snapshot(self) -> np.ndarray:
        with self._lock:
            if not self._frames:
                return np.zeros(0, dtype=np.float32)
            frames = list(self._frames)
        audio = np.concatenate(frames)
        max_len = int(MAX_SECONDS * self.samplerate)
        return audio[:max_len]

    def stop(self) -> np.ndarray:
        with self._lock:
            was_recording = self._recording
            self._recording = False
            frames, self._frames = self._frames, []
        if not KEEP_MIC_OPEN:
            self.close()

        if not was_recording or not frames:
            return np.zeros(0, dtype=np.float32)

        audio = np.concatenate(frames)
        max_len = int(MAX_SECONDS * self.samplerate)
        return audio[:max_len]


SILENCE_DBFS = -45.0
FRAME_SEC = 0.05
MIN_VOICED_FRAMES = 3


FRAME_LEN = int(FRAME_SEC * SAMPLE_RATE)


def frame_dbfs(audio: np.ndarray) -> np.ndarray:
    """Рівень (dBFS) кожного кадру FRAME_SEC. Неповний хвіст відкидається."""
    n = len(audio) // FRAME_LEN
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    frames = audio[: n * FRAME_LEN].reshape(n, FRAME_LEN)
    #Для кожного рядка фрейму рахуємо децибели
    return 20.0 * np.log10(np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-12)


def has_speech(audio: np.ndarray) -> bool:
    """Чи є в шматку хоча б MIN_VOICED_FRAMES гучних кадрів."""
    return int(np.sum(frame_dbfs(audio) >= SILENCE_DBFS)) >= MIN_VOICED_FRAMES


def should_skip_recognition(audio: np.ndarray, last_audio_len: int) -> bool:
    """True, якщо в аудіо, що з'явилося після останнього розпізнавання,
    немає мовлення - тоді перерозпізнавати немає сенсу."""
    return not has_speech(audio[last_audio_len:])


# --- Ковзне вікно ---------------------------------------------------------
# Сказане до паузи "фіксується" (розпізнається один раз і більше не
# перераховується), далі модель бачить лише хвіст після паузи. Тому час
# кроку не росте з довжиною фрази (раніше +~50 мс на кожну секунду).
COMMIT_MIN_SEC = 1.0          # вікно коротше за це — не фіксуємо (мало контексту)
COMMIT_PAUSE_SEC = 0.5       # мінімальна пауза, по якій ріжемо (0.35 різало на звичайних вдихах)
# Пауза, довша за це, вважається кінцем речення ("тверда" межа): крапку й
# велику літеру моделі лишаємо. Коротша пауза або примусовий розріз —
# "м'яка" межа: це середина речення, і join_segments склеює шматки в одне
# речення (прибирає крапку, першу літеру наступного шматка робить малою).
SENTENCE_PAUSE_SEC = 0.5
COMMIT_MAX_WINDOW_SEC = 10   # без пауз довше — ріжемо примусово по найтихішому місцю
FORCED_CUT_SEARCH_SEC = 3.0   # де шукати найтихіше місце: останні N с вікна...
FORCED_CUT_GUARD_SEC = 0.5    # ...але не ближче за це до кінця (там іде мовлення)

# Адаптивний поріг паузи: фіксований SILENCE_DBFS (-45) не бачить пауз,
# якщо в мікрофона/кімнати фоновий шум гучніший. Тому паузою вважається
# кадр, тихіший за (рівень шуму + PAUSE_MARGIN_DB), де рівень шуму —
# PAUSE_NOISE_PERCENTILE-й перцентиль гучності вікна. Поріг обмежено
# зверху: між шумом і мовленням має лишатися хоча б PAUSE_SPEECH_GAP_DB.
PAUSE_NOISE_PERCENTILE = 10
PAUSE_MARGIN_DB = 8.0
PAUSE_SPEECH_GAP_DB = 10.0

PAUSE_SAMPLES = int(COMMIT_PAUSE_SEC * SAMPLE_RATE)


def _step_covers_cut(step_covered: int, cut: int) -> bool:
    """Кінець останнього кроку лежить у межах паузи після cut,
    тобто крок бачив приблизно той самий шматок, що й фіксація."""
    return cut <= step_covered <= cut + PAUSE_SAMPLES


def pause_threshold(dbfs: np.ndarray) -> float:
    noise = float(np.percentile(dbfs, PAUSE_NOISE_PERCENTILE))
    speech = float(np.percentile(dbfs, 90))
    thr = noise + PAUSE_MARGIN_DB
    thr = min(thr, speech - PAUSE_SPEECH_GAP_DB)
    return max(thr, SILENCE_DBFS)


def find_last_pause(window: np.ndarray, min_pause: int, min_end: int
                    ) -> tuple[np.ndarray, tuple[int, int, bool] | None]:
    """Аналіз гучності: рівні кадрів і остання достатньо довга пауза."""
    levels = frame_dbfs(window)
    n_frames = len(levels)
    is_quiet = levels < pause_threshold(levels)

    last = None          # (кадр розрізу, довжина паузи, пауза до кінця вікна)
    pause_len = 0
    for f in range(n_frames + 1):
        if f < n_frames and is_quiet[f]:
            pause_len += 1
            continue
        # f — перший гучний кадр після паузи (або кінець вікна)
        if pause_len >= min_pause and f >= min_end:
            mid = f - pause_len // 2
            last = (mid, pause_len, f == n_frames)
        pause_len = 0
    return levels, last


def find_commit_point(window: np.ndarray) -> tuple[int, bool] | None:
    """Рішення: різати по паузі, чекати чи різати силою."""
    if len(window) < COMMIT_MIN_SEC * SAMPLE_RATE:
        return None
    min_pause = max(1, int(round(COMMIT_PAUSE_SEC / FRAME_SEC)))
    sent_pause = max(min_pause, int(round(SENTENCE_PAUSE_SEC / FRAME_SEC)))
    min_end = int(COMMIT_MIN_SEC / FRAME_SEC)

    levels, last = find_last_pause(window, min_pause, min_end)

    if last is not None:
        cut, length, at_end = last
        short = length < sent_pause
        if not (at_end and short):   # коротка пауза в кінці ще може вирости
            return cut * FRAME_LEN, short

    if len(window) >= COMMIT_MAX_WINDOW_SEC * SAMPLE_RATE:   # вимушений розріз
        end = max(1, len(levels) - int(FORCED_CUT_GUARD_SEC / FRAME_SEC))
        start = max(0, end - int(FORCED_CUT_SEARCH_SEC / FRAME_SEC))
        return (start + int(np.argmin(levels[start:end]))) * FRAME_LEN, True
    return None


# --- Гейт "тільки мій голос" ----------------------------------------------
# Фонове мовлення (відео, телевізор, колеги) модель теж розпізнає — для неї
# це звичайна мова. Але ваш голос біля мікрофона значно гучніший. Гейт
# глушить (обнуляє) усе, що тихіше за VOICE_GATE_DBFS, тож модель отримує
# лише ваш голос, а фон для неї — тиша.
#   VOICE_GATE_HOLD_SEC — скільки тримати гейт відкритим до/після гучного
#   кадру, щоб не обрізати тихі приголосні на краях слів.
# Підбір: якщо фон досі проходить — збільште VOICE_GATE_DBFS (напр. -30).
# Якщо обрізаються ваші слова — навпаки.
VOICE_GATE = True
VOICE_GATE_DBFS = -40.0
VOICE_GATE_HOLD_SEC = 0.35


def voice_gate_threshold(dbfs: np.ndarray) -> float:  # noqa: ARG001
    return VOICE_GATE_DBFS


def apply_voice_gate(audio: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Повертає (аудіо з обнуленим фоном, поріг dBFS, частка заглушеного)."""
    dbfs = frame_dbfs(audio)
    if len(dbfs) == 0:
        return audio, 0.0, 0.0
    thr = voice_gate_threshold(dbfs)
    loud = (dbfs >= thr).astype(np.float32)
    n = len(loud)
    hold = int(round(VOICE_GATE_HOLD_SEC / FRAME_SEC))
    if hold > 0:
        # mode="full" + зріз, а не mode="same": коли вікно (2*hold+1 кадрів)
        # довше за запис, "same" повертає масив довжини ВІКНА, і маска
        # виходила довшою за аудіо (ValueError: could not be broadcast).
        full = np.convolve(loud, np.ones(2 * hold + 1, dtype=np.float32), mode="full")
        loud = (full[hold:hold + n] > 0).astype(np.float32)
    mask = np.repeat(loud, FRAME_LEN)
    if len(mask) < len(audio):  # неповний останній кадр — як попередній
        mask = np.concatenate([mask, np.full(len(audio) - len(mask), loud[-1], dtype=np.float32)])
    return audio * mask[: len(audio)], thr, float(1.0 - loud.mean())


def _decapitalize(text: str) -> str:
    """Перша літера → мала, якщо слово не абревіатура ("США", "ЗСУ", "IT")."""
    word = text.split(maxsplit=1)[0] if text else ""
    if not word or not word[0].isupper():
        return text
    letters = [c for c in word if c.isalpha()]
    if len(letters) > 1 and all(c.isupper() for c in letters):
        return text
    return text[0].lower() + text[1:]


def join_segments(parts: list[str], soft: list[bool] | None = None) -> str:
    """Склеює шматки тексту, розпізнані окремо. Модель вважає КОЖЕН шматок
    окремим записом: ставить крапку в кінці й велику літеру на початку,
    навіть якщо розріз був посеред речення ("рішення. Що прийняти").

    soft[k] — тип межі МІЖ parts[k] і parts[k+1]:
      True  (м'яка: коротка пауза / примусовий розріз) — одне речення:
            крапку з кінця прибираємо, першу літеру наступного робимо малою;
      False (тверда: довга пауза) — лишаємо як дала модель.
    Якщо soft не передано або межа невідома — стара поведінка: крапку
    прибираємо лише коли наступний шматок і так починається з малої."""
    out = ""
    prev_soft = False
    for k, p in enumerate(parts):
        p = p.strip()
        if p:
            if out:
                single_dot = out.endswith(".") and not out.endswith("..")
                if prev_soft and not out.endswith(("?", "!")):
                    # Середина речення: "?"/"!" — свідомий кінець, їх не чіпаємо.
                    p = _decapitalize(p)
                    if single_dot:
                        out = out[:-1]
                elif p[0].islower() and single_dot:
                    out = out[:-1]
            out = f"{out} {p}" if out else p
        # межа після цього шматка; порожній шматок межу не перериває
        if soft is not None and k < len(soft):
            prev_soft = soft[k] if p else (prev_soft and soft[k])
    return out


def average_logprob(logprobs) -> float | None:
    if logprobs is None or len(logprobs) == 0:
        return None
    return float(sum(logprobs)) / len(logprobs)


def is_low_confidence(logprobs, min_avg_logprob: float) -> bool:
    avg = average_logprob(logprobs)
    return avg is not None and avg < min_avg_logprob


# ==========================================================================
# DictationApp
# ==========================================================================

class DictationApp:
    """Уся логіка диктовки. Кожен етап — окремий метод, щоб зовнішні
    спостерігачі (profiling.attach_profiler) могли його обгорнути, а там,
    де обгорнути нічого не вийде, випускаються події в self.events:

        partial_text(text)               — показано проміжний текст
        commit_fallback()                — фіксація взяла текст кроку
        final_audio(raw_audio, muted_share) — почався фінальний крок
        final_skipped()                  — фінал не розпізнавали (нового мовлення немає)
        final_done()                     — фінальний текст показано / надруковано

    Порядок: DictationApp() → [attach_profiler(app)] → load() → run()."""

    def __init__(self) -> None:
        self.qt_app = QApplication.instance() or QApplication(sys.argv)
        self.overlay = Overlay()
        self.events = Events()
        self.recorder = Recorder(SAMPLE_RATE)

        self.is_recording = threading.Event()
        self.is_busy = threading.Event()
        self.wake_event = threading.Event()
        self.translate_enabled = threading.Event()

        self.model = None
        self.ts_model = None
        self._translator = None          # (tokenizer, model)
        self._typer = KbController()

        self.bridge: HotkeyBridge | None = None
        self.hotkeys: HotkeyManager | None = None
        self._timers: list[QTimer] = []

    # ---- завантаження ------------------------------------------------------

    def load(self) -> None:
        provider_label = "CPU" if FORCE_CPU else "GPU (з fallback на CPU)"
        quant_label = QUANTIZATION or "fp32"
        print(f"Модель: {ACTIVE_MODEL.value} | провайдер: {provider_label} | квантизація: {quant_label}")
        print(f"Завантаження моделі {MODEL_TYPE}")
        self.overlay.display("Завантаження моделі...", state="info")

        self.model = self._load_asr_model()
        self.ts_model = self.model.with_timestamps()
        self.get_translator()
        self.overlay.dismiss()

        if KEEP_MIC_OPEN:
            try:
                self.recorder.open()
            except Exception as exc:  # noqa: BLE001
                print(f"(не вдалося відкрити мікрофон заздалегідь: {exc} — відкриється при натисканні)",
                      file=sys.stderr)

    def _load_asr_model(self):
        import onnx_asr  # noqa: PLC0415

        return onnx_asr.load_model(MODEL_TYPE, quantization=QUANTIZATION, providers=PROVIDERS)

    def _load_translator(self):
        from transformers import MarianMTModel, MarianTokenizer  # noqa: PLC0415

        model_name = TRANSLATION_MODELS[TRANSLATE_TARGET_LANG]
        print(f"Завантаження моделі перекладу {model_name} ...")
        self.overlay.display("Завантаження моделі перекладу...", state="info")
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        translator_model = MarianMTModel.from_pretrained(model_name)
        translator_model.eval()
        return tokenizer, translator_model

    def get_translator(self):
        if self._translator is None:
            self._translator = self._load_translator()
        return self._translator

    # ---- переклад ----------------------------------------------------------

    def translate_text(self, text: str) -> str:
        import torch  # noqa: PLC0415

        tokenizer, tr_model = self.get_translator()
        batch = tokenizer([text], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            generated = tr_model.generate(**batch, max_new_tokens=256)
        return tokenizer.decode(generated[0], skip_special_tokens=True).strip()

    def get_translated_text(self, text: str) -> str:
        if not text or not self.translate_enabled.is_set():
            return text

        print(" перекладаю...", end="", flush=True)
        self.overlay.display("Перекладаю..." + self.mode_suffix(), state="processing")
        try:
            return self.translate_text(text)
        except Exception as exc:  # noqa: BLE001
            print(f"\n(помилка перекладу: {exc})", file=sys.stderr)
            return text

    def mode_suffix(self) -> str:
        if self.translate_enabled.is_set():
            return f" [переклад → {TRANSLATE_TARGET_LANG.upper()}]"
        return ""

    # ---- розпізнавання -----------------------------------------------------

    def _run_recognize(self, audio: np.ndarray):
        if ACTIVE_MODEL in WHISPER_MODELS:
            return self.ts_model.recognize(audio, sample_rate=SAMPLE_RATE, language=WHISPER_LANGUAGE)
        return self.ts_model.recognize(audio, sample_rate=SAMPLE_RATE)

    def _confidence_check(self, text: str, logprobs) -> tuple[bool, float]:
        """(це шум?, застосований поріг)."""
        words = len(text.split())
        threshold = MIN_AVG_LOGPROB if words <= CONF_SHORT_WORDS else MIN_AVG_LOGPROB_LONG
        return is_low_confidence(logprobs, threshold), threshold

    def recognize_text(self, audio) -> str:
        res = self._run_recognize(audio)
        text = res.text.strip()
        if not text:
            return ""

        noise, threshold = self._confidence_check(text, res.logprobs)
        if DEBUG_CONFIDENCE or noise:
            # Відкинуте друкуємо ЗАВЖДИ — щоб було видно, що текст зник саме тут.
            avg = average_logprob(res.logprobs)
            avg_str = "n/a" if avg is None else f"{avg:.2f}"
            verdict = f"ВІДКИНУТО (поріг {threshold:.2f}) " if noise else ""
            print(f"\n[conf] avg_logprob={avg_str} {verdict}{text!r}", flush=True)
        return "" if noise else text

    def asr_pass(self, segment: np.ndarray, kind: str = "крок", total_sec: float = 0.0) -> str:
        """Один прохід ASR. kind ("крок" / "фіксація" / "фінал") і total_sec
        (довжина всього запису) самому розпізнаванню не потрібні — це
        контекст для спостерігачів, що обгортають метод."""
        return self.recognize_text(segment)

    # ---- вивід -------------------------------------------------------------

    def show_result(self, recognized: str, output: str) -> None:
        clear_line = "\r" + " " * 40 + "\r"

        if not output:
            print(clear_line + "(тиша)")
            self.overlay.display("(тиша)", state="warning", autohide_ms=OVERLAY_TOO_SHORT_MS)
            return

        print(clear_line + output)
        if output != recognized:
            print(f"    оригінал (uk): {recognized}")
        self.overlay.display(output, state="result", autohide_ms=OVERLAY_AUTOHIDE_MS)

    def deliver_text(self, text: str) -> None:
        """Передає текст у систему відповідно до налаштувань виводу."""
        if AUTO_TYPE and OUTPUT_MODE == "paste":
            self.paste_text(text)
            return
        if COPY_TO_CLIPBOARD:
            self.overlay.copy_text(text)
        if AUTO_TYPE:
            self.type_text(text)

    def type_text(self, text: str) -> None:
        self._typer.type(text)

    def paste_text(self, text: str) -> None:
        """Кладе текст у буфер (через GUI-потік, з очікуванням) і тисне Ctrl+V.
        Якщо буфер не встиг оновитися — падаємо назад на посимвольний друк,
        щоб не вставити старий вміст буфера."""
        import ctypes  # noqa: PLC0415
        from pynput.keyboard import Key, KeyCode  # noqa: PLC0415

        if not self.overlay.copy_text_and_wait(text, timeout=0.5):
            print("\n(буфер обміну не відповів — друкую посимвольно)", file=sys.stderr)
            self._typer.type(text)
            return

        # Чекаємо, поки користувач відпустить Shift/Ctrl/Alt/Win від хоткея,
        # інакше замість Ctrl+V піде Ctrl+Shift+V (до 1 с, далі вставляємо як є).
        user32 = ctypes.windll.user32
        deadline = time.perf_counter() + 1.0
        while time.perf_counter() < deadline and any(
                user32.GetAsyncKeyState(vk) & 0x8000 for vk in (0x10, 0x11, 0x12, 0x5B, 0x5C)
        ):
            time.sleep(0.01)

        # Клавіша V за кодом (VK_V = 0x56), а не символом "v": символ при
        # українській розкладці pynput надсилає як Unicode-подію, і Ctrl+V
        # програми не розпізнають. Код клавіші не залежить від розкладки.
        key_v = KeyCode.from_vk(0x56)
        with self._typer.pressed(Key.ctrl):
            self._typer.press(key_v)
            self._typer.release(key_v)
        print(f"\n[вивід] Ctrl+V надіслано ({len(text)} симв.)", flush=True)

    # ---- цикл розпізнавання ------------------------------------------------

    def _wait_step(self, step_started_at: float) -> bool:
        """Чекає до наступного кроку (або до відпускання клавіші).
        Повертає True, якщо клавішу вже відпустили.

        Чекаємо лише ЗАЛИШОК інтервалу: раніше після кроку в 240 мс
        цикл ще спав повні 300 мс, і текст оновлювався раз на ~550 мс."""
        remaining = STEP_INTERVAL_SEC - (time.perf_counter() - step_started_at)
        self.wake_event.wait(timeout=max(STEP_MIN_GAP_SEC, remaining))
        self.wake_event.clear()
        return not self.is_recording.is_set()

    def _pick_commit_text(self, head: np.ndarray, cut: int, state: StreamState, duration: float) -> str:
        """Розпізнає шматок до паузи. Якщо фільтр відкинув результат,
        але останній крок покривав той самий шматок, бере текст кроку,
        щоб сказане не зникло. Винятки ASR пробрасує нагору."""
        segment = self.asr_pass(head, "фіксація", duration) if has_speech(head) else ""
        if not segment and state.tail_text and _step_covers_cut(state.step_covered(), cut):
            self.events.emit("commit_fallback")
            return state.tail_text
        return segment

    def transcribe_loop(self) -> None:
        """Одне натискання = один цикл. Поки клавіша затиснута, раз на
        STEP_INTERVAL_SEC:
          * якщо у вікні (аудіо після зафіксованої частини) знайшлася пауза —
            шматок до паузи розпізнається ОДИН раз і фіксується в committed;
          * інакше розпізнається лише хвіст після зафіксованої частини.
        Після відпускання розпізнається тільки хвіст (або береться текст
        останнього кроку, якщо нового мовлення не було), а результат —
        зафіксовані сегменти + хвіст."""
        self.is_busy.set()
        state = StreamState()  # новий стан на КОЖНЕ натискання
        step_started_at = time.perf_counter()

        try:
            while True:
                released = self._wait_step(step_started_at)
                step_started_at = time.perf_counter()

                audio = self.recorder.stop() if released else self.recorder.snapshot()
                duration = len(audio) / SAMPLE_RATE
                raw_audio = audio
                muted_share = 0.0
                if VOICE_GATE and len(audio):
                    # Далі ВСЕ (пошук пауз, перевірка мовлення, ASR) бачить
                    # аудіо, де фон обнулено. Гейт рахується по всьому запису,
                    # тож поріг стабільний між кроками.
                    audio, _gate_thr, muted_share = apply_voice_gate(audio)

                if released and duration < MIN_SECONDS:
                    print("\r" + " " * 40 + "\r(занадто коротко, пропущено)")
                    self.overlay.display("(занадто коротко)", state="warning", autohide_ms=OVERLAY_TOO_SHORT_MS)
                    return
                if not released and duration < STEP_MIN_AUDIO_SEC:
                    continue
                #Контекст для розпізнавання.
                window = audio[state.committed_len:]


                # ================= проміжні кроки (клавіша затиснута) =================
                if not released:
                    # 1) Є пауза: фіксуємо шматок до неї (цей прохід замість кроку).
                    commit_point = find_commit_point(window)
                    if commit_point is not None:
                        cut, soft_cut = commit_point
                        try:
                            segment = self._pick_commit_text(window[:cut], cut, state, duration)
                        except Exception as exc:  # noqa: BLE001
                            print(f"\n(помилка розпізнавання: {exc})", file=sys.stderr)
                            continue
                        state.commit(segment, cut, audio, soft_cut)
                        self.overlay.display((state.joined() or "Слухаю...") + self.mode_suffix(),
                                             state="listening")
                        continue

                    text = state.joined(state.tail_text)
                    self.events.emit("partial_text", text=text)
                    self.overlay.display((text or "Слухаю...") + self.mode_suffix(), state="listening")
                    continue
                # ================= фінал (клавішу відпустили) =================
                self.events.emit("final_audio", raw_audio=raw_audio, muted_share=muted_share)
                if state.tail_valid and should_skip_recognition(audio, state.last_audio_len):
                    # Після останнього кроку нового мовлення не було — хвіст уже розпізнано.
                    self.events.emit("final_skipped")
                elif has_speech(window):
                    print(" розпізнаю...", end="", flush=True)
                    self.overlay.display("Розпізнаю...", state="processing")
                    try:
                        final_tail = self.asr_pass(window, "фінал", duration)
                    except Exception as exc:  # noqa: BLE001
                        print(f"\n(помилка розпізнавання: {exc})", file=sys.stderr)
                        final_tail = ""
                    # Фінал порожній (шум у кінці / фільтр), а крок раніше
                    # щось розпізнав — віддаємо текст кроку, а не нічого.
                    if final_tail:
                        state.tail_text = final_tail
                # else: нового мовлення у хвості немає — лишаємо текст останнього кроку

                recognized = state.joined(state.tail_text)
                output = self.get_translated_text(recognized)

                self.show_result(recognized, output)
                if output:
                    self.deliver_text(output)

                self.events.emit("final_done")
                return
        finally:
            self.is_busy.clear()

    # ---- хоткеї ------------------------------------------------------------

    def on_press(self) -> None:
        if self.is_recording.is_set() or self.is_busy.is_set():
            return
        self.wake_event.clear()
        self.is_recording.set()
        self.recorder.start()
        print("запис...", end="", flush=True)
        self.overlay.display("Слухаю..." + self.mode_suffix(), state="listening")
        threading.Thread(target=self.transcribe_loop, daemon=True).start()

    def on_release(self) -> None:
        self.is_recording.clear()
        self.wake_event.set()

    def on_toggle_translate(self) -> None:
        if self.translate_enabled.is_set():
            self.translate_enabled.clear()
            print("\nРежим: ОРИГІНАЛ (без перекладу)")
            self.overlay.display("Режим: оригінал", state="info", autohide_ms=OVERLAY_TOO_SHORT_MS)
        else:
            self.translate_enabled.set()
            print(f"\nРежим: ПЕРЕКЛАД → {TRANSLATE_TARGET_LANG.upper()}")
            self.overlay.display(
                f"Режим: переклад → {TRANSLATE_TARGET_LANG.upper()}",
                state="info",
                autohide_ms=OVERLAY_TOO_SHORT_MS,
            )

    # Колбеки потоку хуку: мусять бути миттєвими (лише emit у черзі Qt).
    def _hook_press(self) -> None:
        self.bridge.dictate_pressed.emit()

    def _hook_release(self) -> None:
        self.bridge.dictate_released.emit()

    # ---- запуск ------------------------------------------------------------

    def _start_timer(self, interval_ms: int, slot: Callable[[], None]) -> None:
        timer = QTimer()
        timer.timeout.connect(slot)
        timer.start(interval_ms)
        self._timers.append(timer)   # тримаємо посилання, інакше GC зупинить таймер

    def run(self) -> int:
        import signal  # noqa: PLC0415

        # Методи підключаються ТУТ (а не в __init__), щоб зовнішні обгортки,
        # поставлені між __init__ і run(), уже були на місці.
        self.bridge = HotkeyBridge()
        self.bridge.dictate_pressed.connect(self.on_press, Qt.QueuedConnection)
        self.bridge.dictate_released.connect(self.on_release, Qt.QueuedConnection)
        self.bridge.translate_toggled.connect(self.on_toggle_translate, Qt.QueuedConnection)

        # Порожній таймер будить event loop, щоб Ctrl+C у консолі долітав до Python.
        self._start_timer(200, lambda: None)

        self.hotkeys = HotkeyManager()
        self.hotkeys.add(HOTKEY, self._hook_press, self._hook_release)
        self.hotkeys.add(TRANSLATE_TOGGLE_HOTKEY, self.bridge.translate_toggled.emit)
        self.hotkeys.start()

        self._start_timer(150, self.hotkeys.check_stuck)

        print(f"\nГотово. Тримайте [{HOTKEY.upper()}] — говоріть, відпустіть — отримаєте текст.")
        print(f"[{TRANSLATE_TOGGLE_HOTKEY.upper()}] — перемкнути режим переклад/оригінал.")
        print("Ctrl+C — вихід.\n")

        signal.signal(signal.SIGINT, lambda *_: self.shutdown())
        self.qt_app.exec()
        return 0

    def shutdown(self) -> None:
        print("\nЗупинено.")
        if self.hotkeys is not None:
            self.hotkeys.stop()
        self.recorder.close()
        self.qt_app.quit()

def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    profile = PROFILE
    if "--profile" in argv:
        profile = True
    if "--no-profile" in argv:
        profile = False
    capture = CAPTURE
    if "--capture" in argv:
        capture = True
    if "--no-capture" in argv:
        capture = False
    app = DictationApp()
    if profile:
        from profiling import attach_profiler  # noqa: PLC0415

        attach_profiler(app)   # до load(): onnx-обгортка; до run(): хоткеї бачать обгортки
    if capture:
        from App.CaptureDataset_ForTests.dataset_capture import attach_capture  # noqa: PLC0415


        attach_capture(app)  # до load() і run(); шлях — dataset_dir_from_env(), як у тестах
    app.load()
    return app.run()


if __name__ == "__main__":
    raise SystemExit(main())