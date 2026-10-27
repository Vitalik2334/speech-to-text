"""
Записує test.wav з вашого мікрофона у форматі, який очікує Parakeet:
16 kHz, моно, PCM_16.

Не просить у драйвера 16 kHz напряму (MME цього часто не вміє і падає
з PaErrorCode -9999). Замість цього шукає робочу комбінацію
пристрій + частота + канали, віддаючи перевагу WASAPI, записує в рідному
форматі мікрофона і сам зводить результат до 16 kHz моно.

Запуск:
    python 00_make_test_wav.py            # 6 секунд, автовибір пристрою
    python 00_make_test_wav.py 10         # 10 секунд
    python 00_make_test_wav.py 6 2        # 6 секунд з пристрою №2
    python 00_make_test_wav.py --list     # тільки показати пристрої
"""

import sys
import wave

import numpy as np
import sounddevice as sd

TARGET_RATE = 16_000
OUT = "test.wav"

PHRASE = (
    "Один, два, три, чотири, п'ять. "
    "Це тестовий запис для перевірки розпізнавання української мови."
)

FAIL_HINT = """
Не вдалося відкрити жоден мікрофон.

Перевірте по черзі:

 1. Параметри → Конфіденційність і безпека → Мікрофон
    Увімкніть "Доступ до мікрофона" І "Дозволити класичним застосункам
    доступ до мікрофона". Python і PyCharm — саме класичні застосунки.
    Це причина номер один для помилки -9999.

 2. Закрийте Zoom, Teams, Discord, OBS і вкладки браузера з дзвінками —
    вони можуть тримати мікрофон в ексклюзивному режимі.

 3. Панель керування → Звук → Запис
    Переконайтеся, що потрібний мікрофон увімкнений і призначений
    пристроєм за замовчуванням.

 4. Запустіть 'python 00_make_test_wav.py --list' і спробуйте вручну
    конкретний індекс: 'python 00_make_test_wav.py 6 <індекс>'
"""


def host_api_name(device_info) -> str:
    return sd.query_hostapis()[device_info["hostapi"]]["name"]


def list_inputs() -> None:
    print(f"{'idx':>4}  {'канали':>6}  {'частота':>8}  {'API':<12}  назва")
    print("-" * 78)
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            print(
                f"{i:>4}  {dev['max_input_channels']:>6}  "
                f"{int(dev['default_samplerate']):>8}  "
                f"{host_api_name(dev):<12}  {dev['name']}"
            )


def candidate_devices(explicit: int | None) -> list[int]:
    """Індекси входів у порядку спадання шансів на успіх."""
    if explicit is not None:
        return [explicit]

    devices = sd.query_devices()
    apis = sd.query_hostapis()
    cands: list[int] = []

    # WASAPI вміє передискретизацію в shared-режимі, тому пробуємо його першим
    for api in apis:
        if "WASAPI" in api["name"].upper() and api["default_input_device"] >= 0:
            cands.append(api["default_input_device"])

    default_in = sd.default.device[0]
    if isinstance(default_in, int) and default_in >= 0:
        cands.append(default_in)

    inputs = [i for i, d in enumerate(devices) if d["max_input_channels"] > 0]
    inputs.sort(key=lambda i: 0 if "WASAPI" in host_api_name(devices[i]).upper() else 1)
    cands.extend(inputs)

    return list(dict.fromkeys(cands))


def probe(device: int, rate: int, channels: int) -> bool:
    """Коротке реальне відкриття потоку — надійніше за check_input_settings."""
    try:
        with sd.InputStream(
            device=device, samplerate=rate, channels=channels, dtype="int16"
        ):
            pass
        return True
    except Exception:
        return False


def find_working(explicit: int | None) -> tuple[int, int, int] | None:
    for device in candidate_devices(explicit):
        try:
            info = sd.query_devices(device)
        except Exception:
            continue
        if info["max_input_channels"] < 1:
            continue

        rates = list(
            dict.fromkeys([TARGET_RATE, int(info["default_samplerate"]), 48000, 44100])
        )
        channel_opts = [1]
        if info["max_input_channels"] >= 2:
            channel_opts.append(2)

        for channels in channel_opts:
            for rate in rates:
                if probe(device, rate, channels):
                    return device, rate, channels
    return None


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Передискретизація через спектр — без aliasing і без scipy."""
    if sr_in == sr_out:
        return x
    n_in = len(x)
    n_out = int(round(n_in * sr_out / sr_in))
    spectrum = np.fft.rfft(x)
    out_bins = n_out // 2 + 1
    resized = np.zeros(out_bins, dtype=complex)
    keep = min(len(spectrum), out_bins)
    resized[:keep] = spectrum[:keep]
    return np.fft.irfft(resized, n=n_out) * (n_out / n_in)


def main() -> int:
    args = [a for a in sys.argv[1:]]

    if args and args[0] in ("--list", "-l"):
        list_inputs()
        return 0

    seconds = float(args[0]) if len(args) > 0 else 6.0
    explicit = int(args[1]) if len(args) > 1 else None

    print("Доступні входи:\n")
    list_inputs()

    print("\nПошук робочої конфігурації ...")
    found = find_working(explicit)
    if found is None:
        print(FAIL_HINT)
        return 1

    device, rate, channels = found
    info = sd.query_devices(device)
    print(
        f"  ✔ [{device}] {info['name']}\n"
        f"    {host_api_name(info)}, {rate} Гц, {channels} кан."
    )
    if rate != TARGET_RATE:
        print(f"    Запис у рідному форматі, зведення до {TARGET_RATE} Гц вже в Python.")

    print(f"\nПрочитайте вголос:\n\n    {PHRASE}\n")
    input("Enter — почати запис...")

    print(f"Запис {seconds:.0f} с ...")
    try:
        raw = sd.rec(
            int(seconds * rate),
            samplerate=rate,
            channels=channels,
            dtype="int16",
            device=device,
        )
        sd.wait()
    except sd.PortAudioError as e:
        print(f"\nПотік відкрився на пробі, але впав на записі: {e}")
        print(FAIL_HINT)
        return 1

    audio = raw.astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.mean(axis=1)
    else:
        audio = audio[:, 0]

    audio = resample(audio, rate, TARGET_RATE)
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767).astype(np.int16)

    with wave.open(OUT, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(TARGET_RATE)
        w.writeframes(pcm.tobytes())

    peak = int(np.abs(pcm).max())
    rms = float(np.sqrt(np.mean((pcm.astype(np.float32) / 32768) ** 2)))

    print(f"\nЗбережено {OUT}  ({len(pcm) / TARGET_RATE:.2f} с)")
    print(f"  пік: {peak} / 32767   RMS: {rms:.4f}")

    if peak < 1500:
        print("  ⚠ Сигнал дуже тихий — підніміть гучність мікрофона і перезапишіть.")
    elif peak >= 32700:
        print("  ⚠ Кліпінг — знизьте гучність мікрофона.")
    else:
        print("  ✔ Рівень нормальний.")

    print("\nДалі: python check_wav.py test.wav")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())