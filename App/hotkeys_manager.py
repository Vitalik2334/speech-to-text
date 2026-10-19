"""
hotkeys.py — глобальні хоткеї з утриманням (push-to-talk) для Windows
на базі pynput + win32_event_filter, без keyboard.is_pressed().

Чому не бібліотека `keyboard`:
  * keyboard.is_pressed() читає ВНУТРІШНІЙ стан бібліотеки, а не ОС.
    Цей стан ламається, коли подію придушено, коли хук втратив KEY_UP
    (таймаут LowLevelHooksTimeout під навантаженням ASR) і коли
    keyboard.write() сам генерує Shift-натискання.
  * keyboard.write(restore_state_after=True) після друку "повертає"
    клавіші, які бібліотека ВВАЖАЄ натиснутими. Якщо стан застарів —
    вона надсилає в ОС Shift DOWN без UP => Shift залипає в системі.

Що робить цей модуль:
  1. Свій стан натиснутих клавіш (vkCode), без is_pressed().
  2. Ігнорує injected-події (наш власний друк тексту, AutoHotkey тощо).
  3. Модифікатори (Shift/Ctrl/Alt) НІКОЛИ не придушуються — ОС завжди
     знає їхній реальний стан, тому залипнути на рівні системи вони
     не можуть.
  4. Придушується тільки клавіша-тригер ("1"/"2"): її DOWN, автоповтори
     і ПАРНИЙ UP — ОС не бачить жодної половини натискання.
  5. Колбек хуку мінімальний (кілька операцій із set) — щоб не впертися
     в LowLevelHooksTimeout, поки модель розпізнавання вантажить CPU.
  6. check_stuck() — сторож для QTimer: звіряє модифікатори з реальним
     станом ОС (GetAsyncKeyState) і відпускає "завислий" хоткей, якщо
     хук пропустив KEY_UP або Windows тихо зняла хук.

Залежність:  uv add pynput
PyInstaller: --hidden-import pynput.keyboard._win32 --hidden-import pynput.mouse._win32
"""
from __future__ import annotations

import ctypes
import threading
from dataclasses import dataclass
from typing import Callable

from pynput import keyboard as pkb

_user32 = ctypes.windll.user32

WM_KEYDOWN, WM_KEYUP, WM_SYSKEYDOWN, WM_SYSKEYUP = 0x0100, 0x0101, 0x0104, 0x0105
LLKHF_INJECTED = 0x10

# Low-level хук завжди повідомляє лівий/правий модифікатор окремим кодом.
_MODIFIERS: dict[str, tuple[int, ...]] = {
    "shift": (0xA0, 0xA1), "left shift": (0xA0,), "right shift": (0xA1,),
    "ctrl": (0xA2, 0xA3), "left ctrl": (0xA2,), "right ctrl": (0xA3,),
    "alt": (0xA4, 0xA5), "left alt": (0xA4,), "right alt": (0xA5,),
    "win": (0x5B, 0x5C),
}
_ALL_MODIFIER_VKS = frozenset(vk for group in _MODIFIERS.values() for vk in group)

_SPECIAL_KEYS = {"space": 0x20, "tab": 0x09, "pause": 0x13, "scroll lock": 0x91, "insert": 0x2D}


def _trigger_vk(name: str) -> int:
    name = name.strip().lower()
    if len(name) == 1 and name.isascii() and name.isalnum():
        return ord(name.upper())            # '0'-'9' -> 0x30-0x39, 'a'-'z' -> 0x41-0x5A
    if name.startswith("f") and name[1:].isdigit():
        return 0x6F + int(name[1:])         # f1 -> 0x70 ... f24 -> 0x87
    if name in _SPECIAL_KEYS:
        return _SPECIAL_KEYS[name]
    raise ValueError(f"Невідома клавіша-тригер: {name!r}")


@dataclass
class _Hotkey:
    modifier_groups: list[tuple[int, ...]]
    trigger: int
    on_press: Callable[[], None]
    on_release: Callable[[], None] | None
    active: bool = False    # хоткей логічно затиснутий (диктовка йде)
    blocked: bool = False   # DOWN тригера придушено -> давимо повтори і парний UP

    def modifiers_held(self, pressed: set[int]) -> bool:
        return all(any(vk in pressed for vk in group) for group in self.modifier_groups)


class HotkeyManager:
    """Комбінації виду "shift+1": останній елемент — тригер, решта — модифікатори.

    Колбеки викликаються з потоку хуку pynput — передавайте туди лише
    щось швидке й потокобезпечне (наприклад, Signal.emit Qt).
    """

    def __init__(self) -> None:
        self._hotkeys: list[_Hotkey] = []
        self._pressed: set[int] = set()
        self._lock = threading.Lock()
        self._listener: pkb.Listener | None = None

    def add(self, combo: str, on_press: Callable[[], None],
            on_release: Callable[[], None] | None = None) -> None:
        *mods, trigger = [p.strip().lower() for p in combo.split("+")]
        self._hotkeys.append(
            _Hotkey([_MODIFIERS[m] for m in mods], _trigger_vk(trigger), on_press, on_release)
        )

    def start(self) -> None:
        self._listener = pkb.Listener(win32_event_filter=self._filter)
        self._listener.start()
        self._listener.wait()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    # ---- потік хуку: має відпрацювати якнайшвидше --------------------------

    def _filter(self, msg: int, data) -> bool:
        if data.flags & LLKHF_INJECTED:
            return False  # синтетичні події (наш друк тексту) стан не чіпають

        vk = data.vkCode
        is_down = msg in (WM_KEYDOWN, WM_SYSKEYDOWN)
        calls: list[Callable[[], None]] = []
        suppress = False

        with self._lock:
            if is_down:
                self._pressed.add(vk)
            else:
                self._pressed.discard(vk)

            for hk in self._hotkeys:
                if vk == hk.trigger:
                    if is_down:
                        if hk.blocked:
                            suppress = True                 # автоповтор утримання
                        elif hk.modifiers_held(self._pressed):
                            hk.blocked = hk.active = True
                            suppress = True
                            calls.append(hk.on_press)
                    else:
                        if hk.blocked:
                            hk.blocked = False
                            suppress = True                 # парний UP до придушеного DOWN
                        if hk.active:
                            hk.active = False
                            if hk.on_release:
                                calls.append(hk.on_release)
                elif (hk.active and not is_down and vk in _ALL_MODIFIER_VKS
                      and not hk.modifiers_held(self._pressed)):
                    # Shift відпустили раніше за "1": диктовку завершуємо,
                    # сам Shift UP пропускаємо в ОС (модифікатори не давимо).
                    hk.active = False
                    if hk.on_release:
                        calls.append(hk.on_release)

        for call in calls:
            call()

        if suppress and self._listener is not None:
            self._listener.suppress_event()   # кидає виняток всередині pynput — це нормально
        return False  # у on_press/on_release pynput подію не передаємо, в ОС — так

    # ---- GUI-потік (QTimer): сторож від завислого стану ---------------------

    def check_stuck(self) -> None:
        """Звіряє модифікатори з реальним станом ОС. Тригер так перевіряти
        не можна: його DOWN придушено, тож для ОС він завжди "відпущений"."""
        calls: list[Callable[[], None]] = []
        with self._lock:
            for vk in [v for v in self._pressed if v in _ALL_MODIFIER_VKS]:
                if not (_user32.GetAsyncKeyState(vk) & 0x8000):
                    self._pressed.discard(vk)
            for hk in self._hotkeys:
                if hk.active and not hk.modifiers_held(self._pressed):
                    hk.active = False
                    hk.blocked = False
                    if hk.on_release:
                        calls.append(hk.on_release)
        for call in calls:
            call()