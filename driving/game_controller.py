
import time
import threading

import pydirectinput
from pynput import keyboard


class GameController:
    """
    Sends driving inputs to Slow Roads.

    F8: Toggle game control
    ESC: Emergency stop and exit
    """

    def __init__(self):
        self.enabled = False
        self.emergency = False

        self.lock = threading.RLock()
        self.held_keys = set()

        self.keys = {
            "accelerate": "w",
            "brake": "s",
            "left": "a",
            "right": "d",
        }

        self.pulse_period = 0.20
        self.last_update = time.monotonic()

        pydirectinput.PAUSE = 0

        self.listener = keyboard.Listener(
            on_press=self._on_press
        )
        self.listener.start()

    def _on_press(self, key):
        with self.lock:
            if key == keyboard.Key.esc:
                self.emergency = True
                self.enabled = False
                self.release_all()
                return False

            if key == keyboard.Key.f8:
                self.enabled = not self.enabled
                self.release_all()

                print(
                    "[WHEELS] Game control:",
                    "ENABLED" if self.enabled else "DISABLED"
                )

    def _press(self, key):
        if key not in self.held_keys:
            pydirectinput.keyDown(key)
            self.held_keys.add(key)

    def _release(self, key):
        if key in self.held_keys:
            pydirectinput.keyUp(key)
            self.held_keys.remove(key)

    def release_all(self):
        with self.lock:
            for key in list(self.held_keys):
                self._release(key)

    def update(self, throttle, steering, brake):
        with self.lock:
            if not self.enabled or self.emergency:
                self.release_all()
                return

            throttle = max(0.0, min(1.0, throttle))

            # Steering
            if steering < -0.5:
                self._press(self.keys["left"])
                self._release(self.keys["right"])

            elif steering > 0.5:
                self._press(self.keys["right"])
                self._release(self.keys["left"])

            else:
                self._release(self.keys["left"])
                self._release(self.keys["right"])

            # Braking takes priority
            if brake:
                self._release(self.keys["accelerate"])
                self._press(self.keys["brake"])
                return

            self._release(self.keys["brake"])

            # Pulse-width-modulated acceleration
            if throttle <= 0:
                self._release(self.keys["accelerate"])
                return

            phase = (
                time.monotonic() % self.pulse_period
            ) / self.pulse_period

            if phase < throttle:
                self._press(self.keys["accelerate"])
            else:
                self._release(self.keys["accelerate"])

    def close(self):
        with self.lock:
            self.enabled = False
            self.release_all()

        self.listener.stop()
