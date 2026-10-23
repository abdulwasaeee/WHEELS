
import threading
import time

from pynput import keyboard


class GameController:
    def __init__(self):
        self.lock = threading.RLock()
        self.keyboard = keyboard.Controller()

        self.enabled = False
        self.emergency = False
        self.autodrive = False
        self.running = True

        self.held_keys = set()

        self.keys = {
            "forward": keyboard.Key.up,
            "reverse": keyboard.Key.down,
            "left": keyboard.Key.left,
            "right": keyboard.Key.right,
            "handbrake": keyboard.Key.space,
            "autodrive": "f",
        }

        self.pulse_period = 0.15
        self.watchdog_timeout = 0.5
        self.last_update = time.monotonic()

        self.last_f8 = 0.0

        self.listener = keyboard.Listener(
            on_press=self._on_press
        )
        self.listener.start()

        self.watchdog = threading.Thread(
            target=self._watchdog_loop,
            daemon=True
        )
        self.watchdog.start()

        print("[WHEELS] pynput controller ready.")
        print("[WHEELS] F8: Enable/disable")
        print("[WHEELS] ESC: Emergency stop")

    def _press(self, key):
        if key not in self.held_keys:
            self.keyboard.press(key)
            self.held_keys.add(key)

    def _release(self, key):
        if key in self.held_keys:
            self.keyboard.release(key)
            self.held_keys.remove(key)

    def release_all(self):
        with self.lock:
            for key in list(self.held_keys):
                self._release(key)

    def _tap(self, key):
        self.keyboard.press(key)
        try:
            time.sleep(0.08)
        finally:
            self.keyboard.release(key)

    def _on_press(self, key):
        with self.lock:
            if key == keyboard.Key.esc:
                self.emergency = True
                self.enabled = False
                self.release_all()
                print("[WHEELS] EMERGENCY STOP")
                return False

            if key == keyboard.Key.f8:
                now = time.monotonic()

                if now - self.last_f8 < 0.4:
                    return

                self.last_f8 = now
                self.enabled = not self.enabled
                self.release_all()
                self.last_update = now

                print(
                    "[WHEELS] Controls:",
                    "ENABLED" if self.enabled
                    else "DISABLED"
                )

    def toggle_autodrive(self):
        with self.lock:
            if not self.enabled or self.emergency:
                return False

            self.release_all()

            # The game's F key is assumed to toggle
            # its native autodrive feature.
            self._tap(self.keys["autodrive"])

            self.autodrive = not self.autodrive
            self.last_update = time.monotonic()

            print(
                "[WHEELS] Autodrive requested:",
                "ON" if self.autodrive else "OFF"
            )

            return True

    def _pulse(self, key, amount, phase):
        amount = max(0.0, min(1.0, amount))

        if amount > 0 and phase < amount:
            self._press(key)
        else:
            self._release(key)

    def update(
        self,
        throttle=0.0,
        steering=0.0,
        handbrake=False,
        reverse=False,
    ):
        with self.lock:
            self.last_update = time.monotonic()

            if (
                not self.enabled
                or self.emergency
                or self.autodrive
            ):
                self.release_all()
                return

            throttle = max(0.0, min(1.0, throttle))
            steering = max(-1.0, min(1.0, steering))

            phase = (
                time.monotonic() % self.pulse_period
            ) / self.pulse_period

            # Driving priority:
            # reverse > handbrake > forward
            if reverse:
                self._release(self.keys["forward"])
                self._release(self.keys["handbrake"])
                self._press(self.keys["reverse"])

            elif handbrake:
                self._release(self.keys["forward"])
                self._release(self.keys["reverse"])
                self._press(self.keys["handbrake"])

            else:
                self._release(self.keys["reverse"])
                self._release(self.keys["handbrake"])

                self._pulse(
                    self.keys["forward"],
                    throttle,
                    phase
                )

            # Proportional steering
            if steering < -0.02:
                self._release(self.keys["right"])
                self._pulse(
                    self.keys["left"],
                    abs(steering),
                    phase
                )

            elif steering > 0.02:
                self._release(self.keys["left"])
                self._pulse(
                    self.keys["right"],
                    steering,
                    phase
                )

            else:
                self._release(self.keys["left"])
                self._release(self.keys["right"])

    def _watchdog_loop(self):
        while self.running:
            with self.lock:
                stale = (
                    time.monotonic() - self.last_update
                    > self.watchdog_timeout
                )

                if stale:
                    self.release_all()

            time.sleep(0.05)

    def close(self):
        with self.lock:
            self.running = False
            self.enabled = False
            self.release_all()

        self.listener.stop()
        self.watchdog.join(timeout=1.0)
