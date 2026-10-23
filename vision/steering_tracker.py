
import time


class SteeringTracker:
    def __init__(
        self,
        dead_zone=0.035,
        max_offset=0.22,
        smoothing=0.22,
    ):
        self.dead_zone = dead_zone
        self.max_offset = max_offset
        self.smoothing = smoothing

        self.neutral_x = None
        self.filtered_steering = 0.0
        self.current_palm_x = None
        self.last_seen = None

    @staticmethod
    def get_palm_center(landmarks):
        palm_ids = [0, 5, 9, 13, 17]

        x = sum(
            landmarks[i].x for i in palm_ids
        ) / len(palm_ids)

        y = sum(
            landmarks[i].y for i in palm_ids
        ) / len(palm_ids)

        return x, y

    def calibrate(self):
        if self.current_palm_x is None:
            print("[WHEELS] Cannot calibrate: left hand missing.")
            return False

        self.neutral_x = self.current_palm_x
        self.filtered_steering = 0.0

        print(
            f"[WHEELS] Steering calibrated: "
            f"neutral_x={self.neutral_x:.3f}"
        )
        return True

    def update(self, landmarks):
        x, y = self.get_palm_center(landmarks)

        self.current_palm_x = x
        self.last_seen = time.monotonic()

        if self.neutral_x is None:
            return 0.0

        offset = x - self.neutral_x

        if abs(offset) <= self.dead_zone:
            target = 0.0
        else:
            direction = 1.0 if offset > 0 else -1.0

            magnitude = (
                abs(offset) - self.dead_zone
            ) / max(
                self.max_offset - self.dead_zone,
                1e-6
            )

            target = direction * min(
                1.0,
                max(0.0, magnitude)
            )

        self.filtered_steering += (
            self.smoothing
            * (target - self.filtered_steering)
        )

        if abs(self.filtered_steering) < 0.02:
            self.filtered_steering = 0.0

        return self.filtered_steering

    def reset_tracking(self):
        self.current_palm_x = None
        self.filtered_steering = 0.0
        self.last_seen = None

    @property
    def calibrated(self):
        return self.neutral_x is not None
