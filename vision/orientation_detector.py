
import math
import time


class OrientationDetector:
    """
    Detects when a hand's fingers point toward the
    bottom of the mirrored webcam image.

    This is an image-plane orientation estimate,
    not full 3D wrist rotation.
    """

    FINGER_TIPS = (8, 12, 16, 20)
    FINGER_PIPS = (6, 10, 14, 18)
    FINGER_MCPS = (5, 9, 13, 17)

    def __init__(
        self,
        downward_angle_degrees=35,
        min_extended_fingers=2,
    ):
        self.angle_limit = downward_angle_degrees
        self.min_extended = min_extended_fingers

    @staticmethod
    def distance(a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    @staticmethod
    def joint_angle(a, b, c):
        abx, aby = a.x - b.x, a.y - b.y
        cbx, cby = c.x - b.x, c.y - b.y

        denominator = math.hypot(abx, aby) * math.hypot(cbx, cby)

        if denominator < 1e-8:
            return 0.0

        cosine = (abx * cbx + aby * cby) / denominator
        cosine = max(-1.0, min(1.0, cosine))

        return math.degrees(math.acos(cosine))

    def classify(self, landmarks):
        wrist = landmarks[0]
        middle_mcp = landmarks[9]

        # Wrist -> middle knuckle indicates palm direction.
        dx = middle_mcp.x - wrist.x
        dy = middle_mcp.y - wrist.y

        # Image y increases downward.
        angle = math.degrees(math.atan2(dx, dy))

        palm_points_down = (
            dy > 0.04
            and abs(angle) <= self.angle_limit
        )

        extended_count = 0
        downward_tips = 0

        for mcp_id, pip_id, tip_id in zip(
            self.FINGER_MCPS,
            self.FINGER_PIPS,
            self.FINGER_TIPS,
        ):
            mcp = landmarks[mcp_id]
            pip = landmarks[pip_id]
            tip = landmarks[tip_id]

            straight = self.joint_angle(
                mcp, pip, tip
            ) > 150

            extended = (
                straight
                and self.distance(wrist, tip)
                > self.distance(wrist, pip) * 1.10
            )

            if extended:
                extended_count += 1

                if tip.y > mcp.y + 0.03:
                    downward_tips += 1

        is_down = (
            palm_points_down
            and extended_count >= self.min_extended
            and downward_tips >= self.min_extended
        )

        return {
            "down": is_down,
            "angle": angle,
            "extended_fingers": extended_count,
        }


class HoldGesture:
    """
    Debounced hold gesture with cooldown.

    Fires once after the hold duration.
    Must be released before it can fire again.
    """

    def __init__(
        self,
        hold_seconds=1.2,
        cooldown_seconds=2.0,
        release_seconds=0.35,
    ):
        self.hold_seconds = hold_seconds
        self.cooldown_seconds = cooldown_seconds
        self.release_seconds = release_seconds

        self.started_at = None
        self.released_at = None
        self.last_trigger = -float("inf")
        self.armed = True
        self.triggered = False

    def update(self, active):
        now = time.monotonic()
        self.triggered = False

        if active:
            self.released_at = None

            if self.started_at is None:
                self.started_at = now

            elapsed = now - self.started_at

            if (
                self.armed
                and elapsed >= self.hold_seconds
                and now - self.last_trigger
                >= self.cooldown_seconds
            ):
                self.triggered = True
                self.armed = False
                self.last_trigger = now

        else:
            self.started_at = None

            if self.released_at is None:
                self.released_at = now

            if (
                now - self.released_at
                >= self.release_seconds
            ):
                self.armed = True

        return self.triggered

    def progress(self):
        if self.started_at is None:
            return 0.0

        elapsed = time.monotonic() - self.started_at

        return min(
            1.0,
            elapsed / self.hold_seconds
        )
