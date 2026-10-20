
import math


class GestureClassifier:
    FINGERS = {
        "index": (5, 6, 8),
        "middle": (9, 10, 12),
        "ring": (13, 14, 16),
        "pinky": (17, 18, 20),
    }

    @staticmethod
    def distance(a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    @staticmethod
    def angle(a, b, c):
        ab = (a.x - b.x, a.y - b.y)
        cb = (c.x - b.x, c.y - b.y)

        dot = ab[0] * cb[0] + ab[1] * cb[1]
        mag_ab = math.hypot(*ab)
        mag_cb = math.hypot(*cb)

        if mag_ab * mag_cb < 1e-8:
            return 0.0

        cosine = max(
            -1.0,
            min(1.0, dot / (mag_ab * mag_cb))
        )

        return math.degrees(math.acos(cosine))

    @classmethod
    def finger_states(cls, landmarks):
        states = {}
        wrist = landmarks[0]

        # Four fingers: check straightness and extension.
        for name, (base_id, pip_id, tip_id) in (
            cls.FINGERS.items()
        ):
            base = landmarks[base_id]
            pip = landmarks[pip_id]
            tip = landmarks[tip_id]

            joint_angle = cls.angle(base, pip, tip)

            extended = (
                joint_angle > 155
                and cls.distance(wrist, tip)
                > cls.distance(wrist, pip) * 1.12
            )

            states[name] = extended

        # Thumb requires separate geometry.
        thumb_mcp = landmarks[2]
        thumb_ip = landmarks[3]
        thumb_tip = landmarks[4]
        index_mcp = landmarks[5]
        pinky_mcp = landmarks[17]

        thumb_angle = cls.angle(
            thumb_mcp, thumb_ip, thumb_tip
        )

        palm_width = max(
            cls.distance(index_mcp, pinky_mcp),
            1e-6
        )

        thumb_spread = (
            cls.distance(thumb_tip, index_mcp)
            / palm_width
        )

        states["thumb"] = (
            thumb_angle > 150
            and thumb_spread > 0.65
        )

        return states

    @classmethod
    def classify_left(cls, landmarks):
        states = cls.finger_states(landmarks)
        count = sum(states.values())

        return {
            "gesture": (
                "HANDBRAKE"
                if count == 0
                else f"SPEED LEVEL {count}"
            ),
            "fingers": count,
            "throttle": count / 5.0,
            "brake": count == 0,
            "finger_states": states,
        }

    @classmethod
    def classify_right(cls, landmarks):
        states = cls.finger_states(landmarks)

        thumb_only = (
            states["thumb"]
            and not any(
                states[name]
                for name in (
                    "index", "middle", "ring", "pinky"
                )
            )
        )

        pinky_only = (
            states["pinky"]
            and not any(
                states[name]
                for name in (
                    "thumb", "index", "middle", "ring"
                )
            )
        )

        if thumb_only:
            return {
                "gesture": "STEER LEFT",
                "steering": -1.0,
            }

        if pinky_only:
            return {
                "gesture": "STEER RIGHT",
                "steering": 1.0,
            }

        return {
            "gesture": "STRAIGHT",
            "steering": 0.0,
        }
