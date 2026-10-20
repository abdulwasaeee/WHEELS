
import time
from collections import Counter, deque

import cv2

from vision.hand_tracker import HandTracker
from vision.gesture_classifier import GestureClassifier


# ============================================================
# GESTURE SMOOTHING
# ============================================================

class GestureSmoother:
    def __init__(self, window_size=7, required_votes=5):
        self.history = deque(maxlen=window_size)
        self.required_votes = required_votes
        self.stable_value = None

    def update(self, value):
        self.history.append(value)

        if len(self.history) < self.required_votes:
            return self.stable_value

        counts = Counter(self.history)
        candidate, votes = counts.most_common(1)[0]

        if votes >= self.required_votes:
            self.stable_value = candidate

        return self.stable_value

    def clear(self):
        self.history.clear()
        self.stable_value = None


# ============================================================
# DASHBOARD HELPERS
# ============================================================

def draw_text(
    frame,
    text,
    x,
    y,
    color=(255, 255, 255),
    scale=0.65
):
    cv2.putText(
        frame,
        str(text),
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
        cv2.LINE_AA
    )


def draw_throttle_bar(frame, throttle):
    x, y = 20, 170
    width, height = 250, 15

    cv2.rectangle(
        frame,
        (x, y),
        (x + width, y + height),
        (100, 100, 100),
        2
    )

    fill_width = int(width * throttle)

    color = (
        (0, 0, 255)
        if throttle >= 0.8
        else (0, 255, 0)
    )

    if fill_width > 0:
        cv2.rectangle(
            frame,
            (x, y),
            (x + fill_width, y + height),
            color,
            -1
        )


def draw_steering_indicator(frame, steering):
    height, width = frame.shape[:2]

    center_x = width // 2
    y = height - 75

    cv2.line(
        frame,
        (center_x - 110, y),
        (center_x + 110, y),
        (255, 255, 255),
        2
    )

    cv2.line(
        frame,
        (center_x, y - 12),
        (center_x, y + 12),
        (255, 255, 255),
        2
    )

    steering_x = int(center_x + steering * 110)

    cv2.circle(
        frame,
        (steering_x, y),
        12,
        (0, 255, 0),
        -1
    )


def draw_finger_states(frame, states, x=20, y=230):
    """
    Show which fingers are currently detected as extended.
    Useful for debugging false 4-to-5 finger counts.
    """
    if not states:
        draw_text(
            frame,
            "FINGERS: NOT DETECTED",
            x,
            y,
            (150, 150, 150),
            0.55
        )
        return

    names = ["thumb", "index", "middle", "ring", "pinky"]

    for i, name in enumerate(names):
        active = states.get(name, False)

        color = (
            (0, 255, 0)
            if active
            else (100, 100, 100)
        )

        draw_text(
            frame,
            f"{name.upper()}: {'UP' if active else 'DOWN'}",
            x,
            y + i * 28,
            color,
            0.5
        )


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    print("=" * 50)
    print("          WHEELS AI - GESTURE ENGINE")
    print("=" * 50)

    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not camera.isOpened():
        print("[ERROR] Webcam unavailable.")
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    tracker = None
    classifier = GestureClassifier()

    speed_smoother = GestureSmoother(
        window_size=7,
        required_votes=5
    )

    steering_smoother = GestureSmoother(
        window_size=7,
        required_votes=5
    )

    previous_time = time.perf_counter()
    fps = 0.0

    try:
        tracker = HandTracker()

        print("[WHEELS] Gesture engine online.")
        print("[WHEELS] RIGHT HAND: Speed / Handbrake")
        print("[WHEELS] LEFT HAND: Steering")
        print("[WHEELS] LEFT THUMB: RIGHT")
        print("[WHEELS] LEFT PINKY: LEFT")
        print("[WHEELS] Press Q to exit.")

        while True:
            success, frame = camera.read()

            if not success:
                print("[ERROR] Camera frame unavailable.")
                break

            # Mirror webcam
            frame = cv2.flip(frame, 1)

            # Detect hands
            frame, result = tracker.process(frame)

            speed_data = None
            steering_data = None
            finger_states = None

            # ----------------------------------------
            # HAND IDENTIFICATION
            # ----------------------------------------
            for i, landmarks in enumerate(
                result.hand_landmarks
            ):
                hand_label = (
                    result.handedness[i][0].category_name
                )

                # RIGHT HAND: SPEED
                if hand_label == "Right":
                    speed_data = classifier.classify_left(
                        landmarks
                    )

                    finger_states = speed_data.get(
                        "finger_states"
                    )

                # LEFT HAND: STEERING
                elif hand_label == "Left":
                    steering_data = classifier.classify_right(
                        landmarks
                    )

            # ----------------------------------------
            # SAFE DEFAULT VALUES
            # ----------------------------------------
            throttle = 0.0
            brake = False
            steering = 0.0

            speed_status = "NOT DETECTED"
            steering_status = "NOT DETECTED"

            # ----------------------------------------
            # RIGHT HAND: THROTTLE / HANDBRAKE
            # ----------------------------------------
            if speed_data is not None:
                level = speed_smoother.update(
                    speed_data["fingers"]
                )

                if level is not None:
                    throttle = level / 5.0
                    brake = level == 0

                    speed_status = (
                        "HANDBRAKE"
                        if brake
                        else f"SPEED LEVEL {level}"
                    )
                else:
                    speed_status = "STABILIZING"
            else:
                speed_smoother.clear()

            # ----------------------------------------
            # LEFT HAND: STEERING
            # ----------------------------------------
            if steering_data is not None:
                original_steering = (
                    steering_data["steering"]
                )

                # Reverse original mapping:
                # Thumb -> RIGHT
                # Pinky -> LEFT
                stable_steering = steering_smoother.update(
                    -original_steering
                )

                if stable_steering is not None:
                    steering = stable_steering

                    if steering < 0:
                        steering_status = "LEFT (PINKY)"
                    elif steering > 0:
                        steering_status = "RIGHT (THUMB)"
                    else:
                        steering_status = "STRAIGHT"
                else:
                    steering_status = "STABILIZING"
            else:
                steering_smoother.clear()

            # ----------------------------------------
            # FPS CALCULATION
            # ----------------------------------------
            current_time = time.perf_counter()

            dt = max(
                current_time - previous_time,
                0.001
            )

            previous_time = current_time

            instant_fps = 1.0 / dt

            fps = (
                instant_fps
                if fps == 0
                else fps * 0.85 + instant_fps * 0.15
            )

            height, width = frame.shape[:2]

            # ----------------------------------------
            # DASHBOARD BACKGROUND
            # ----------------------------------------
            overlay = frame.copy()

            cv2.rectangle(
                overlay,
                (0, 0),
                (width, 205),
                (20, 20, 20),
                -1
            )

            cv2.rectangle(
                overlay,
                (0, 205),
                (275, 390),
                (20, 20, 20),
                -1
            )

            frame = cv2.addWeighted(
                overlay,
                0.65,
                frame,
                0.35,
                0
            )

            # ----------------------------------------
            # DASHBOARD TEXT
            # ----------------------------------------
            draw_text(
                frame,
                "WHEELS AI - GESTURE CONTROL",
                20,
                35,
                (0, 255, 0),
                0.85
            )

            draw_text(
                frame,
                f"RIGHT HAND (SPEED): {speed_status}",
                20,
                75,
                (0, 255, 255)
            )

            draw_text(
                frame,
                f"THROTTLE: {int(throttle * 100)}%",
                20,
                110
            )

            draw_text(
                frame,
                f"LEFT HAND (STEERING): {steering_status}",
                20,
                145,
                (255, 255, 0)
            )

            draw_text(
                frame,
                f"FPS: {fps:.1f}",
                max(20, width - 155),
                35
            )

            # ----------------------------------------
            # THROTTLE BAR
            # ----------------------------------------
            draw_throttle_bar(
                frame,
                throttle
            )

            # ----------------------------------------
            # FINGER DEBUGGING
            # ----------------------------------------
            draw_finger_states(
                frame,
                finger_states,
                x=20,
                y=230
            )

            # ----------------------------------------
            # STEERING INDICATOR
            # ----------------------------------------
            draw_steering_indicator(
                frame,
                steering
            )

            draw_text(
                frame,
                "GESTURE MODE | PRESS Q TO EXIT",
                20,
                height - 20,
                (255, 255, 255),
                0.55
            )

            # ----------------------------------------
            # DISPLAY
            # ----------------------------------------
            cv2.imshow(
                "WHEELS - Gesture Engine",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("[WHEELS] Interrupted.")

    finally:
        print("[WHEELS] Shutting down...")

        camera.release()

        if tracker is not None:
            tracker.close()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
