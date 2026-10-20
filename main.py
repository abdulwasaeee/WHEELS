
import time
from collections import Counter, deque

import cv2

from vision.hand_tracker import HandTracker
from vision.gesture_classifier import GestureClassifier
from driving.game_controller import GameController


class GestureSmoother:
    def __init__(self, window_size=7, required_votes=5):
        self.history = deque(maxlen=window_size)
        self.required_votes = required_votes
        self.stable_value = None

    def update(self, value):
        self.history.append(value)
        counts = Counter(self.history)

        if counts:
            candidate, votes = counts.most_common(1)[0]
            if votes >= self.required_votes:
                self.stable_value = candidate

        return self.stable_value

    def clear(self):
        self.history.clear()
        self.stable_value = None


def draw_text(frame, text, x, y,
              color=(255, 255, 255), scale=0.65):
    cv2.putText(
        frame, str(text), (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale, color, 2, cv2.LINE_AA
    )


def draw_fingers(frame, states):
    names = ["thumb", "index", "middle", "ring", "pinky"]

    for i, name in enumerate(names):
        active = bool(states and states.get(name, False))

        color = (
            (0, 255, 0) if active
            else (110, 110, 110)
        )

        draw_text(
            frame,
            f"{name.upper()}: {'UP' if active else 'DOWN'}",
            20, 235 + i * 28,
            color, 0.5
        )


def main():
    print("=" * 48)
    print("WHEELS AI - SLOW ROADS CONTROL")
    print("=" * 48)

    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not camera.isOpened():
        print("[ERROR] Camera unavailable.")
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    tracker = None
    controller = None

    classifier = GestureClassifier()
    speed_filter = GestureSmoother()
    steer_filter = GestureSmoother()

    previous_time = time.perf_counter()
    fps = 0.0

    try:
        tracker = HandTracker()
        controller = GameController()

        print("[WHEELS] Vision online.")
        print("[WHEELS] F8: Enable / disable controls")
        print("[WHEELS] ESC: Emergency stop")
        print("[WHEELS] Q: Exit when preview focused")

        while not controller.emergency:
            success, frame = camera.read()

            if not success:
                print("[ERROR] Camera frame lost.")
                break

            frame = cv2.flip(frame, 1)
            frame, result = tracker.process(frame)

            speed_data = None
            steering_data = None
            finger_states = None

            for i, landmarks in enumerate(
                result.hand_landmarks
            ):
                hand = result.handedness[i][0].category_name

                if hand == "Right":
                    speed_data = classifier.classify_left(
                        landmarks
                    )
                    finger_states = speed_data.get(
                        "finger_states"
                    )

                elif hand == "Left":
                    steering_data = classifier.classify_right(
                        landmarks
                    )

            throttle = 0.0
            steering = 0.0
            brake = False

            speed_status = "NOT DETECTED"
            steering_status = "NOT DETECTED"

            # RIGHT HAND: SPEED
            if speed_data is not None:
                level = speed_filter.update(
                    speed_data["fingers"]
                )

                if level is not None:
                    throttle = level / 5.0
                    brake = level == 0

                    speed_status = (
                        "BRAKE" if brake
                        else f"LEVEL {level}"
                    )
                else:
                    speed_status = "STABILIZING"
            else:
                speed_filter.clear()

            # LEFT HAND: STEERING
            if steering_data is not None:
                # Classifier original:
                # thumb = -1, pinky = +1
                # Reverse for WHEELS.
                value = -steering_data["steering"]
                stable = steer_filter.update(value)

                if stable is not None:
                    steering = stable

                    if steering < 0:
                        steering_status = "LEFT (PINKY)"
                    elif steering > 0:
                        steering_status = "RIGHT (THUMB)"
                    else:
                        steering_status = "STRAIGHT"
                else:
                    steering_status = "STABILIZING"
            else:
                steer_filter.clear()

            # Only drive when BOTH hands are visible
            # and both gestures have stabilized.
            ready = (
                speed_data is not None
                and steering_data is not None
                and speed_filter.stable_value is not None
                and steer_filter.stable_value is not None
            )

            if ready:
                controller.update(
                    throttle=throttle,
                    steering=steering,
                    brake=brake
                )
            else:
                controller.release_all()

            # FPS
            now = time.perf_counter()
            dt = max(now - previous_time, 0.001)
            previous_time = now

            instant_fps = 1.0 / dt
            fps = (
                instant_fps if fps == 0
                else 0.85 * fps + 0.15 * instant_fps
            )

            height, width = frame.shape[:2]

            # Dashboard
            overlay = frame.copy()

            cv2.rectangle(
                overlay, (0, 0),
                (width, 205),
                (20, 20, 20), -1
            )

            cv2.rectangle(
                overlay, (0, 205),
                (285, 395),
                (20, 20, 20), -1
            )

            frame = cv2.addWeighted(
                overlay, 0.65, frame, 0.35, 0
            )

            draw_text(
                frame,
                "WHEELS AI - SLOW ROADS",
                20, 35, (0, 255, 0), 0.85
            )

            draw_text(
                frame,
                f"RIGHT SPEED: {speed_status}",
                20, 75, (0, 255, 255)
            )

            draw_text(
                frame,
                f"THROTTLE: {int(throttle * 100)}%",
                20, 110
            )

            draw_text(
                frame,
                f"LEFT STEERING: {steering_status}",
                20, 145, (255, 255, 0)
            )

            mode = (
                "ACTIVE"
                if controller.enabled and ready
                else "WAITING FOR HANDS"
                if controller.enabled
                else "DISABLED"
            )

            draw_text(
                frame,
                f"GAME CONTROL: {mode}",
                20, 180,
                (0, 255, 0) if mode == "ACTIVE"
                else (0, 165, 255)
            )

            draw_text(
                frame,
                f"FPS: {fps:.1f}",
                max(20, width - 160), 35
            )

            draw_fingers(frame, finger_states)

            # Throttle bar
            bar_x, bar_y = 20, 395
            bar_width = 250

            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (bar_x + bar_width, bar_y + 15),
                (100, 100, 100), 2
            )

            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (
                    bar_x + int(bar_width * throttle),
                    bar_y + 15
                ),
                (0, 255, 0), -1
            )

            # Steering indicator
            cx = width // 2
            sy = height - 75

            cv2.line(
                frame,
                (cx - 110, sy),
                (cx + 110, sy),
                (255, 255, 255), 2
            )

            cv2.circle(
                frame,
                (int(cx + steering * 110), sy),
                12, (0, 255, 0), -1
            )

            draw_text(
                frame,
                "F8: TOGGLE | ESC: STOP | Q: QUIT",
                20, height - 20,
                (255, 255, 255), 0.55
            )

            cv2.imshow(
                "WHEELS - Driving Dashboard",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("[WHEELS] Interrupted.")

    finally:
        print("[WHEELS] Releasing controls...")

        if controller is not None:
            controller.close()

        camera.release()

        if tracker is not None:
            tracker.close()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
