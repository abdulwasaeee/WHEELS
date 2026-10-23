
import time
from collections import Counter, deque

import cv2

from vision.hand_tracker import HandTracker
from vision.gesture_classifier import GestureClassifier
from vision.steering_tracker import SteeringTracker
from vision.orientation_detector import (
    OrientationDetector,
    HoldGesture,
)
from vision.background_remover import BackgroundRemover
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


def draw_text(
    frame, text, x, y,
    color=(255, 255, 255), scale=0.6
):
    cv2.putText(
        frame, str(text), (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale, color, 2, cv2.LINE_AA
    )


def draw_bar(frame, x, y, width, value, color):
    value = max(0.0, min(1.0, value))

    cv2.rectangle(
        frame,
        (x, y),
        (x + width, y + 15),
        (120, 120, 120), 2
    )

    if value > 0:
        cv2.rectangle(
            frame,
            (x, y),
            (x + int(width * value), y + 15),
            color, -1
        )


def draw_fingers(frame, states):
    names = [
        "thumb", "index", "middle",
        "ring", "pinky"
    ]

    for i, name in enumerate(names):
        active = bool(
            states and states.get(name, False)
        )

        color = (
            (0, 255, 0)
            if active else (120, 120, 120)
        )

        draw_text(
            frame,
            f"{name.upper()}: "
            f"{'UP' if active else 'DOWN'}",
            20, 330 + i * 27,
            color, 0.5
        )


def draw_steering(frame, steering):
    height, width = frame.shape[:2]

    cx = width // 2
    y = height - 85
    radius = 140

    cv2.line(
        frame,
        (cx - radius, y),
        (cx + radius, y),
        (255, 255, 255), 2
    )

    cv2.line(
        frame,
        (cx, y - 15),
        (cx, y + 15),
        (0, 255, 0), 2
    )

    marker_x = int(cx + steering * radius)

    cv2.circle(
        frame,
        (marker_x, y),
        12,
        (0, 255, 255), -1
    )


def main():
    print("=" * 50)
    print("WHEELS AI - HAND PRESERVING GREEN SCREEN")
    print("=" * 50)

    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not camera.isOpened():
        print("[ERROR] Webcam unavailable.")
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    tracker = None
    controller = None
    background_remover = None

    classifier = GestureClassifier()
    orientation = OrientationDetector()

    steering_tracker = SteeringTracker(
        dead_zone=0.035,
        max_offset=0.22,
        smoothing=0.22
    )

    speed_smoother = GestureSmoother()

    autodrive_hold = HoldGesture(
        hold_seconds=1.2,
        cooldown_seconds=2.0,
        release_seconds=0.35
    )

    reverse_hold = HoldGesture(
        hold_seconds=0.35,
        cooldown_seconds=0.0,
        release_seconds=0.15
    )

    reverse_active = False
    previous_time = time.perf_counter()
    fps = 0.0

    try:
        tracker = HandTracker()
        background_remover = BackgroundRemover()
        controller = GameController()

        print("[WHEELS] Vision online.")
        print("[WHEELS] Hands protected from green screen.")
        print("[WHEELS] Right hand: Forward / Reverse")
        print("[WHEELS] Left hand: Steering")
        print("[WHEELS] Both hands down: Autodrive")
        print("[WHEELS] C: Calibrate steering")
        print("[WHEELS] F8: Enable / Disable")
        print("[WHEELS] ESC: Emergency stop")
        print("[WHEELS] Q: Quit")

        while not controller.emergency:
            success, frame = camera.read()

            if not success:
                print("[ERROR] Camera frame unavailable.")
                break

            frame = cv2.flip(frame, 1)

            # Keep an unannotated image for segmentation.
            raw_frame = frame.copy()

            # MediaPipe processes the original camera image.
            annotated_frame, result = tracker.process(frame)

            # Preserve hands while replacing the room.
            frame = background_remover.apply(
                raw_frame,
                result.hand_landmarks
            )

            # Draw landmarks AFTER segmentation.
            height, width = frame.shape[:2]

            connections = (
                cv2  # Placeholder avoided below
            )

            import mediapipe as mp

            hand_connections = (
                mp.tasks.vision
                .HandLandmarksConnections
                .HAND_CONNECTIONS
            )

            for landmarks in result.hand_landmarks:
                for connection in hand_connections:
                    a = landmarks[connection.start]
                    b = landmarks[connection.end]

                    cv2.line(
                        frame,
                        (int(a.x * width), int(a.y * height)),
                        (int(b.x * width), int(b.y * height)),
                        (0, 255, 255),
                        2
                    )

                for landmark in landmarks:
                    cv2.circle(
                        frame,
                        (
                            int(landmark.x * width),
                            int(landmark.y * height)
                        ),
                        4,
                        (0, 0, 255),
                        -1
                    )

            speed_data = None
            steering_landmarks = None
            finger_states = None
            right_down = False
            left_down = False

            for i, landmarks in enumerate(
                result.hand_landmarks
            ):
                hand = (
                    result.handedness[i][0]
                    .category_name
                )

                orientation_data = (
                    orientation.classify(landmarks)
                )

                if hand == "Right":
                    speed_data = (
                        classifier.classify_left(
                            landmarks
                        )
                    )
                    finger_states = speed_data.get(
                        "finger_states"
                    )
                    right_down = orientation_data["down"]

                elif hand == "Left":
                    steering_landmarks = landmarks
                    left_down = orientation_data["down"]

            both_visible = (
                speed_data is not None
                and steering_landmarks is not None
            )

            both_down = (
                both_visible
                and right_down
                and left_down
            )

            autodrive_triggered = (
                autodrive_hold.update(
                    both_down and controller.enabled
                )
            )

            if autodrive_triggered:
                controller.toggle_autodrive()

            reverse_candidate = (
                controller.enabled
                and right_down
                and not left_down
                and not both_down
                and not controller.autodrive
            )

            reverse_hold.update(reverse_candidate)

            if not reverse_candidate:
                reverse_active = False

            elif (
                reverse_hold.started_at is not None
                and time.monotonic()
                - reverse_hold.started_at >= 0.35
            ):
                reverse_active = True

            # Right-hand speed
            throttle = 0.0
            handbrake = False
            speed_status = "NOT DETECTED"

            if speed_data is not None:
                if right_down:
                    speed_smoother.clear()

                    if both_down:
                        speed_status = "AUTODRIVE GESTURE"
                    elif reverse_active:
                        speed_status = "REVERSE"
                    else:
                        speed_status = "REVERSE HOLD"

                else:
                    level = speed_smoother.update(
                        speed_data["fingers"]
                    )

                    if level is not None:
                        throttle = level / 5.0
                        handbrake = level == 0

                        speed_status = (
                            "HANDBRAKE"
                            if handbrake
                            else f"FORWARD LEVEL {level}"
                        )
                    else:
                        speed_status = "STABILIZING"
            else:
                speed_smoother.clear()

            # Left-hand steering
            steering = 0.0
            steering_status = "NOT DETECTED"

            if steering_landmarks is not None:
                if both_down:
                    steering_tracker.reset_tracking()
                    steering_status = "AUTODRIVE GESTURE"
                else:
                    steering = steering_tracker.update(
                        steering_landmarks
                    )

                    if not steering_tracker.calibrated:
                        steering_status = "PRESS C"

                    elif steering < -0.02:
                        steering_status = (
                            f"LEFT {abs(steering)*100:.0f}%"
                        )

                    elif steering > 0.02:
                        steering_status = (
                            f"RIGHT {steering*100:.0f}%"
                        )

                    else:
                        steering_status = "STRAIGHT"
            else:
                steering_tracker.reset_tracking()

            ready = (
                both_visible
                and speed_smoother.stable_value
                is not None
                and steering_tracker.calibrated
            )

            if controller.autodrive or both_down:
                controller.release_all()

            elif reverse_candidate:
                if (
                    reverse_active
                    and steering_tracker.calibrated
                    and both_visible
                ):
                    controller.update(
                        throttle=0.0,
                        steering=steering,
                        handbrake=False,
                        reverse=True
                    )
                else:
                    controller.release_all()

            elif ready:
                controller.update(
                    throttle=throttle,
                    steering=steering,
                    handbrake=handbrake,
                    reverse=False
                )
            else:
                controller.release_all()

            now = time.perf_counter()
            dt = max(now - previous_time, 0.001)
            previous_time = now

            instant_fps = 1.0 / dt
            fps = (
                instant_fps if fps == 0
                else 0.85 * fps
                + 0.15 * instant_fps
            )

            height, width = frame.shape[:2]

            overlay = frame.copy()

            cv2.rectangle(
                overlay, (0, 0),
                (width, 300),
                (20, 20, 20), -1
            )

            cv2.rectangle(
                overlay, (0, 305),
                (285, 485),
                (20, 20, 20), -1
            )

            frame = cv2.addWeighted(
                overlay, 0.65,
                frame, 0.35, 0
            )

            draw_text(
                frame,
                "WHEELS AI - GREEN SCREEN",
                20, 35,
                (0, 255, 0), 0.85
            )

            draw_text(
                frame,
                f"RIGHT HAND: {speed_status}",
                20, 75,
                (0, 255, 255)
            )

            draw_text(
                frame,
                f"THROTTLE: {throttle*100:.0f}%",
                20, 110
            )

            draw_text(
                frame,
                f"LEFT HAND: {steering_status}",
                20, 145,
                (255, 255, 0)
            )

            mode = (
                "DISABLED"
                if not controller.enabled
                else "AUTODRIVE"
                if controller.autodrive
                else "GESTURE HOLD"
                if both_down
                else "REVERSE"
                if reverse_active
                else "MANUAL"
                if ready
                else "WAITING"
            )

            draw_text(
                frame,
                f"MODE: {mode}",
                20, 180,
                (0, 255, 0)
                if mode in ("MANUAL", "AUTODRIVE")
                else (0, 165, 255)
            )

            draw_text(
                frame,
                f"RIGHT DOWN: {right_down}",
                20, 215
            )

            draw_text(
                frame,
                f"LEFT DOWN: {left_down}",
                20, 245
            )

            draw_text(
                frame,
                f"FPS: {fps:.1f}",
                max(20, width - 155),
                35
            )

            progress = (
                autodrive_hold.progress()
                if both_down else 0.0
            )

            draw_text(
                frame,
                "AUTODRIVE HOLD",
                20, 280,
                (255, 255, 0), 0.5
            )

            draw_bar(
                frame, 185, 266,
                160, progress,
                (0, 255, 255)
            )

            draw_fingers(frame, finger_states)

            draw_bar(
                frame, 20, 505, 250,
                throttle, (0, 255, 0)
            )

            draw_steering(frame, steering)

            draw_text(
                frame,
                "C: CALIBRATE | F8: TOGGLE | ESC: STOP",
                20, height - 20,
                (255, 255, 255), 0.5
            )

            cv2.imshow(
                "WHEELS - Hand-Preserving Green Screen",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("c"):
                if steering_landmarks is not None:
                    steering_tracker.update(
                        steering_landmarks
                    )
                steering_tracker.calibrate()

            elif key == ord("q"):
                break

    except KeyboardInterrupt:
        print("[WHEELS] Interrupted.")

    finally:
        print("[WHEELS] Shutting down...")

        if controller is not None:
            controller.close()

        camera.release()

        if tracker is not None:
            tracker.close()

        if background_remover is not None:
            background_remover.close()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
