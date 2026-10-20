
import time

import cv2

from vision.hand_tracker import HandTracker


def main():
    print("=" * 45)
    print("       WHEELS - AI DRIVING COPILOT")
    print("=" * 45)
    print("[WHEELS] Initializing webcam...")

    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not camera.isOpened():
        print("[ERROR] Could not access webcam.")
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    tracker = None

    previous_time = time.perf_counter()
    smoothed_fps = 0.0

    try:
        tracker = HandTracker()

        print("[WHEELS] Hand tracking initialized.")
        print("[WHEELS] Show both hands to the camera.")
        print("[WHEELS] Press Q to quit.")

        while True:
            success, frame = camera.read()

            if not success:
                print("[ERROR] Unable to read camera frame.")
                break

            # Mirror camera for natural interaction
            frame = cv2.flip(frame, 1)

            # Run MediaPipe hand tracking
            frame, result = tracker.process(frame)

            hand_count = len(result.hand_landmarks)

            # Calculate FPS
            current_time = time.perf_counter()
            delta = current_time - previous_time
            previous_time = current_time

            instant_fps = 1.0 / max(delta, 0.001)

            if smoothed_fps == 0:
                smoothed_fps = instant_fps
            else:
                smoothed_fps = (
                    0.85 * smoothed_fps
                    + 0.15 * instant_fps
                )

            height, width = frame.shape[:2]

            # Dashboard background
            overlay = frame.copy()

            cv2.rectangle(
                overlay,
                (0, 0),
                (width, 115),
                (20, 20, 20),
                -1
            )

            frame = cv2.addWeighted(
                overlay, 0.65, frame, 0.35, 0
            )

            # Dashboard title
            cv2.putText(
                frame,
                "WHEELS AI - VISION SYSTEM",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                frame,
                f"Hands detected: {hand_count}/2",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"FPS: {smoothed_fps:.1f}",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            # Bottom status
            cv2.rectangle(
                frame,
                (0, height - 45),
                (width, height),
                (20, 20, 20),
                -1
            )

            cv2.putText(
                frame,
                "MODE: HAND TRACKING | PRESS Q TO EXIT",
                (20, height - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            cv2.imshow(
                "WHEELS - AI Driving Copilot",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("[WHEELS] Interrupted by user.")

    finally:
        print("[WHEELS] Shutting down...")

        camera.release()

        if tracker is not None:
            tracker.close()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
