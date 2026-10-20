
from pathlib import Path
import time

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandTracker:
    def __init__(self):
        model_path = (
            Path(__file__).resolve().parent.parent
            / "models"
            / "hand_landmarker.task"
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe model not found: {model_path}"
            )

        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(model_path)
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.5,
        )

        self.landmarker = (
            vision.HandLandmarker.create_from_options(options)
        )

        self.start_time = time.monotonic()
        self.last_timestamp_ms = -1

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp_ms = max(
            self.last_timestamp_ms + 1,
            int((time.monotonic() - self.start_time) * 1000)
        )
        self.last_timestamp_ms = timestamp_ms

        result = self.landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        height, width = frame.shape[:2]

        connections = (
            mp.tasks.vision
            .HandLandmarksConnections
            .HAND_CONNECTIONS
        )

        for hand_index, landmarks in enumerate(
            result.hand_landmarks
        ):
            # Draw connections
            for connection in connections:
                start = connection.start
                end = connection.end

                p1 = landmarks[start]
                p2 = landmarks[end]

                cv2.line(
                    frame,
                    (int(p1.x * width), int(p1.y * height)),
                    (int(p2.x * width), int(p2.y * height)),
                    (0, 255, 0),
                    2
                )

            # Draw all 21 landmarks
            for landmark_id, landmark in enumerate(landmarks):
                x = int(landmark.x * width)
                y = int(landmark.y * height)

                cv2.circle(
                    frame,
                    (x, y),
                    4,
                    (0, 0, 255),
                    -1
                )

            # Display handedness
            handedness = result.handedness[hand_index][0]
            hand_label = handedness.category_name
            confidence = handedness.score

            wrist = landmarks[0]

            text_x = max(10, int(wrist.x * width))
            text_y = max(30, int(wrist.y * height) - 20)

            cv2.putText(
                frame,
                f"{hand_label} ({confidence:.2f})",
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
                cv2.LINE_AA
            )

        return frame, result

    def close(self):
        self.landmarker.close()
