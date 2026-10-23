
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class BackgroundRemover:
    def __init__(self):
        model_path = (
            Path(__file__).resolve().parent.parent
            / "models"
            / "selfie_segmenter.tflite"
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Segmentation model missing: {model_path}"
            )

        options = vision.ImageSegmenterOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(model_path)
            ),
            running_mode=vision.RunningMode.IMAGE,
            output_confidence_masks=True,
            output_category_mask=False,
        )

        self.segmenter = (
            vision.ImageSegmenter.create_from_options(options)
        )

        self.background_color = (0, 190, 0)
        self.previous_mask = None

    def _hand_mask(self, shape, hand_landmarks):
        height, width = shape[:2]

        mask = np.zeros(
            (height, width), dtype=np.uint8
        )

        for hand in hand_landmarks:
            points = np.array(
                [
                    [
                        int(np.clip(p.x, 0, 1) * (width - 1)),
                        int(np.clip(p.y, 0, 1) * (height - 1)),
                    ]
                    for p in hand
                ],
                dtype=np.int32,
            )

            # Fill the palm and surrounding finger area.
            hull = cv2.convexHull(points)
            cv2.fillConvexPoly(mask, hull, 255)

            # Protect individual fingers.
            for point in points:
                cv2.circle(
                    mask,
                    tuple(point),
                    15,
                    255,
                    -1,
                )

            # Protect the entire hand silhouette.
            for connection in (
                mp.tasks.vision
                .HandLandmarksConnections
                .HAND_CONNECTIONS
            ):
                start = tuple(points[connection.start])
                end = tuple(points[connection.end])

                cv2.line(
                    mask,
                    start,
                    end,
                    26,
                    255,
                )

        mask = cv2.GaussianBlur(
            mask, (9, 9), 0
        )

        return mask.astype(np.float32) / 255.0

    def apply(self, frame, hand_landmarks=None):
        if hand_landmarks is None:
            hand_landmarks = []

        height, width = frame.shape[:2]

        rgb = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb),
        )

        result = self.segmenter.segment(mp_image)

        if (
            result.confidence_masks is None
            or len(result.confidence_masks) < 2
        ):
            # Fail closed rather than showing the room.
            return np.full_like(
                frame, self.background_color
            )

        person_mask = np.array(
            result.confidence_masks[1].numpy_view(),
            dtype=np.float32,
        )

        if person_mask.shape != (height, width):
            person_mask = cv2.resize(
                person_mask,
                (width, height),
            )

        # Less aggressive threshold for body visibility.
        person_alpha = np.clip(
            (person_mask - 0.18) / 0.45,
            0.0,
            1.0,
        )

        # Protect detected hands independently.
        hand_alpha = self._hand_mask(
            frame.shape,
            hand_landmarks,
        )

        # Combine person and hand masks.
        alpha = np.maximum(
            person_alpha,
            hand_alpha,
        )

        alpha = cv2.GaussianBlur(
            alpha, (5, 5), 0
        )

        # Smooth segmentation across frames.
        if self.previous_mask is not None:
            alpha = (
                0.8 * alpha
                + 0.2 * self.previous_mask
            )

        self.previous_mask = alpha.copy()

        alpha = np.clip(
            alpha, 0.0, 1.0
        )[:, :, None]

        green = np.full_like(
            frame,
            self.background_color,
        )

        output = (
            frame.astype(np.float32) * alpha
            + green.astype(np.float32) * (1 - alpha)
        )

        return np.clip(
            output, 0, 255
        ).astype(np.uint8)

    def close(self):
        self.segmenter.close()
