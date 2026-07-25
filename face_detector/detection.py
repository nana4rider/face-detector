import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.components.containers.detections import (
    DetectionResult as MediaPipeDetectionResult,
)


@dataclass(frozen=True)
class DetectionSuccess:
    image_bytes: bytes
    dimensions: tuple[int, int]


@dataclass(frozen=True)
class DetectionFailure:
    message: str
    status_code: int


type FaceDetectionOutcome = DetectionSuccess | DetectionFailure


class FaceDetector:
    """Owns the MediaPipe Task and serializes access to its mutable state."""

    def __init__(self, model_path: Path, min_confidence: float) -> None:
        options = mp.tasks.vision.FaceDetectorOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(model_path),
                delegate=mp.tasks.BaseOptions.Delegate.CPU,
            ),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            min_detection_confidence=min_confidence,
        )
        self._detector = mp.tasks.vision.FaceDetector.create_from_options(options)
        self._lock = threading.Lock()

    def process(self, image: np.ndarray) -> MediaPipeDetectionResult:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        media_pipe_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb_image),
        )
        with self._lock:
            return self._detector.detect(media_pipe_image)

    def close(self) -> None:
        self._detector.close()


def detect_face(
    image: np.ndarray,
    detector: FaceDetector,
    score_threshold: float,
    min_size: int,
) -> FaceDetectionOutcome:
    results = detector.process(image)
    if not results.detections:
        return DetectionFailure("顔が検出されませんでした。", 404)

    image_height, image_width = image.shape[:2]
    faces: list[tuple[float, int, int, int, int]] = []
    for detection in results.detections:
        if not detection.categories:
            continue
        score = float(detection.categories[0].score)
        bbox = detection.bounding_box
        x1 = max(0, bbox.origin_x)
        y1 = max(0, bbox.origin_y)
        x2 = min(image_width, bbox.origin_x + bbox.width)
        y2 = min(image_height, bbox.origin_y + bbox.height)
        width, height = x2 - x1, y2 - y1
        if score >= score_threshold and width >= min_size and height >= min_size:
            faces.append((score, x1, y1, width, height))

    if not faces:
        return DetectionFailure("指定された条件を満たす顔が検出されませんでした。", 404)

    _, x, y, width, height = max(faces, key=lambda face: face[0])
    face_image = image[y : y + height, x : x + width]
    encoded, buffer = cv2.imencode(".jpg", face_image)
    if not encoded:
        return DetectionFailure("検出画像をJPEGに変換できませんでした。", 500)
    return DetectionSuccess(buffer.tobytes(), (width, height))
