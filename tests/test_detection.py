from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np

from face_detector.detection import (
    DetectionFailure,
    DetectionSuccess,
    FaceDetector,
    detect_face,
)


@dataclass(frozen=True)
class BoundingBox:
    origin_x: int
    origin_y: int
    width: int
    height: int


@dataclass(frozen=True)
class Category:
    score: float


@dataclass(frozen=True)
class Detection:
    bounding_box: BoundingBox
    categories: list[Category]


@dataclass(frozen=True)
class DetectionResult:
    detections: list[Detection]


class FakeFaceDetector:
    def __init__(self, detections: list[Detection]) -> None:
        self._result = DetectionResult(detections)

    def process(self, _: np.ndarray) -> DetectionResult:
        return self._result


def detector_with(*detections: Detection) -> FaceDetector:
    return cast(FaceDetector, FakeFaceDetector(list(detections)))


def detection(score: float, x: int, y: int, width: int, height: int) -> Detection:
    return Detection(BoundingBox(x, y, width, height), [Category(score)])


def test_no_face_returns_failure() -> None:
    image = np.zeros((40, 40, 3), dtype=np.uint8)

    result = detect_face(image, detector_with(), 0.5, 0)

    assert result == DetectionFailure("顔が検出されませんでした。", 404)


def test_filters_by_score_and_minimum_size() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    detector = detector_with(
        detection(0.4, 0, 0, 50, 50),
        detection(0.9, 0, 0, 10, 10),
    )

    result = detect_face(image, detector, score_threshold=0.5, min_size=20)

    assert result == DetectionFailure(
        "指定された条件を満たす顔が検出されませんでした。", 404
    )


def test_selects_highest_score_face() -> None:
    image = np.zeros((60, 80, 3), dtype=np.uint8)
    detector = detector_with(
        detection(0.7, 0, 0, 20, 30),
        detection(0.9, 30, 10, 40, 25),
    )

    result = detect_face(image, detector, score_threshold=0.5, min_size=0)

    assert isinstance(result, DetectionSuccess)
    assert result.dimensions == (40, 25)
    decoded = cv2.imdecode(
        np.frombuffer(result.image_bytes, np.uint8), cv2.IMREAD_COLOR
    )
    assert decoded is not None
    assert decoded.shape[:2] == (25, 40)


def test_adds_padding_around_detected_face() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    detector = detector_with(detection(0.9, 20, 30, 40, 25))

    result = detect_face(image, detector, score_threshold=0.5, min_size=0, padding=5)

    assert isinstance(result, DetectionSuccess)
    assert result.dimensions == (50, 35)
    decoded = cv2.imdecode(
        np.frombuffer(result.image_bytes, np.uint8), cv2.IMREAD_COLOR
    )
    assert decoded is not None
    assert decoded.shape[:2] == (35, 50)


def test_clamps_bounding_box_to_image() -> None:
    image = np.zeros((50, 60, 3), dtype=np.uint8)
    detector = detector_with(detection(0.9, -10, -5, 100, 80))

    result = detect_face(image, detector, score_threshold=0.5, min_size=0)

    assert isinstance(result, DetectionSuccess)
    assert result.dimensions == (60, 50)


def test_clamps_padding_to_image_edges() -> None:
    image = np.zeros((50, 60, 3), dtype=np.uint8)
    detector = detector_with(detection(0.9, 0, 0, 20, 20))

    result = detect_face(image, detector, score_threshold=0.5, min_size=0, padding=10)

    assert isinstance(result, DetectionSuccess)
    assert result.dimensions == (30, 30)


def test_padding_can_extend_outside_detection_crop() -> None:
    source_image = np.zeros((100, 120, 3), dtype=np.uint8)
    detection_image = source_image[10:90, 40:100]
    detector = detector_with(detection(0.9, 5, 20, 30, 30))

    result = detect_face(
        detection_image,
        detector,
        score_threshold=0.5,
        min_size=0,
        padding=20,
        source_image=source_image,
        source_origin=(40, 10),
    )

    assert isinstance(result, DetectionSuccess)
    assert result.dimensions == (70, 70)
    decoded = cv2.imdecode(
        np.frombuffer(result.image_bytes, np.uint8), cv2.IMREAD_COLOR
    )
    assert decoded is not None
    assert decoded.shape[:2] == (70, 70)
