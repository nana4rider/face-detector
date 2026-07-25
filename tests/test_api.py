from io import BytesIO
from typing import cast

import cv2
import numpy as np
import pytest
from fastapi import UploadFile
from fastapi.responses import JSONResponse, Response

from face_detector import api
from face_detector.detection import DetectionFailure, DetectionSuccess, FaceDetector
from face_detector.main import create_app


def jpeg_bytes(width: int = 20, height: int = 10) -> bytes:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    encoded, buffer = cv2.imencode(".jpg", image)
    assert encoded
    return buffer.tobytes()


def upload_file(content: bytes) -> UploadFile:
    return UploadFile(BytesIO(content), filename="image.jpg")


def detector_stub() -> FaceDetector:
    return cast(FaceDetector, object())


def response_json(response: JSONResponse) -> str:
    return response.body.decode()


def test_health() -> None:
    assert api.health() == {"status": "ok"}


def test_runtime_openapi_is_disabled() -> None:
    app = create_app()

    assert app.openapi_url is None
    assert app.docs_url is None
    assert app.redoc_url is None


def test_openapi_contains_public_endpoints() -> None:
    schema = create_app().openapi()

    assert set(schema["paths"]) == {"/detect", "/health"}
    assert "post" in schema["paths"]["/detect"]


def test_detect_rejects_invalid_image() -> None:
    response = api.detect(
        file=upload_file(b"not an image"),
        detector=detector_stub(),
    )

    assert response.status_code == 400
    assert isinstance(response, JSONResponse)
    assert "入力画像を読み込めませんでした。" in response_json(response)


@pytest.mark.parametrize(
    ("form_data", "message"),
    [
        (
            {"score_threshold_parameter": "1.1"},
            "scoreThresholdは0から1の範囲で指定してください。",
        ),
        ({"minSize": "-1"}, "minSizeは0以上で指定してください。"),
    ],
)
def test_detect_rejects_invalid_parameters(
    form_data: dict[str, str],
    message: str,
) -> None:
    response = api.detect(
        file=upload_file(jpeg_bytes()),
        detector=detector_stub(),
        **form_data,
    )

    assert response.status_code == 400
    assert isinstance(response, JSONResponse)
    assert message in response_json(response)


def test_detect_returns_jpeg_and_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api,
        "detect_face",
        lambda *_: DetectionSuccess(jpeg_bytes(8, 6), (8, 6)),
    )

    response = api.detect(
        file=upload_file(jpeg_bytes()),
        detector=detector_stub(),
    )

    assert isinstance(response, Response)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["x-image-width"] == "8"
    assert response.headers["x-image-height"] == "6"


def test_detect_returns_detection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api,
        "detect_face",
        lambda *_: DetectionFailure("顔が検出されませんでした。", 404),
    )

    response = api.detect(
        file=upload_file(jpeg_bytes()),
        detector=detector_stub(),
    )

    assert response.status_code == 404
    assert isinstance(response, JSONResponse)
    assert "顔が検出されませんでした。" in response_json(response)
