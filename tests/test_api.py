from io import BytesIO
from typing import cast

import cv2
import numpy as np
import pytest
from fastapi import UploadFile
from fastapi.responses import JSONResponse, Response
from starlette.routing import Mount

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
    return bytes(response.body).decode()


def test_health() -> None:
    assert api.health() == {"status": "ok"}


def test_root_redirects_to_ui() -> None:
    response = api.root()

    assert response.status_code == 307
    assert response.headers["location"] == "/detect"


def test_icon_response_uses_static_icon() -> None:
    response = api.icon()

    assert response.media_type == "image/png"
    assert response.path == api.settings.static_dir / "icon.png"


def test_runtime_openapi_is_disabled() -> None:
    app = create_app()

    assert app.openapi_url is None
    assert app.docs_url is None
    assert app.redoc_url is None


def test_openapi_contains_public_endpoints() -> None:
    app = create_app()
    schema = app.openapi()

    assert set(schema["paths"]) == {"/detect", "/health"}
    assert "post" in schema["paths"]["/detect"]
    assert any(
        isinstance(route, Mount) and route.path == "/static" for route in app.routes
    )


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
        ({"padding": "-1"}, "paddingは0以上で指定してください。"),
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


def test_detect_passes_padding_to_face_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_padding: int | None = None

    def detect_with_padding(
        _image: np.ndarray,
        _detector: FaceDetector,
        _score_threshold: float,
        _min_size: int,
        padding: int,
    ) -> DetectionSuccess:
        nonlocal seen_padding
        seen_padding = padding
        return DetectionSuccess(jpeg_bytes(8, 6), (8, 6))

    monkeypatch.setattr(api, "detect_face", detect_with_padding)
    response = api.detect(
        file=upload_file(jpeg_bytes()),
        detector=detector_stub(),
        padding="5",
    )

    assert isinstance(response, Response)
    assert response.status_code == 200
    assert seen_padding == 5


def test_detect_applies_partial_crop_coordinates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_shape: tuple[int, ...] | None = None

    def detect_cropped_image(image: np.ndarray, *_: object) -> DetectionSuccess:
        nonlocal seen_shape
        seen_shape = image.shape
        return DetectionSuccess(
            jpeg_bytes(image.shape[1], image.shape[0]), (image.shape[1], image.shape[0])
        )

    monkeypatch.setattr(api, "detect_face", detect_cropped_image)

    response = api.detect(
        file=upload_file(jpeg_bytes(width=960, height=1280)),
        detector=detector_stub(),
        startY="300",
    )

    assert isinstance(response, Response)
    assert response.status_code == 200
    assert seen_shape == (980, 960, 3)


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
