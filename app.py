import os
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from mediapipe.tasks.python.components.containers.detections import (
    DetectionResult as MediaPipeDetectionResult,
)

STATIC_DIR = Path(__file__).parent / "static"
DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "blaze_face_short_range.tflite"
MODEL_PATH = Path(os.getenv("FACE_DETECTION_MODEL", str(DEFAULT_MODEL_PATH)))
MODEL_MIN_CONFIDENCE = float(os.getenv("FACE_DETECTION_CONFIDENCE", "0.5"))


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.face_detector = FaceDetector(MODEL_PATH, MODEL_MIN_CONFIDENCE)
    yield
    app.state.face_detector.close()


app = FastAPI(
    title="Face Detector",
    description="画像から最も信頼度の高い顔を検出して切り抜くAPIです。",
    version="3.0.0",
    lifespan=lifespan,
    openapi_url=None,
    docs_url=None,
    redoc_url=None,
)


def error_response(message: str, status_code: int) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status_code)


@app.exception_handler(RequestValidationError)
def validation_error_handler(_, exc: RequestValidationError) -> JSONResponse:
    missing_file = any(
        error["type"] == "missing" and error["loc"][-1] == "file"
        for error in exc.errors()
    )
    message = (
        "ファイルがアップロードされていません。"
        if missing_file
        else "リクエストパラメータが不正です。"
    )
    return error_response(message, 400)


def optional_int(value: str | None, name: str) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name}は整数で指定してください。") from exc


def optional_float(value: str | None, name: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name}は数値で指定してください。") from exc


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


@app.post(
    "/detect",
    response_class=Response,
    responses={
        200: {
            "content": {"image/jpeg": {}},
            "description": "検出した顔のJPEG画像",
            "headers": {
                "X-Image-Width": {"schema": {"type": "integer"}},
                "X-Image-Height": {"schema": {"type": "integer"}},
            },
        },
        400: {"description": "入力エラー"},
        404: {"description": "条件を満たす顔がない"},
        500: {"description": "画像処理エラー"},
    },
)
def detect(
    file: Annotated[UploadFile, File(description="処理する画像")],
    score_threshold_parameter: Annotated[
        str | None,
        Form(
            alias="scoreThreshold",
            description="検出結果のscoreに対する下限値（0.0～1.0）",
        ),
    ] = None,
    minSize: Annotated[str | None, Form()] = None,
    startX: Annotated[str | None, Form()] = None,
    startY: Annotated[str | None, Form()] = None,
    endX: Annotated[str | None, Form()] = None,
    endY: Annotated[str | None, Form()] = None,
):
    try:
        score_threshold = optional_float(score_threshold_parameter, "scoreThreshold")
        score_threshold = (
            MODEL_MIN_CONFIDENCE if score_threshold is None else score_threshold
        )
        if not 0 <= score_threshold <= 1:
            return error_response(
                "scoreThresholdは0から1の範囲で指定してください。", 400
            )

        min_size = optional_int(minSize, "minSize") or 0
        if min_size < 0:
            return error_response("minSizeは0以上で指定してください。", 400)

        image_data = file.file.read()
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return error_response("入力画像を読み込めませんでした。", 400)

        image_height, image_width = image.shape[:2]
        crop_values = (startX, startY, endX, endY)
        if all(value not in (None, "") for value in crop_values):
            start_x = max(0, min(optional_int(startX, "startX") or 0, image_width))
            start_y = max(0, min(optional_int(startY, "startY") or 0, image_height))
            end_x = max(
                start_x, min(optional_int(endX, "endX") or image_width, image_width)
            )
            end_y = max(
                start_y,
                min(optional_int(endY, "endY") or image_height, image_height),
            )
            if start_x == end_x or start_y == end_y:
                return error_response("切り抜き範囲が空です。", 400)
            image = image[start_y:end_y, start_x:end_x]

        result = detect_face(image, app.state.face_detector, score_threshold, min_size)
        if isinstance(result, DetectionFailure):
            return error_response(result.message, result.status_code)

        return Response(
            content=result.image_bytes,
            media_type="image/jpeg",
            headers={
                "X-Image-Width": str(result.dimensions[0]),
                "X-Image-Height": str(result.dimensions[1]),
            },
        )
    except ValueError as exc:
        return error_response(str(exc), 400)
    except Exception as exc:  # noqa: BLE001
        return error_response(str(exc), 500)


@app.get("/detect", response_class=FileResponse, include_in_schema=False)
def detect_form() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
