from typing import Annotated, cast

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response

from face_detector.config import settings
from face_detector.detection import DetectionFailure, FaceDetector, detect_face

router = APIRouter()


def error_response(message: str, status_code: int) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status_code)


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


def get_face_detector(request: Request) -> FaceDetector:
    return cast(FaceDetector, request.app.state.face_detector)


type FaceDetectorDependency = Annotated[FaceDetector, Depends(get_face_detector)]


@router.post(
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
    detector: FaceDetectorDependency,
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
) -> Response:
    try:
        score_threshold = optional_float(score_threshold_parameter, "scoreThreshold")
        score_threshold = (
            settings.model_min_confidence
            if score_threshold is None
            else score_threshold
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
        if any(value not in (None, "") for value in crop_values):
            parsed_start_x = optional_int(startX, "startX")
            parsed_start_y = optional_int(startY, "startY")
            parsed_end_x = optional_int(endX, "endX")
            parsed_end_y = optional_int(endY, "endY")
            start_x = max(
                0, min(0 if parsed_start_x is None else parsed_start_x, image_width)
            )
            start_y = max(
                0, min(0 if parsed_start_y is None else parsed_start_y, image_height)
            )
            end_x = max(
                start_x,
                min(image_width if parsed_end_x is None else parsed_end_x, image_width),
            )
            end_y = max(
                start_y,
                min(
                    image_height if parsed_end_y is None else parsed_end_y, image_height
                ),
            )
            if start_x == end_x or start_y == end_y:
                return error_response("切り抜き範囲が空です。", 400)
            image = image[start_y:end_y, start_x:end_x]

        result = detect_face(
            image,
            detector,
            score_threshold,
            min_size,
        )
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


@router.get("/detect", response_class=FileResponse, include_in_schema=False)
def detect_form() -> FileResponse:
    return FileResponse(settings.static_dir / "index.html", media_type="text/html")


@router.get("/", response_class=RedirectResponse, include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse("/detect")


@router.get("/icon.png", response_class=FileResponse, include_in_schema=False)
def icon() -> FileResponse:
    return FileResponse(settings.static_dir / "icon.png", media_type="image/png")


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
