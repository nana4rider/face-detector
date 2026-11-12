from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from importlib.metadata import version

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from face_detector.api import error_response, router
from face_detector.config import settings
from face_detector.detection import FaceDetector

APP_VERSION = version("face-detector")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    app.state.face_detector = FaceDetector(
        settings.model_path,
        settings.model_min_confidence,
    )
    yield
    app.state.face_detector.close()


def create_app() -> FastAPI:
    application = FastAPI(
        title="Face Detector",
        description="画像から最も信頼度の高い顔を検出して切り抜くAPIです。",
        version=APP_VERSION,
        lifespan=lifespan,
        openapi_url=None,
        docs_url=None,
        redoc_url=None,
    )

    @application.exception_handler(RequestValidationError)
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

    application.include_router(router)
    application.mount(
        "/static",
        StaticFiles(directory=settings.static_dir),
        name="static",
    )
    return application


app = create_app()
