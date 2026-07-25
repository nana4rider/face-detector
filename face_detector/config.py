import os
from dataclasses import dataclass
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Settings:
    static_dir: Path = PACKAGE_ROOT / "static"
    model_path: Path = Path(
        os.getenv(
            "FACE_DETECTION_MODEL",
            str(PACKAGE_ROOT / "models" / "blaze_face_short_range.tflite"),
        )
    )
    model_min_confidence: float = float(os.getenv("FACE_DETECTION_CONFIDENCE", "0.5"))


settings = Settings()
