"""Export the OpenAPI schema generated from the FastAPI application."""

import json
import sys
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "docs" / "openapi.json"
sys.path.insert(0, str(PROJECT_ROOT))

from face_detector.main import app

with (PROJECT_ROOT / "pyproject.toml").open("rb") as project_file:
    PROJECT_VERSION: str = tomllib.load(project_file)["project"]["version"]


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    schema = app.openapi()
    schema["info"]["version"] = PROJECT_VERSION
    OUTPUT_PATH.write_text(
        json.dumps(schema, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"OpenAPI schema exported to {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
