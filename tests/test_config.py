from face_detector.config import settings


def test_default_settings_are_valid() -> None:
    assert settings.model_path.is_file()
    assert settings.static_dir.joinpath("index.html").is_file()
    assert settings.static_dir.joinpath("icon.png").is_file()
    assert 0 <= settings.model_min_confidence <= 1
