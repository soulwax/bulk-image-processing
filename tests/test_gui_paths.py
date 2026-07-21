from pathlib import Path

import gui


def test_ensure_directory_creates_missing_folders(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "output"

    created = gui.ensure_directory(target)

    assert created == target
    assert target.exists()
    assert target.is_dir()
