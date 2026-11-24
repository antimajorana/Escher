"""Check the license and source information shipped with the optional native extension."""

from importlib.metadata import PackageNotFoundError, distribution
import os
from pathlib import Path

import pytest


def test_native_wheel_includes_project_license():
    try:
        native = distribution("papertilling-rs")
    except PackageNotFoundError:
        pytest.skip("the optional native wheel is not installed")

    assert native.metadata["License-Expression"] == "GPL-3.0-only"
    assert "LICENSE" in native.metadata.get_all("License-File", [])
    licenses = [p for p in native.files or () if p.as_posix().endswith(".dist-info/licenses/LICENSE")]
    assert len(licenses) == 1
    project_license = Path(__file__).resolve().parents[1] / "LICENSE"
    assert licenses[0].read_text(encoding="utf-8") == project_license.read_text(encoding="utf-8")

    revision = os.environ.get("GITHUB_SHA")
    repository = f"https://github.com/{os.environ.get('GITHUB_REPOSITORY', '')}"
    if revision and f"Homepage, {repository}" in native.metadata.get_all("Project-URL", []):
        import papertilling_rs

        assert papertilling_rs.SOURCE_URL == f"{repository}/tree/{revision}"
