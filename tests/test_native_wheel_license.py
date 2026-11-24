"""Check the license and source information shipped with the optional native extension."""

from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

import pytest


def test_native_wheel_includes_project_license():
    try:
        native = distribution("papertilling-rs")
    except PackageNotFoundError:
        pytest.skip("the optional native wheel is not installed")

    assert native.metadata["License-Expression"] == "GPL-3.0-only"
    assert {"LICENSE", "SOURCE-NOTICE.txt"} <= set(native.metadata.get_all("License-File", []))
    licenses = [p for p in native.files or () if p.as_posix().endswith(".dist-info/licenses/LICENSE")]
    assert len(licenses) == 1
    project_license = Path(__file__).resolve().parents[1] / "LICENSE"
    assert licenses[0].read_text(encoding="utf-8") == project_license.read_text(encoding="utf-8")

    notices = [p for p in native.files or () if p.as_posix().endswith(".dist-info/licenses/SOURCE-NOTICE.txt")]
    assert len(notices) == 1
    assert "https://github.com/antimajorana/Escher/releases" in notices[0].read_text(encoding="utf-8")
