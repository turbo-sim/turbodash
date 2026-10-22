"""Keep relocated studies runnable and their local documentation navigable."""

import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote

import pytest


ROOT = Path(__file__).resolve().parents[1]
STUDIES = ROOT / "development"


@pytest.mark.parametrize("relative,root_variable", [
    ("loss_model_analysis/ainley_mathieson_fitting/run.py", "ROOT"),
    ("loss_model_analysis/ainley_mathieson_interpolation/study.py", "REPO"),
    ("loss_model_analysis/kacker_okapuu_shock/study.py", "REPO"),
    ("loss_model_analysis/kacker_okapuu_trailing_edge/digitize.py", "REPO"),
    ("loss_model_analysis/loss_model_audit/individual_functions/scripts/common.py", "REPO"),
    ("loss_model_analysis/loss_model_audit/overall_stage/study.py", "REPO"),
    ("optimization/verify_kkt.py", "ROOT"),
    ("blade_parametrization/parametrization_camberline/sensitivity_study.py", "REPO"),
    ("blade_parametrization/parametrization_thickness/sensitivity_study.py", "REPO"),
])
def test_studies_resolve_repository_outside_checkout(relative, root_variable, tmp_path):
    script = STUDIES / relative
    code = "\n".join([
        "import pathlib, runpy, sys",
        f"sys.path.insert(0, {str(script.parent)!r})",
        f"namespace = runpy.run_path({str(script)!r}, run_name='relocation_probe')",
        f"assert namespace[{root_variable!r}].resolve() == pathlib.Path({str(ROOT)!r})",
        "import turbodash",
        f"assert pathlib.Path(turbodash.__file__).resolve().is_relative_to(pathlib.Path({str(ROOT)!r}))",
    ])
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=tmp_path,
                            env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("name", ["loss_model_analysis", "blade_parametrization", "optimization"])
def test_study_markdown_links_resolve_after_relocation(name):
    failures = []
    assert (STUDIES / name).is_dir(), f"Study directory is missing: {name}"
    for document in (STUDIES / name).rglob("*.md"):
        if any(part.startswith((".cache", "output", "__pycache__")) for part in document.parts):
            continue
        text = re.sub(r"```.*?```", "", document.read_text(encoding="utf-8-sig"), flags=re.S)
        for match in re.finditer(r"!?\[[^\]]*\]\((<[^>]+>|[^)]+)\)", text):
            link = match.group(1).strip().strip("<>")
            if not link or link.startswith(("#", "http:", "https:", "mailto:", "data:")):
                continue
            target = unquote(link.split("#", 1)[0])
            if not (document.parent / target).resolve().exists():
                failures.append(f"{document.relative_to(ROOT)}: {link}")
    assert not failures, "\n".join(failures)


def test_source_papers_remain_available_to_digitization():
    from development.loss_model_analysis.pdf_sources import CHARTS, ROOT as PAPER_ROOT

    for relative, *_ in CHARTS.values():
        with (PAPER_ROOT / relative).open("rb") as stream:
            assert stream.read(5) == b"%PDF-"


def test_blade_studies_retain_formulation_and_production_kernel_reference():
    directory = STUDIES / "blade_parametrization"
    formulation = directory / "parametrization_blades/README.md"
    guide = directory / "parametrization_blades/parameter_guide.md"
    assert formulation.is_file() and guide.is_file()
    assert "turbodash/geometry/blade_parametrization/geometry.py" in formulation.read_text(encoding="utf-8")
    for name in ("parametrization_camberline", "parametrization_thickness"):
        contents = (directory / name / "sensitivity_study.py").read_text(encoding="utf-8")
        assert "from turbodash.geometry.blade_parametrization" in contents
