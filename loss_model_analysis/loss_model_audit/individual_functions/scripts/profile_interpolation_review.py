"""Refresh the profile interpolation figures, including the join close-up."""
import importlib.util
import json
import shutil
from common import BASE, REPO, sha256, write_json

STUDY = REPO / "loss_model_analysis/ainley_mathieson_interpolation"
DATA = BASE / "data/profile_interpolation"


def load(name):
    spec = importlib.util.spec_from_file_location("ko_profile_study_"+name, STUDY / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def calculate():
    module = load("study")
    module.calculate()
    result = module.verify()
    DATA.mkdir(parents=True, exist_ok=True)
    for name in (*(f"{n}.csv" for n in (*module.TABLES, *module.ZOOM_TABLES)),
                 "summary.json", "validation.json", "provenance.json"):
        shutil.copyfile(STUDY / "data" / name, DATA / name)
        assert sha256(STUDY / "data" / name) == sha256(DATA / name)
    write_json(BASE / "data/validation/profile_interpolation.json", result)
    return result


def render():
    return load("plot").render(DATA, BASE / "figures", prefix="ko_profile_")


def verify_saved():
    provenance = json.loads((DATA / "provenance.json").read_text(encoding="utf-8"))
    assert sha256(STUDY / "study.py") == provenance["calculation_script_sha256"]
    assert all(sha256(REPO / name) == digest for name, digest in provenance["production_sources"].items())
    assert all(sha256(DATA / name) == digest for name, digest in provenance["data_hashes"].items())
    validation = json.loads((DATA / "validation.json").read_text(encoding="utf-8"))
    assert validation["passed"] and validation["zero_zoom"]["passed"]


if __name__ == "__main__":
    print(calculate())
    print(render())
