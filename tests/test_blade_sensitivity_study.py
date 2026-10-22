"""Audit the reproducible gallery against independent production outputs.

The committed manifest is deliberately tested, not just the sweep helpers:
stale figures, extra input changes and silent parameter repairs must fail.
"""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from turbodash.geometry.blade_parametrization.geometry import (
    build_section, map_section_coordinates,
)
from turbodash.geometry.blade_parametrization.parameters import (
    FIELD_SPECS, load_case, resolve_parameters,
)
from turbodash.geometry.meanline import REACTION_PRESET, IMPULSE_PRESET


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs/source/_static/blade_sensitivity"
SPEC = importlib.util.spec_from_file_location(
    "blade_sensitivity_study", ROOT / "development/blade_parametrization/parameter_sensitivity/study.py"
)
study = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = study
SPEC.loader.exec_module(study)


@pytest.fixture(scope="module")
def manifest():
    return json.loads((ASSETS / "manifest.json").read_text(encoding="utf-8"))


def flatten(values):
    return {f"{group}.{key}": value for group, members in values.items() for key, value in members.items()}


def sweep_entry(manifest, path):
    return next(item for item in manifest["sweeps"] if item["parameter"] == path)


def test_all_current_gui_inputs_are_covered_without_retired_controls(manifest):
    hidden = study.hidden_gui_paths()
    paths = [entry["parameter"] for entry in manifest["sweeps"]]
    assert set(paths) | set(manifest["coverage"]["fixed_gui_inputs"]) == set(FIELD_SPECS) - hidden
    assert "cascade.blade_count" not in paths
    assert len(paths) == len(set(paths))
    assert set(manifest["coverage"]["hidden_kernel_inputs"]) == hidden
    assert all(not path.startswith("trailing_edge.") for path in paths)
    assert all("pressure_side" not in path for path in paths)


def test_every_record_changes_only_its_selected_independent_input(manifest):
    for entry in manifest["sweeps"]:
        path = entry["parameter"]
        for family, records in entry["families"].items():
            baseline = flatten(manifest["baselines"][f"{family}/{entry['context']}"]["inputs"])
            assert sum(record["is_baseline"] for record in records) == 1
            assert any(record["value"] == baseline[path] for record in records)
            if FIELD_SPECS[path]["type"] not in {"enum", "bool"}:
                assert len(records) == 4
            assert len({r["value"] for r in records}) == len(records)
            if FIELD_SPECS[path]["type"] == "int":
                assert all(type(record["value"]) is int for record in records)
            if FIELD_SPECS[path]["type"] == "enum":
                assert {record["value"] for record in records} == set(FIELD_SPECS[path]["options"])
            if FIELD_SPECS[path]["type"] == "bool":
                assert {record["value"] for record in records} == {False, True}
            for record in records:
                requested = flatten(record["requested_inputs"])
                delta = {key for key in baseline if baseline[key] != requested[key]}
                assert delta == (set() if record["is_baseline"] else {path})
                assert requested[path] == record["value"]
                assert requested["cascade.blade_count"] == baseline["cascade.blade_count"]
                assert record["changed_independent_paths"] == list(delta)
                if record["resolved_inputs"] is not None:
                    assert flatten(record["resolved_inputs"]) == requested
                if record["validation"]["status"] == "rejected":
                    assert record["validation"]["error"]
                else:
                    assert record["diagnostics"]["intersection_check"] == "sampled"
                    if record["is_baseline"]:
                        assert record["baseline_max_coordinate_error_m"] == 0


def test_a_rejected_attempt_still_records_exact_inputs_without_drawing_it(manifest):
    sweep = next(s for s in study.SWEEPS if s.path == "geometry.inlet_radius")
    baseline = build_section(manifest["baselines"]["reaction/radial"]["inputs"])
    record, section, _ = study.evaluate(sweep, baseline.parameters["geometry"]["outlet_radius"], baseline)
    assert record["validation"]["status"] == "rejected"
    assert "must differ" in record["validation"]["error"]
    assert record["changed_independent_paths"] == ["geometry.inlet_radius"]
    assert section is None


@pytest.mark.parametrize("family,preset", [("reaction", REACTION_PRESET), ("impulse", IMPULSE_PRESET)])
def test_exact_baselines_are_reference_dimensions_with_current_production_presets(manifest, family, preset):
    values = resolve_parameters(load_case(ROOT / f"turbodash/geometry/blade_parametrization/cases/{family}.yaml")).values
    for path, value in preset.items():
        group, key = path.split(".")
        values[group][key] = value
    for context in ("axial", "radial", "radial_corrected"):
        expected = deepcopy(values)
        expected["geometry"]["type"] = "axial" if context == "axial" else "radial"
        expected["correction"]["preserve_physical_thickness"] = context == "radial_corrected"
        record = manifest["baselines"][f"{family}/{context}"]
        assert record["inputs"] == expected
        section = build_section(expected)
        assert record["coordinate_sha256"] == study.coordinate_hash(section)


def test_leading_relief_changes_total_thickness_and_only_the_selected_surface(manifest):
    entry = sweep_entry(manifest, "leading_edge.relief_strength")
    for records in entry["families"].values():
        off = build_section(min(records, key=lambda r: r["value"])["resolved_inputs"])
        on = build_section(max(records, key=lambda r: r["value"])["resolved_inputs"])
        np.testing.assert_array_equal(off.camber, on.camber)
        np.testing.assert_array_equal(off.minus, on.minus)  # Both references turn positively.
        assert np.max(np.abs(off.plus-on.plus)) > 0
        assert np.max(off.thickness["measured_physical"]-on.thickness["measured_physical"]) > 0
        # This rules out plotting an unaffected side as the entire thickness.
        _, upper, _ = study.panel_arrays(on, off)
        np.testing.assert_allclose(upper[:, 1], 500*on.thickness["measured_physical"])


def test_radial_plot_and_thickness_use_production_physical_mapping(manifest):
    entry = sweep_entry(manifest, "correction.preserve_physical_thickness")
    for records in entry["families"].values():
        for record in records:
            section = build_section(record["resolved_inputs"])
            mapped = map_section_coordinates(section)
            np.testing.assert_array_equal(study.physical(section, "boundary"), 1000*mapped["boundary"][:, :2])
            thickness = np.linalg.norm(mapped["plus"]-mapped["minus"], axis=1)
            np.testing.assert_allclose(thickness, section.thickness["measured_physical"], atol=2e-14)
            assert study.coordinate_hash(section) == record["coordinate_sha256"]


def test_artifacts_and_physical_aspect_checks_match_manifest(manifest):
    names = set()
    panel_boxes = None
    for entry in manifest["sweeps"]:
        assert len(entry["physical_panel_checks"]) == 2*len(entry["figures"])
        for figure in entry["figures"]:
            assert (figure["rows"], figure["columns"]) == (1, 3)
            assert len(figure["panels"]) == 3
            assert all(panel["all_spines_visible"] for panel in figure["panels"])
            boxes = [panel["box"] for panel in figure["panels"]]
            if panel_boxes is None:
                panel_boxes = boxes
            np.testing.assert_allclose(boxes, panel_boxes, atol=2e-12, rtol=0)
        for panel in entry["physical_panel_checks"]:
            assert panel["physical_scale_ratio"] == pytest.approx(1, abs=1e-9)
            assert panel["xlim_mm"][0] < panel["xlim_mm"][1]
            assert panel["ylim_mm"][0] < panel["ylim_mm"][1]
        for artifact in entry["artifacts"]:
            path = ASSETS / artifact["path"]
            assert path.suffix in {".png", ".svg"}
            assert path.stat().st_size > 1000
            assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
            assert path.name not in names
            names.add(path.name)
    assert {path.name for path in ASSETS.glob("*.png")} | {path.name for path in ASSETS.glob("*.svg")} == names


def test_four_plasma_colors_and_thinner_lines_are_recorded(manifest):
    options = manifest["plot_options"]
    assert options["style_function"] == "turbodash.set_plot_options"
    assert options["colormap"] == "plasma"
    assert options["colormap_interval"] == [.25, .75]
    assert options["variation_linewidth"] == .85
    assert options["baseline_linewidth"] == 1.15
    for entry in manifest["sweeps"]:
        for records in entry["families"].values():
            positions = np.linspace(.25, .75, len(records))
            for record, position in zip(records, positions):
                assert record["colormap_position"] == pytest.approx(position)
                assert record["plot_color"] == study.matplotlib.colors.to_hex(study.matplotlib.colormaps["plasma"](position))


def test_actual_figure_has_three_boxed_panels_and_thin_curves(manifest, monkeypatch):
    entry = sweep_entry(manifest, "thickness.maximum_thickness_ratio")
    sweep = next(s for s in study.SWEEPS if s.path == entry["parameter"])
    baseline = build_section(manifest["baselines"]["reaction/axial"]["inputs"])
    cases = [(record, build_section(record["resolved_inputs"]), {}) for record in entry["families"]["reaction"]]
    captured = []

    def inspect(fig, stem, directory):
        assert len(fig.axes) == 3
        for ax in fig.axes:
            assert all(spine.get_visible() for spine in ax.spines.values())
            assert "?" not in ax.get_xlabel() + ax.get_ylabel()
            assert len(ax.lines) == 4
            assert sorted(line.get_linewidth() for line in ax.lines) == [.85, .85, .85, 1.15]
        assert len(fig.legends[0].get_texts()) == 4
        captured.append(stem)
        study.plt.close(fig)
        return []

    monkeypatch.setattr(study, "save_figure", inspect)
    with study.plt.rc_context():
        study.set_plot_options(fontsize=11, linewidth=.85)
        study.draw_family(sweep, "reaction", baseline, cases, ASSETS)
    assert captured


def test_recorded_kernel_sources_are_current(manifest):
    for relative, digest in manifest["source_sha256"].items():
        # The editor's complete hash is provenance, not a kernel dependency.
        # YAML dialogs/callbacks may change independently; the GUI field set
        # and hidden controls are checked explicitly by the coverage test.
        if relative.startswith(("turbo_moc/", "turbodash/apps/")):
            continue
        else:
            assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest, f"Regenerate gallery after changing {relative}"
