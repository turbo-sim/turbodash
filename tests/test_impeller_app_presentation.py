"""Dimensional display and paired engineering figures for the impeller app."""

from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pytest

from turbodash.apps.app_impeller import controls, plots
from turbodash.geometry.impeller import build_impeller, load_config


@pytest.fixture(scope="module")
def geometry():
    config = load_config(
        Path(__file__).parents[1]
        / "development/impeller/examples/reversible_impeller_zero_rake.yaml"
    )
    config["numerics"].update(
        streamwise_points=41, spanwise_points=5, integration_points=501
    )
    return build_impeller(config)


def test_control_dimensions_and_exact_typed_value(geometry):
    config = geometry.config
    path = "sections.0.thickness.maximum_thickness_m"
    native = controls.get_value(config, path)
    assert controls.to_display(path, native) == native * 1000
    typed = 4.123456789
    assert controls.from_display(path, typed) == typed / 1000
    assert controls.from_display("blade_count", 19) == 19
    with pytest.raises(ValueError, match="integer"):
        controls.from_display("blade_count", 19.2)
    with pytest.raises(ValueError, match="finite"):
        controls.from_display(path, float("nan"))


def test_shared_hub_shroud_figures_match_evaluated_sections(geometry):
    figures = plots.build_figures(geometry)
    for key in ("angles", "thickness"):
        assert [trace.name for trace in figures[key].data] == ["Hub", "Shroud"]
        for trace, section in zip(
            figures[key].data, (geometry.sections[0], geometry.sections[-1])
        ):
            assert_allclose(trace.x, geometry.u, rtol=0, atol=0)
            expected = (
                np.degrees(section.beta)
                if key == "angles"
                else section.target_thickness_m * 1000
            )
            assert_allclose(trace.y, expected, rtol=0, atol=0)
    assert figures["thickness"].data[1].line.dash == "dash"
    assert figures["channel"].layout.yaxis.scaleanchor == "x"
    assert_allclose(
        figures["stacking"].data[1].y,
        np.degrees(geometry.sections[0].theta[-1]),
        atol=1e-12,
    )


def test_control_batch_order_matches_rendered_fields(geometry):
    def walk(component):
        yield component
        children = getattr(component, "children", None)
        if children is not None:
            for child in (
                children if isinstance(children, (list, tuple)) else [children]
            ):
                yield from walk(child)

    numeric_paths = []
    for component in walk(controls.sidebar(geometry.config)):
        identity = getattr(component, "id", None)
        if isinstance(identity, dict) and identity.get("type") == "impeller-number":
            numeric_paths.append(identity["path"])
    assert numeric_paths == list(controls.FIELD_PATHS)
    view = controls.control_view(geometry.config)
    disabled = dict(zip(controls.FIELD_PATHS, view["disabled"]))
    assert disabled["sections.-1.stacking_offset_deg"]
    assert not disabled["rake.angle_deg"]
