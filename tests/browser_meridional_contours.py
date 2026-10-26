"""Real-browser coverage of the meridional contour display control.

Run with ``python tests/browser_meridional_contours.py --output <directory>``.
Uses the optimization browser harness, which starts and stops its own server.
"""

from copy import deepcopy
import time

import numpy as np
from playwright.sync_api import expect
import yaml

import browser_optimization as harness


def exercise(context, address, output):
    page = context.new_page()
    errors, results, figures = [], [], []
    page.on("pageerror", lambda error: errors.append(str(error)))

    def capture(response):
        if "_dash-update-component" not in response.url:
            return
        if response.status >= 400:
            errors.append(f"Callback HTTP {response.status}")
        if response.status != 200:
            return
        try:
            body = response.json().get("response", {})
        except Exception:
            return
        if "result_store" in body:
            results.append(body["result_store"]["data"])
        if "meridional_plot" in body:
            inputs = response.request.post_data_json.get("inputs", [])
            degree = next(
                (item["value"] for item in inputs if item["id"] == "td-contour-degree"),
                None,
            )
            figures.append((body["meridional_plot"]["figure"], degree))
        if body.get("td-load-error", {}).get("children"):
            errors.append(body["td-load-error"]["children"])

    page.on("response", capture)

    def rendered():
        return page.locator("#meridional_plot .js-plotly-plot").evaluate(
            """graph => graph.data.map(trace => ({
                name: trace.name, x: Array.from(trace.x || []),
                y: Array.from(trace.y || [])
            }))"""
        )

    def trace(traces, name):
        return next(item for item in traces if item["name"] == name)

    def points(item):
        return np.column_stack([item["x"], item["y"]])

    def wait_figure(after, degree):
        deadline = time.monotonic() + 30
        while True:
            matching = [
                figure for figure, selected in figures[after:]
                if selected == degree and any(
                    item.get("name") == "Hub" for item in figure.get("data", [])
                )
            ]
            if matching:
                break
            if time.monotonic() > deadline:
                raise AssertionError(f"No meridional redraw; errors={errors}")
            page.wait_for_timeout(50)
        expected = [
            {key: item[key] for key in ("name", "x", "y")}
            for item in matching[-1]["data"]
        ]
        page.wait_for_function(
            """expected => {
                const data = document.querySelector('#meridional_plot .js-plotly-plot')?.data;
                return data && data.length === expected.length && expected.every((trace, i) =>
                    trace.name === data[i].name && ['x', 'y'].every(axis =>
                        trace[axis].length === data[i][axis].length &&
                        trace[axis].every((value, j) => Math.abs(value - data[i][axis][j]) < 1e-9)));
            }""",
            arg=expected,
        )
        return rendered()

    def choose(degree):
        count = len(figures)
        page.locator("#td-contour-degree label").filter(
            has_text={1: "Linear", 3: "Smooth"}[degree]
        ).click()
        return wait_figure(count, degree)

    def check_stations(linear, curved):
        for name in ("Hub", "Shroud"):
            anchors = points(trace(linear, name))
            sampled = points(trace(curved, name))
            for anchor in anchors:
                assert np.min(np.linalg.norm(sampled - anchor, axis=1)) < 1e-8
        for name in ("Stator", "Rotor"):
            assert len(trace(curved, name)["x"]) > len(trace(linear, name)["x"])

    def check_extensions(traces, axial):
        # Straight extensions have identical transverse coordinates at their
        # two endpoints. Endpoint derivatives are checked by the numerical tests.
        axis = "y" if axial else "x"
        for name in ("Hub", "Shroud"):
            values = trace(traces, name)[axis]
            np.testing.assert_allclose(values[0], values[1], rtol=0, atol=1e-9)
            np.testing.assert_allclose(values[-1], values[-2], rtol=0, atol=1e-9)

    def screenshot(name):
        page.locator("#meridional_plot").locator("xpath=..").screenshot(
            path=str(output / name)
        )

    page.goto(address)
    page.wait_for_selector("#perf_table .dash-table-container")
    page.wait_for_selector("#td-contour-degree", state="attached")
    linear = wait_figure(0, 1)
    baseline_results = len(results)
    page.locator("#td-expand-all").click()
    expect(page.locator('[data-controls-workspace="meanline"] .td-section[open]')).to_have_count(6)
    expect(page.locator('[data-controls-workspace="meanline"] .td-subsection:not([open])')).to_have_count(0)
    expect(page.locator(".td-opt-picker[open]")).to_have_count(0)
    page.locator("#td-collapse-all").click()
    expect(page.locator('[data-controls-workspace="meanline"] .td-section[open]')).to_have_count(0)
    assert len(results) == baseline_results, "Expanding/collapsing sections recalculated the turbine"
    setup = page.locator("summary").filter(has_text="Turbine setup")
    section = setup.locator("xpath=..")
    if section.get_attribute("open") is None:
        setup.click()
    expect(page.locator("#td-contour-degree")).to_be_visible()
    expect(page.locator("#td-contour-degree label")).to_have_text(["Linear", "Smooth"])
    assert page.evaluate("""() => {
        const contour = document.getElementById('td-contour-degree');
        const type = document.getElementById(JSON.stringify({key: 'turbine_type', scope: 'overall'}));
        const count = document.getElementById('stage_count_label');
        return contour.closest('details') === type.closest('details') &&
            !!(type.compareDocumentPosition(contour) & Node.DOCUMENT_POSITION_FOLLOWING) &&
            !!(contour.compareDocumentPosition(count) & Node.DOCUMENT_POSITION_FOLLOWING);
    }"""), "Global contour selector must follow turbine type and precede stage count"
    expect(page.locator("#td-contour-degree label:has(input:checked)")).to_have_text(
        "Linear"
    )
    assert results
    baseline_count = len(results)
    baseline = deepcopy(results[-1])
    screenshot("axial-linear.png")

    smooth = choose(3)
    check_stations(linear, smooth)
    check_extensions(smooth, axial=True)
    screenshot("axial-smooth.png")
    page.screenshot(path=str(output / "axial-smooth-workbench.png"))
    page.wait_for_timeout(400)
    assert len(results) == baseline_count, "Display changes recalculated the turbine"
    assert results[-1] == baseline

    page.locator('.td-theme-toggle [data-theme="dark"]').click()
    page.wait_for_function(
        "document.querySelector('#meridional_plot .js-plotly-plot')._tdTheme === 'dark'"
    )
    choose(1)
    page.wait_for_function(
        "document.querySelector('#meridional_plot .js-plotly-plot')._tdTheme === 'dark'"
    )
    choose(3)
    screenshot("axial-smooth-dark.png")
    assert len(results) == baseline_count

    previous = len(figures)
    page.reload()
    page.wait_for_selector("#perf_table .dash-table-container")
    refreshed = wait_figure(previous, 3)
    setup = page.locator("summary").filter(has_text="Turbine setup")
    if setup.locator("xpath=..").get_attribute("open") is None:
        setup.click()
    expect(page.locator("#td-contour-degree label:has(input:checked)")).to_have_text(
        "Smooth"
    )
    assert len(trace(refreshed, "Hub")["x"]) == len(trace(smooth, "Hub")["x"])

    # A newly solved radial case must redraw in the retained smooth setting.
    config = yaml.safe_load(
        (harness.ROOT / "demos/optimization/case_radial.yaml").read_text(encoding="utf-8")
    )
    before_results, before_figures = len(results), len(figures)
    page.locator('#load_button input[type="file"]').set_input_files(
        {
            "name": "radial-contours.yaml",
            "mimeType": "application/yaml",
            "buffer": yaml.safe_dump(config).encode(),
        }
    )
    deadline = time.monotonic() + 30
    while len(results) <= before_results or results[-1]["inputs"]["turbine_type"] != "radial":
        if time.monotonic() > deadline:
            raise AssertionError(f"Radial design did not calculate; errors={errors}")
        page.wait_for_timeout(50)
    radial = wait_figure(before_figures, 3)
    setup = page.locator("summary").filter(has_text="Turbine setup")
    if setup.locator("xpath=..").get_attribute("open") is None:
        setup.click()
    expect(page.locator("#td-contour-degree label:has(input:checked)")).to_have_text(
        "Smooth"
    )
    check_extensions(radial, axial=False)
    screenshot("radial-smooth-dark.png")
    count = len(results)
    radial_linear = choose(1)
    radial_smooth = choose(3)
    check_stations(radial_linear, radial_smooth)
    check_stations(radial_linear, radial)
    check_extensions(radial_smooth, axial=False)
    page.wait_for_timeout(400)
    assert len(results) == count
    assert not errors, errors
    return {
        "passed": True,
        "checks": [
            "expand/collapse all sections without opening picker menus or recalculating",
            "Linear/Smooth axial and radial contours",
            "global selector between turbine type and stage count",
            "all station coordinates retained",
            "curved blade-row fills",
            "purely axial/radial straight extensions",
            "display changes do not recalculate",
            "dark theme retained",
            "selection retained after reload and new results",
        ],
        "browser_errors": errors,
    }


if __name__ == "__main__":
    harness.exercise = exercise
    harness.main()
