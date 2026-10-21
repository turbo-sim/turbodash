"""Optional real-browser regression: python tests/browser_optimization.py.

Requires Playwright and its Chromium browser. Starts and stops its own server.
Screenshots and a JSON check report are written to the chosen output directory.
"""

import argparse
import base64
from copy import deepcopy
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

import yaml
import numpy as np
from playwright.sync_api import sync_playwright, expect
from browser_picker_helpers import choose_path

from turbodash.visualization import plotly_turbine


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chromium", help="Optional installed Chromium executable")
    parser.add_argument("--output", type=Path, default=ROOT / "development/app_design/output_optimization_browser")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    for folder in ("temp", "profile", "downloads", "cache"):
        (args.output / folder).mkdir(parents=True, exist_ok=True)
    os.environ.update(TEMP=str(args.output / "temp"), TMP=str(args.output / "temp"), PYTHONDONTWRITEBYTECODE="1")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    address = f"http://127.0.0.1:{port}"
    code = f"from turbodash.apps.app_turbine_optimization.app import app; app.run(port={port}, debug=False)"
    with (args.output / "server.log").open("w", encoding="utf-8") as log:
        server = subprocess.Popen([sys.executable, "-c", code], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                  creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            deadline = time.monotonic() + 30
            while True:
                try:
                    urlopen(address, timeout=1).close()
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise RuntimeError("Browser test server did not start")
                    time.sleep(.1)
            with sync_playwright() as playwright:
                context = playwright.chromium.launch_persistent_context(
                    str(args.output / "profile"), headless=True,
                    viewport={"width": 1700, "height": 1100},
                    downloads_path=str(args.output / "downloads"), accept_downloads=True,
                    args=["--disable-background-networking", "--no-first-run", "--disable-component-update",
                          "--disable-crash-reporter", f"--disk-cache-dir={args.output / 'cache'}"],
                    **({"executable_path": args.chromium} if args.chromium else {}))
                try:
                    report = exercise(context, address, args.output)
                finally:
                    context.close()
            (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(report, indent=2))
        finally:
            server.terminate()
            server.wait(timeout=10)


def exercise(context, address, output):
    page = context.new_page()
    errors, results, applied_results, rendered_results = [], [], [], []
    page.on("pageerror", lambda error: errors.append(str(error)))

    def capture(response):
        if "_dash-update-component" not in response.url:
            return
        if response.status >= 400:
            errors.append(f"Callback HTTP {response.status}: {response.request.post_data[:120]}")
        if response.status == 200:
            try:
                body = response.json().get("response", {})
            except Exception:
                return
            if "result_store" in body:
                results.append(body["result_store"]["data"])
                rendered_results.append(body)
            if "opt-applied-result" in body:
                applied_results.append(body["opt-applied-result"]["data"])

    page.on("response", capture)
    page.goto(address)
    page.wait_for_selector("#perf_table .dash-table-container")
    assert page.evaluate("""() => {
        const stages = document.getElementById('stages_accordion_container');
        const panel = document.getElementById('opt-panel');
        return panel.closest('#design-controls') !== null
            && !!(stages.compareDocumentPosition(panel) & Node.DOCUMENT_POSITION_FOLLOWING);
    }"""), "Optimization must follow every stage's controls in the left sidebar"
    def open_section(identity):
        section = page.locator(f"#{identity}")
        if section.get_attribute("open") is None:
            section.locator(":scope > summary").click()
        expect(section).to_have_attribute("open", "")

    open_section("opt-panel")
    page.get_by_role("tab", name="Design variables", exact=True).click()
    choose_path(page, "variable", "blade_velocity_ratio")
    expect(page.locator('#opt-variable-rows .td-opt-row')).to_have_count(1)
    first = page.locator('#opt-variable-rows .td-opt-row').first
    numbers = first.locator('input[type="number"]')
    expect(numbers.nth(1)).to_have_attribute("placeholder", "Current: 0.6")
    numbers.nth(0).fill("0.5")
    numbers.nth(1).fill("0.9")
    numbers.nth(2).fill("0.65")
    numbers.nth(2).blur()
    choose_path(page, "variable", "rotational_speed")
    expect(page.locator('#opt-variable-rows .td-opt-row')).to_have_count(2)
    expect(numbers.nth(1)).to_have_js_property("valueAsNumber", 0.9)
    page.get_by_role("button", name="Remove variable", exact=True).last.click()
    expect(page.locator('#opt-variable-rows .td-opt-row')).to_have_count(1)
    page.get_by_role("tab", name="Constraints", exact=True).click()
    choose_path(page, "constraint", "$stages_performance.*.geometry_meanline.*.flaring_angle")
    expect(page.locator('#opt-constraint-rows .td-opt-row')).to_have_count(1)
    choose_path(page, "constraint", "$overall_performance.maximum_mach_number")
    expect(page.locator('#opt-constraint-rows .td-opt-row')).to_have_count(2)
    page.get_by_role("button", name="Remove constraint", exact=True).last.click()
    page.get_by_role("button", name="Remove constraint", exact=True).last.click()
    expect(page.locator('#opt-constraint-rows .td-opt-row')).to_have_count(0)
    page.get_by_role("tab", name="Design variables", exact=True).click()
    page.screenshot(path=output / "editor.png")

    def check_applied_design():
        applied = applied_results[-1]
        expected = applied["config"]["inputs"]
        displayed = results[-1]["inputs"]
        assert displayed["stages"] == expected["stages"]
        assert flat_speed(displayed) == flat_speed(expected)
        controls = page.evaluate("""() => Array.from(document.querySelectorAll('input[type="number"][id]'))
            .flatMap(input => {
                try {
                    const id = JSON.parse(input.id);
                    return id.scope === 'overall' || (id.scope === 'stage' && id.elem === 'input')
                        ? [{id, value: input.value}] : [];
                } catch (_) { return []; }
            })""")
        stage_fields = set()
        for control in controls:
            identity = control["id"]
            key = identity["key"]
            if identity["scope"] == "overall":
                value = flat_speed(expected)[1] if key == "rotational_speed_value" else expected[key]
            else:
                stage = expected["stages"][identity["stage"]]
                row, _, parameter = key.partition(".")
                value = stage["blade_geometry"][row][parameter] if parameter else stage[key]
                stage_fields.add((identity["stage"], key))
            assert np.isclose(float(control["value"]), value, rtol=1e-10, atol=1e-12), control
        assert {index for index, _ in stage_fields} == set(range(len(expected["stages"])))
        for index, stage in enumerate(expected["stages"]):
            expected_fields = {key for key, value in stage.items() if isinstance(value, (int, float))}
            expected_fields.update(f"{row}.{key}" for row in ("stator", "rotor")
                                   for key in stage["blade_geometry"][row]
                                   if key != "aspect_ratio" or expected["turbine_type"] != "radial")
            assert {key for stage_index, key in stage_fields if stage_index == index} == expected_fields
        assert page.locator('#stages_accordion_container .rc-slider').count() == 0

        # The primary callback publishes the model and overview figures; the
        # independent triangle callback renders that model in the selected units.
        # Derive its expected figures from the published model, avoiding a stale
        # triangle callback snapshot while the optimized design is being applied.
        figures = {}
        published = rendered_results[-1]
        for identity, props in published.items():
            if "figure" in props:
                figures[identity] = props["figure"]
        selected_units = page.locator("#td-triangle-mode label:has(input:checked)").inner_text()
        mode = {"Mach numbers": "mach", "Velocity [m/s]": "velocity"}[selected_units.strip()]
        for index, figure in enumerate(
            plotly_turbine.plot_velocity_triangles_turbine(
                published["result_store"]["data"], mode=mode
            )
        ):
            figures[f"triangles_plot_{index}"] = json.loads(figure.to_json())
        assert len(figures) == 4 + len(expected["stages"])
        samples = {}
        for identity, figure in figures.items():
            samples[identity] = []
            for trace in figure["data"]:
                axes = {}
                for axis in ("x", "y"):
                    values = trace.get(axis, [])
                    if isinstance(values, dict) and "bdata" in values:
                        values = np.frombuffer(base64.b64decode(values["bdata"]), dtype=values["dtype"]).tolist()
                    axes[axis] = [len(values), *[values[i] for i in (0, len(values) // 2, len(values) - 1)]] if len(values) else [0]
                samples[identity].append(axes)
        page.wait_for_function("""expected => {
            const same = (a, b) => typeof b === 'number'
                ? typeof a === 'number' && Math.abs(a - b) <= 1e-10 * Math.max(1, Math.abs(b))
                : a === b;
            return Object.entries(expected).every(([id, traces]) => {
                const graph = document.getElementById(id)?.querySelector('.js-plotly-plot');
                if (!graph?._fullData || graph._fullData.length !== traces.length) return false;
                return traces.every((trace, i) => Object.entries(trace).every(([axis, sample]) => {
                    const values = graph._fullData[i][axis] || [];
                    const actual = values.length
                        ? [values.length, values[0], values[Math.floor(values.length / 2)], values[values.length - 1]]
                        : [0];
                    return actual.length === sample.length && actual.every((value, j) => same(value, sample[j]));
                }));
            });
        }""", arg=samples)

    def run(expected_success=True):
        open_section("opt-panel")
        active_tab = page.locator("#opt-tabs .nav-link.active").inner_text()
        page.locator("#opt-start").click()
        try:
            expect(page.locator("#opt-start")).to_be_disabled(timeout=10000)
        except AssertionError:
            raise AssertionError(page.locator("#opt-status").text_content())
        # The other panes stay interactive while the numerical worker runs.
        page.locator('.td-workspace-rail [data-workspace="geometry"]').click()
        page.locator('.td-workspace-rail [data-workspace="meanline"]').click()
        expect(page.locator("#opt-start")).to_be_enabled(timeout=60000)
        expect(page.locator("#opt-panel")).to_have_js_property("open", True)
        expect(page.locator("#opt-tabs .nav-link.active")).to_have_text(active_tab)
        status = page.locator("#opt-status").text_content()
        if expected_success:
            assert "Optimization applied" in status, status
            expect(page.locator("#opt-log")).to_contain_text("Infeasibility")
            expect(page.locator("#opt-log")).to_contain_text("Success: True")
            check_applied_design()
        return status

    run()
    expect(page.locator("#opt-log")).to_contain_text("clipped from 0.9 to 0.65")
    ratio = page.locator('input[id*=blade_velocity_ratio]')
    expect(ratio).to_have_js_property("valueAsNumber", 0.65)
    page.wait_for_timeout(700)
    assert abs(results[-1]["inputs"]["blade_velocity_ratio"] - .65) < 1e-8
    assert len(results[-1]["stages_performance"]) == 1
    # A manual edit still feeds the same plots, table and source-of-truth store.
    open_section("td-section-speed")
    ratio.fill("0.61")
    ratio.blur()
    expect(page.locator('#opt-variable-rows input[placeholder="Current: 0.61"]')).to_have_count(1, timeout=15000)
    assert abs(results[-1]["inputs"]["blade_velocity_ratio"] - .61) < 1e-8

    # Two scalar equality constraints with one free variable must fail safely.
    open_section("opt-panel")
    page.get_by_role("tab", name="Constraints", exact=True).click()
    choose_path(page, "constraint", "$stages_performance.*.geometry_meanline.*.flaring_angle")
    comparison = page.locator('#opt-constraint-rows [id*="opt-comparison"]')
    comparison.click()
    comparison.get_by_text("=", exact=True).click()
    before = deepcopy(results[-1]["inputs"])
    failed_status = run(False)
    expect(page.locator("#opt-log")).to_contain_text("more scalar equality constraints")
    assert results[-1]["inputs"] == before
    expect(ratio).to_have_js_property("valueAsNumber", 0.61)
    page.get_by_role("button", name="Remove constraint", exact=True).click()

    def upload(config, filename):
        before = len(results)
        page.locator('#load_button input[type="file"]').set_input_files({
            "name": filename, "mimeType": "text/yaml", "buffer": yaml.safe_dump(config).encode(),
        })
        deadline = time.monotonic() + 30
        expected = config["inputs"]
        while (len(results) <= before or results[-1]["inputs"]["turbine_type"] != expected["turbine_type"]
               or len(results[-1]["inputs"]["stages"]) != len(expected["stages"])
               or flat_speed(results[-1]["inputs"]) != flat_speed(expected)):
            if time.monotonic() > deadline:
                raise AssertionError("Loaded design did not calculate: " + page.locator("#td-load-error").inner_text()
                                     + page.locator("#td-calculation-status").inner_text())
            page.wait_for_timeout(100)
        expect(page.locator("#opt-panel")).to_have_js_property("open", False)
        open_section("opt-panel")

    axial = yaml.safe_load((ROOT / "demos/optimization/case_axial.yaml").read_text())
    axial["inputs"].update(rotational_speed_type="actual", rotational_speed=30000)
    axial["optimization"]["design_variables"] = {"rotational_speed": {"lower_value": 25000, "upper_value": 35000}}
    axial["optimization"]["constraints"] = []
    upload(axial, "actual-speed.yaml")
    page.get_by_role("tab", name="Design variables", exact=True).click()
    expect(page.locator('#opt-variable-rows')).to_contain_text("Rotational speed [rpm]")
    expect(page.locator('#opt-variable-rows input[type="number"]').first).to_have_js_property("valueAsNumber", 25000)
    run()
    page.wait_for_timeout(700)
    actual = results[-1]
    assert flat_speed(actual["inputs"])[0] == "actual"
    assert 25000 <= flat_speed(actual["inputs"])[1] <= 35000

    # Force global, stage and blade values to change across two stages, then
    # verify every mounted numeric input and plotted result after apply.
    multiple = deepcopy(axial)
    multiple["inputs"].update(rotational_speed_type="specific", rotational_speed=.7)
    multiple["inputs"]["stages"].append(deepcopy(multiple["inputs"]["stages"][0]))
    multiple["inputs"]["stages"][1]["name"] = "stage_2"
    multiple["optimization"]["design_variables"] = {
        "blade_velocity_ratio": dict(lower_value=.62, upper_value=.64),
        "stages.0.degree_reaction": dict(lower_value=.55, upper_value=.6),
        "stages.1.degree_reaction": dict(lower_value=.35, upper_value=.45),
        "stages.1.blade_geometry.rotor.solidity_chord": dict(lower_value=1.3, upper_value=1.4),
    }
    upload(multiple, "two-stages.yaml")
    expect(page.locator('#stage_count_label')).to_have_text("2")
    expect(page.locator('#opt-variable-rows .td-opt-row')).to_have_count(4)
    page.locator("#td-triangle-mode").get_by_text("Velocity [m/s]", exact=True).click()
    page.wait_for_function(
        "document.querySelector('#triangles_plot_0 .js-plotly-plot').layout.xaxis.title.text.includes('m/s')"
    )
    run()
    assert len(results[-1]["stages_performance"]) == 2
    expect(page.locator("#td-triangle-mode label:has(input:checked)")).to_have_text("Velocity [m/s]")

    radial = yaml.safe_load((ROOT / "demos/optimization/case_radial.yaml").read_text())
    upload(radial, "radial.yaml")
    expect(page.locator('#opt-variable-rows .td-opt-row')).to_have_count(12)
    expect(page.locator('#opt-variable-rows input[type="number"]').first).to_have_js_property("valueAsNumber", 0.5)
    run()
    page.wait_for_timeout(700)
    final = results[-1]
    assert final["inputs"]["stages"] == applied_results[-1]["config"]["inputs"]["stages"]
    assert flat_speed(final["inputs"]) == flat_speed(applied_results[-1]["config"]["inputs"])
    assert final["inputs"]["turbine_type"] == "radial"
    for row in ("stator", "rotor"):
        assert abs(final["stages_performance"][0]["geometry_meanline"][row]["flaring_angle"]) < 1e-5
    page.locator('#opt-terminal').scroll_into_view_if_needed()
    page.screenshot(path=output / "radial-terminal.png")

    # Independent browser contexts get distinct cookie-backed owners and editors.
    other = context.browser.new_context()
    second = other.new_page()
    second.goto(address)
    second.wait_for_selector("#perf_table .dash-table-container")
    second_log = second.locator("#opt-log").text_content()
    assert second_log.startswith("Convergence output will appear"), second_log
    assert context.cookies()[0]["value"] != other.cookies()[0]["value"]
    other.close()
    assert not errors, errors
    return dict(checks=["add/remove editors", "current defaults", "numeric override retention", "clipping log",
                        "responsive panes and duplicate-start prevention", "optimization below every stage",
                        "all input boxes and plotted traces after apply", "two-stage global/stage/blade apply",
                        "velocity unit selection retained across apply", "manual editing",
                        "failed-worker recovery", "actual rpm", "radial zero-flaring solve", "YAML bounds retained", "browser session isolation",
                        "optimization panel and active tab stay open after successful and failed runs",
                        "explicit YAML loads still collapse the optimization panel"],
                failed_run_diagnostic=failed_status, browser_errors=errors,
                radial_efficiency=final["overall_performance"]["efficiency_ts"])


def flat_speed(inputs):
    rotation = inputs["rotational_speed"]
    return (rotation["type"], rotation["value"]) if isinstance(rotation, dict) else (inputs["rotational_speed_type"], rotation)


if __name__ == "__main__":
    main()
