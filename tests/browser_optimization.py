"""Optional real-browser regression: python tests/browser_optimization.py.

Requires Playwright and its Chromium browser. Starts and stops its own server.
Screenshots and a JSON check report are written to the chosen output directory.
"""

import argparse
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
from playwright.sync_api import sync_playwright, expect


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chromium", help="Optional installed Chromium executable")
    parser.add_argument("--output", type=Path, default=ROOT / "development/optimization/output_browser")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    address = f"http://127.0.0.1:{port}"
    code = f"from turbodash.apps.turbine_optimization.app import app; app.run(port={port}, debug=False)"
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
                browser = playwright.chromium.launch(headless=True, **({"executable_path": args.chromium} if args.chromium else {}))
                try:
                    report = exercise(browser, address, args.output)
                finally:
                    browser.close()
            (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(report, indent=2))
        finally:
            server.terminate()
            server.wait(timeout=10)


def exercise(browser, address, output):
    context = browser.new_context(viewport={"width": 1700, "height": 1100})
    page = context.new_page()
    errors, results, applied_results = [], [], []
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
            if "opt-applied-result" in body:
                applied_results.append(body["opt-applied-result"]["data"])

    page.on("response", capture)
    page.goto(address)
    page.wait_for_selector("#perf_table .dash-table-container")
    page.locator("#opt-panel > summary").click()
    page.get_by_role("tab", name="Design variables", exact=True).click()
    page.locator("#opt-add-variable").click()
    expect(page.locator('#opt-variable-rows .td-opt-row')).to_have_count(1)
    first = page.locator('#opt-variable-rows .td-opt-row').first
    numbers = first.locator('input[type="number"]')
    expect(numbers.nth(1)).to_have_attribute("placeholder", "Current: 0.6")
    numbers.nth(0).fill("0.5")
    numbers.nth(1).fill("0.9")
    numbers.nth(2).fill("0.65")
    numbers.nth(2).blur()
    page.locator("#opt-add-variable").click()
    expect(page.locator('#opt-variable-rows .td-opt-row')).to_have_count(2)
    expect(numbers.nth(1)).to_have_value("0.9")
    page.get_by_role("button", name="Remove variable", exact=True).last.click()
    expect(page.locator('#opt-variable-rows .td-opt-row')).to_have_count(1)
    page.get_by_role("tab", name="Constraints", exact=True).click()
    page.locator("#opt-add-constraint").click()
    expect(page.locator('#opt-constraint-rows .td-opt-row')).to_have_count(1)
    page.locator("#opt-add-constraint").click()
    expect(page.locator('#opt-constraint-rows .td-opt-row')).to_have_count(2)
    page.get_by_role("button", name="Remove constraint", exact=True).last.click()
    page.get_by_role("button", name="Remove constraint", exact=True).last.click()
    expect(page.locator('#opt-constraint-rows .td-opt-row')).to_have_count(0)
    page.get_by_role("tab", name="Design variables", exact=True).click()
    page.screenshot(path=output / "editor.png")

    def run(expected_success=True):
        page.locator("#opt-start").click()
        try:
            expect(page.locator("#opt-start")).to_be_disabled(timeout=10000)
        except AssertionError:
            raise AssertionError(page.locator("#opt-status").inner_text())
        # The other panes stay interactive while the numerical worker runs.
        page.get_by_text("Turbine geometry", exact=True).first.click()
        page.get_by_text("Meanline", exact=True).first.click()
        expect(page.locator("#opt-start")).to_be_enabled(timeout=60000)
        status = page.locator("#opt-status").inner_text()
        if expected_success:
            assert "Optimization applied" in status, status
            expect(page.locator("#opt-log")).to_contain_text("Infeasibility")
            expect(page.locator("#opt-log")).to_contain_text("SUCCESS")
        return status

    run()
    expect(page.locator("#opt-log")).to_contain_text("clipped from 0.9 to 0.65")
    ratio = page.locator('input[id*=blade_velocity_ratio]')
    expect(ratio).to_have_value("0.65")
    page.wait_for_timeout(700)
    assert abs(results[-1]["inputs"]["blade_velocity_ratio"] - .65) < 1e-8
    assert len(results[-1]["stages_performance"]) == 1
    # A manual edit still feeds the same plots, table and source-of-truth store.
    ratio.fill("0.61")
    ratio.blur()
    expect(page.locator('#opt-variable-rows input[placeholder="Current: 0.61"]')).to_have_count(1, timeout=15000)
    assert abs(results[-1]["inputs"]["blade_velocity_ratio"] - .61) < 1e-8

    # Two scalar equality constraints with one free variable must fail safely.
    page.get_by_role("tab", name="Constraints", exact=True).click()
    page.locator("#opt-add-constraint").click()
    before = deepcopy(results[-1]["inputs"])
    failed_status = run(False)
    expect(page.locator("#opt-log")).to_contain_text("more scalar equality constraints")
    assert results[-1]["inputs"] == before
    expect(ratio).to_have_value("0.61")
    page.get_by_role("button", name="Remove constraint", exact=True).click()

    def upload(config, filename):
        page.locator('#load_button input[type="file"]').set_input_files({
            "name": filename, "mimeType": "text/yaml", "buffer": yaml.safe_dump(config).encode(),
        })

    axial = yaml.safe_load((ROOT / "development/optimization/case_axial.yaml").read_text())
    axial["inputs"].update(rotational_speed_type="actual", rotational_speed=30000)
    axial["optimization"]["design_variables"] = {"rotational_speed": {"lower_value": 25000, "upper_value": 35000}}
    axial["optimization"]["constraints"] = []
    upload(axial, "actual-speed.yaml")
    page.get_by_role("tab", name="Design variables", exact=True).click()
    expect(page.locator('#opt-variable-rows')).to_contain_text("Rotational speed [rpm]")
    expect(page.locator('#opt-variable-rows input[type="number"]').first).to_have_value("25000")
    run()
    page.wait_for_timeout(700)
    actual = results[-1]
    assert flat_speed(actual["inputs"])[0] == "actual"
    assert 25000 <= flat_speed(actual["inputs"])[1] <= 35000

    radial = yaml.safe_load((ROOT / "development/optimization/case_radial.yaml").read_text())
    upload(radial, "radial.yaml")
    expect(page.locator('#opt-variable-rows .td-opt-row')).to_have_count(12)
    expect(page.locator('#opt-variable-rows input[type="number"]').first).to_have_value("0.5")
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
    other = browser.new_context()
    second = other.new_page()
    second.goto(address)
    second.wait_for_selector("#perf_table .dash-table-container")
    second_log = second.locator("#opt-log").text_content()
    assert second_log.startswith("Convergence output will appear"), second_log
    assert context.cookies()[0]["value"] != other.cookies()[0]["value"]
    other.close()
    assert not errors, errors
    context.close()
    return dict(checks=["add/remove editors", "current defaults", "numeric override retention", "clipping log",
                        "responsive panes and duplicate-start prevention", "atomic result/control update", "manual editing",
                        "failed-worker recovery", "actual rpm", "radial zero-flaring solve", "YAML bounds retained", "browser session isolation"],
                failed_run_diagnostic=failed_status, browser_errors=errors,
                radial_efficiency=final["overall_performance"]["efficiency_ts"])


def flat_speed(inputs):
    rotation = inputs["rotational_speed"]
    return (rotation["type"], rotation["value"]) if isinstance(rotation, dict) else (inputs["rotational_speed_type"], rotation)


if __name__ == "__main__":
    main()
