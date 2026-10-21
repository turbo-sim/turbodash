"""Optional browser regression for lossless numeric presentation in the workbench.

Starts its own local app; browser profile, temporary files, logs and downloads
are written only under development/app_design/output_numeric/.
"""

import argparse
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

import yaml
from browser_picker_helpers import choose_path


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--browser", default="C:/Program Files/Google/Chrome/Application/chrome.exe")
    args = parser.parse_args()
    output = ROOT / "development/app_design/output_numeric"
    for folder in ("temp", "profile", "downloads"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    os.environ.update(TEMP=str(output / "temp"), TMP=str(output / "temp"), PYTHONDONTWRITEBYTECODE="1")
    from playwright.sync_api import sync_playwright, expect

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    address = f"http://127.0.0.1:{port}"
    command = f"from turbodash.apps.app_turbine_optimization.app import app; app.run(host='127.0.0.1',port={port},debug=False)"
    checks, errors, results = [], [], []
    with (output / "server.log").open("w", encoding="utf-8") as log:
        server = subprocess.Popen([sys.executable, "-B", "-c", command], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                  creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            deadline = time.monotonic() + 30
            while True:
                try:
                    urlopen(address, timeout=1).close()
                    break
                except OSError:
                    if server.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError("Numeric-format test server did not start; see server.log")
                    time.sleep(.1)
            with sync_playwright() as playwright:
                context = playwright.chromium.launch_persistent_context(
                    str(output / "profile"), executable_path=args.browser, headless=True,
                    viewport={"width": 1680, "height": 1080}, downloads_path=str(output / "downloads"), accept_downloads=True,
                    args=["--disable-background-networking", "--no-first-run", "--disable-component-update", "--disable-crash-reporter"],
                )
                try:
                    page = context.new_page()
                    page.on("pageerror", lambda error: errors.append(str(error)))

                    def capture(response):
                        if "_dash-update-component" not in response.url:
                            return
                        if response.status >= 400:
                            errors.append(f"Callback HTTP {response.status}: {response.request.post_data[:150]}")
                        if response.status == 200:
                            try:
                                body = response.json().get("response", {})
                            except Exception:
                                # Context shutdown can abort an in-flight response body.
                                return
                            if "result_store" in body:
                                results.append(body["result_store"]["data"])

                    page.on("response", capture)
                    page.goto(address)
                    page.wait_for_selector("#perf_table .dash-table-container", timeout=60000)
                    page.wait_for_timeout(400)

                    def field(key, stage=None):
                        scope = "overall" if stage is None else "stage"
                        selector = f'input[id*=\'"scope":"{scope}"\'][id*=\'"key":"{key}"\']'
                        if stage is not None:
                            selector += f'[id*=\'"stage":{stage}\']'
                        return page.locator(selector)

                    def search(query):
                        page.keyboard.press("Control+k")
                        page.locator("#td-parameter-search").fill(query)
                        expect(page.locator(".td-search-result").first).to_be_visible()
                        page.locator("#td-parameter-search").press("Enter")

                    def wait_value(key, value):
                        for _ in range(80):
                            if results and results[-1]["inputs"][key] == value:
                                return
                            page.wait_for_timeout(100)
                        raise AssertionError(f"No successful result with {key}={value!r}")

                    ratio = field("blade_velocity_ratio")
                    mass = field("mass_flow_rate")
                    angle = field("inlet_flow_angle")
                    reaction = field("degree_reaction", 0)
                    expect(mass).to_have_value("10.000")
                    expect(ratio).to_have_value("0.600")
                    expect(angle).to_have_value("0.000")
                    expect(reaction).to_have_value("0.500")
                    expect(field("rotor.solidity_chord", 0)).to_have_value("1.137411139662947")
                    expect(page.locator("#opt-tolerance")).to_have_value("1.000e-8")
                    expect(page.locator("#opt-iterations")).to_have_value("150")
                    expect(page.locator("#stage_count_label")).to_have_text("1")
                    mesh_layers = page.locator('input[id*=\'"scope":"mesh-row"\'][id*=\'"key":"span_layers"\']').first
                    expect(mesh_layers).to_have_value("12")
                    mesh_stretch = page.locator('input[id*=\'"scope":"mesh-row"\'][id*=\'"key":"span_stretch"\']').first
                    expect(mesh_stretch).to_have_value("1.500")
                    checks.append("Default continuous values have at least three decimals; full precision and integer counts remain intact")

                    before = len(results)
                    for _ in range(5):
                        page.evaluate("TurbodashWorkbench.refresh()")
                    search("blade velocity ratio")
                    ratio.blur()
                    page.wait_for_timeout(500)
                    assert len(results) == before, "Presentation-only padding or unchanged blur reran the solver"
                    checks.append("Formatting and unchanged focus/blur generate no numerical recalculation")

                    ratio.fill("0.63")
                    page.evaluate("TurbodashWorkbench.refresh()")
                    page.wait_for_timeout(200)
                    expect(ratio).to_be_focused()
                    expect(ratio).to_have_value("0.63")
                    ratio.press("Enter")
                    wait_value("blade_velocity_ratio", .63)
                    expect(ratio).to_have_value("0.63")
                    ratio.blur()
                    expect(ratio).to_have_value("0.630")
                    checks.append("Focused editing and Enter commit retain typed text; blur pads shorter fractions")

                    ratio.fill("0.6200007")
                    ratio.blur()
                    wait_value("blade_velocity_ratio", .6200007)
                    expect(ratio).to_have_value("0.6200007")
                    before = len(results)
                    ratio.fill("0.6200007000")
                    ratio.blur()
                    expect(ratio).to_have_value("0.6200007000")
                    page.evaluate("TurbodashWorkbench.refresh()")
                    page.wait_for_timeout(300)
                    assert len(results) == before, "Trailing zeros changed the numerical model"
                    page.evaluate("id => dash_clientside.set_props(JSON.parse(id), {value: 0.6200007})", ratio.get_attribute("id"))
                    expect(ratio).to_have_value("0.6200007000")
                    checks.append("Additional significant digits and deliberate trailing zeros survive refresh without rounding or callbacks")

                    search("inlet flow angle")
                    angle.fill("-.5")
                    angle.blur()
                    expect(angle).to_have_value("-0.500")
                    wait_value("inlet_flow_angle", -.5)
                    angle.fill("-12.5")
                    angle.blur()
                    expect(angle).to_have_value("-12.500")
                    wait_value("inlet_flow_angle", -12.5)
                    angle.fill("-0")
                    angle.blur()
                    expect(angle).to_have_value("-0.000")
                    wait_value("inlet_flow_angle", 0)
                    checks.append("Negative values and signed zero retain their meaning")

                    search("degree reaction")
                    reaction.fill("10.1")
                    reaction.blur()
                    expect(reaction).to_have_value("10.1")
                    assert reaction.evaluate("input => !input.validity.valid")
                    reaction.fill("")
                    reaction.blur()
                    page.evaluate("TurbodashWorkbench.refresh()")
                    page.wait_for_timeout(150)
                    expect(reaction).to_have_value("")
                    reaction.focus()
                    reaction.press_sequentially("1e")
                    page.evaluate("TurbodashWorkbench.refresh()")
                    page.wait_for_timeout(150)
                    expect(reaction).to_have_value("")
                    reaction.fill("0.5")
                    reaction.blur()
                    expect(reaction).to_have_value("0.500")
                    checks.append("Out-of-range, blank and partially entered exponent drafts are never reformatted")

                    search("solver tolerance")
                    tolerance = page.locator("#opt-tolerance")
                    tolerance.fill("2.5E-8")
                    tolerance.blur()
                    expect(tolerance).to_have_value("2.500E-8")
                    tolerance.fill("2.50000e-8")
                    tolerance.blur()
                    expect(tolerance).to_have_value("2.50000e-8")
                    checks.append("Tiny scientific-notation tolerances keep their exponent and all supplied mantissa digits")

                    page.get_by_role("tab", name="Design variables", exact=True).click()
                    choose_path(page, "variable", "blade_velocity_ratio")
                    expect(page.locator("#opt-variable-rows .td-opt-row")).to_have_count(1)
                    first = page.locator("#opt-variable-rows .td-opt-row").first
                    numbers = first.locator('input[type="number"]')
                    expect(numbers.nth(1)).to_have_value("")
                    expect(numbers.nth(0)).to_have_value(re.compile(r"-?\d+\.\d{3,}(?:[eE][+-]?\d+)?"))
                    numbers.nth(1).fill("0.6200007000")
                    numbers.nth(1).blur()
                    choose_path(page, "variable", "rotational_speed")
                    expect(page.locator("#opt-variable-rows .td-opt-row")).to_have_count(2)
                    expect(numbers.nth(1)).to_have_value("0.6200007000")
                    page.get_by_role("button", name="Remove variable", exact=True).last.click()
                    expect(page.locator("#opt-variable-rows .td-opt-row")).to_have_count(1)
                    checks.append("Dynamic optimization bounds are padded while blanks and user precision survive row rebuilding")

                    # A property-only Dash update must be formatted without a synthetic input/change event.
                    mass_id = mass.get_attribute("id")
                    page.evaluate("id => dash_clientside.set_props(JSON.parse(id), {value: 11})", mass_id)
                    expect(mass).to_have_value("11.000")
                    wait_value("mass_flow_rate", 11)
                    page.wait_for_timeout(400)
                    with page.expect_download(timeout=20000) as pending:
                        page.locator("#save_button").click()
                    saved_path = output / "downloads" / "precision.yaml"
                    pending.value.save_as(saved_path)
                    saved = yaml.safe_load(saved_path.read_text(encoding="utf-8"))
                    assert saved["inputs"]["blade_velocity_ratio"] == .6200007
                    assert saved["inputs"]["mass_flow_rate"] == 11
                    assert saved["inputs"]["stages"][0]["blade_geometry"]["rotor"]["solidity_chord"] == 1.137411139662947
                    page.locator('#load_button input[type="file"]').set_input_files(saved_path)
                    expect(mass).to_have_value("11.000")
                    assert ratio.evaluate("input => input.valueAsNumber") == .6200007
                    expect(field("rotor.solidity_chord", 0)).to_have_value("1.137411139662947")
                    checks.append("Property-only updates and YAML save/reload retain exact numerical values independently of display padding")
                    assert not errors, errors
                    report = {"checks": checks, "browser_errors": errors, "successful_result_updates": len(results)}
                    destination = output / "report.json"
                    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
                    print(json.dumps(report, indent=2))
                except Exception:
                    page.screenshot(path=output / "failure.png", full_page=True)
                    (output / "failure.json").write_text(json.dumps({"checks": checks, "errors": errors,
                        "status": page.locator("#td-calculation-status").inner_text()}, indent=2), encoding="utf-8")
                    raise
                finally:
                    page.remove_listener("response", capture)
                    context.close()
        finally:
            server.terminate()
            server.wait(timeout=10)


if __name__ == "__main__":
    main()
