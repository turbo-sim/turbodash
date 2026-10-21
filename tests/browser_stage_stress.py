"""Real-browser stress checks for stage edits and turbine-type transactions."""

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
from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chromium", default="C:/Program Files/Google/Chrome/Application/chrome.exe")
    parser.add_argument("--output", type=Path, default=ROOT / "development/app_design/output_stress_stages")
    args = parser.parse_args()
    output = args.output.resolve()
    for folder in ("temp", "profile", "cache", "downloads"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    os.environ.update(TEMP=str(output / "temp"), TMP=str(output / "temp"), PYTHONDONTWRITEBYTECODE="1")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    address = f"http://127.0.0.1:{port}"
    code = f"from turbodash.apps.app_turbine_optimization.app import app; app.run(port={port}, debug=False)"
    with (output / "server.log").open("w", encoding="utf-8") as log:
        server = subprocess.Popen([sys.executable, "-c", code], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                  creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            deadline = time.monotonic() + 40
            while True:
                try:
                    urlopen(address, timeout=1).close()
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise RuntimeError("Browser server did not start")
                    time.sleep(.1)
            with sync_playwright() as playwright:
                context = playwright.chromium.launch_persistent_context(
                    str(output / "profile"), executable_path=args.chromium, headless=True,
                    viewport={"width": 1680, "height": 1080}, accept_downloads=True,
                    downloads_path=str(output / "downloads"),
                    args=["--disable-background-networking", "--no-first-run", "--disable-component-update",
                          "--disable-crash-reporter", f"--disk-cache-dir={output / 'cache'}"])
                try:
                    report = exercise(context, address, output)
                finally:
                    context.close()
            (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(report, indent=2))
        finally:
            server.terminate()
            server.wait(timeout=10)


def exercise(context, address, output):
    page = context.new_page()
    errors, results, callbacks, checks = [], [], [], []
    page.on("pageerror", lambda error: errors.append(str(error)))

    def capture(response):
        if "_dash-update-component" not in response.url:
            return
        request = response.request.post_data_json
        entry = {"status": response.status, "output": request["output"], "changed": request["changedPropIds"]}
        if response.status >= 400:
            errors.append(f"Callback HTTP {response.status}: {response.request.post_data[:150]}")
        if response.status == 200:
            try:
                body = response.json().get("response", {})
                entry["response_keys"] = list(body)
                if "result_store" in body:
                    results.append(body["result_store"]["data"])
            except Exception:
                pass
        callbacks.append(entry)

    page.on("response", capture)

    def open_section(identity):
        section = page.locator(f"#{identity}")
        if section.get_attribute("open") is None:
            section.locator(":scope > summary").click()

    def wait_result(stage_count, turbine_type):
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            current = accepted_result()
            if current and len(current["inputs"]["stages"]) == stage_count and current["inputs"]["turbine_type"] == turbine_type:
                return
            page.wait_for_timeout(100)
        raise AssertionError(f"Expected {stage_count} {turbine_type} stages; got {current['inputs'] if current else 'none'}; status: {page.locator('#td-calculation-status').inner_text()}")

    def accepted_result():
        return page.evaluate("""() => {
            const state = window.dash_stores[0].getState();
            return state.paths.strs.result_store.reduce((node, key) => node[key], state.layout).props.data;
        }""")

    def stage_field(key, stage=0):
        return page.locator(f'input[id*=\'"scope":"stage"\'][id*=\'"key":"{key}"\'][id*=\'"stage":{stage}\']')

    def search(query):
        page.keyboard.press("Control+k")
        page.locator("#td-parameter-search").fill(query)
        expect(page.locator(".td-search-result").first).to_be_visible()
        page.locator("#td-parameter-search").press("Enter")

    def wait_speed(mode, value, tolerance=1e-8):
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            current = accepted_result()
            rotation = current["inputs"]["rotational_speed"] if current else {}
            if isinstance(rotation, dict) and rotation.get("type") == mode and abs(rotation["value"] - value) <= tolerance * max(1, abs(value)):
                return
            page.wait_for_timeout(100)
        raise AssertionError(f"Speed did not converge to {mode}: {value}; last result {rotation}")

    try:
        page.goto(address)
        page.wait_for_selector("#perf_table .dash-table-container", timeout=90000)
        open_section("td-section-setup")
        page.locator("#stage_plus").click()
        expect(page.locator("#stage_count_label")).to_have_text("2", timeout=15000)
        wait_result(2, "axial")
        checks.append("Adding a stage updates both controls and results")
        page.get_by_label("Radial", exact=True).check()
        wait_result(2, "radial")
        checks.append("Switching to radial updates the two-stage result")
        page.locator("#stage_minus").click()
        expect(page.locator("#stage_count_label")).to_have_text("1")
        wait_result(1, "radial")
        page.get_by_label("Axial", exact=True).check()
        wait_result(1, "axial")
        checks.append("Removing a stage and restoring axial updates results")
        page.locator("#stage_plus").click(click_count=4, delay=25)
        expect(page.locator("#stage_count_label")).to_have_text("5", timeout=15000)
        wait_result(5, "axial")
        checks.append("Four rapid stage additions retain every click")
        page.locator("#stage_minus").click(click_count=3, delay=25)
        expect(page.locator("#stage_count_label")).to_have_text("2", timeout=15000)
        wait_result(2, "axial")
        page.locator("#stage_minus").click(click_count=4, delay=25)
        expect(page.locator("#stage_count_label")).to_have_text("1", timeout=15000)
        wait_result(1, "axial")
        # Browser events are individual user actions even when network responses
        # are pending; clamping at one must not consume later additions.
        for control in ("stage_minus", "stage_plus", "stage_plus", "stage_minus", "stage_plus"):
            page.locator(f"#{control}").click()
        expect(page.locator("#stage_count_label")).to_have_text("3", timeout=15000)
        wait_result(3, "axial")
        checks.append("Rapid removals and interleaved add/remove clamp at one without losing later additions")

        search("stage_1 degree of reaction")
        reaction = stage_field("degree_reaction")
        expect(reaction).to_be_focused()
        reaction.fill("0.4123456789")
        # Clicking the count control must commit the still-focused draft before
        # the stage editor is rebuilt; do not explicitly blur in the test.
        open_section("td-section-setup")
        page.locator("#stage_plus").click()
        expect(page.locator("#stage_count_label")).to_have_text("4", timeout=15000)
        expect(reaction).to_have_js_property("valueAsNumber", .4123456789)
        wait_result(4, "axial")
        assert accepted_result()["inputs"]["stages"][0]["degree_reaction"] == .4123456789
        checks.append("Adding stages retains a precise numeric draft without requiring a separate blur")

        # Load after many clicks resets the absolute target to the file, and
        # subsequent clicks operate from that loaded design.
        config = yaml.safe_load((ROOT / "demos/optimization/case_axial.yaml").read_text(encoding="utf-8"))
        config["inputs"]["stages"][0]["name"] = "High pressure"
        config["inputs"]["stages"].append(deepcopy(config["inputs"]["stages"][0]))
        config["inputs"]["stages"][1]["name"] = "Low pressure"
        page.locator('#load_button input[type="file"]').set_input_files({
            "name": "stage-stress.yaml", "mimeType": "text/yaml", "buffer": yaml.safe_dump(config).encode()})
        expect(page.locator("#stage_count_label")).to_have_text("2", timeout=15000)
        wait_result(2, "axial")
        open_section("td-section-setup")
        page.locator("#stage_plus").click()
        expect(page.locator("#stage_count_label")).to_have_text("3", timeout=15000)
        wait_result(3, "axial")
        assert [item["name"] for item in accepted_result()["inputs"]["stages"]] == ["High pressure", "Low pressure", "stage_3"]
        checks.append("Loading a named design resets the stage-count target and preserves stage names")

        for turbine_type in ("Radial", "Axial", "Radial", "Axial"):
            page.get_by_label(turbine_type, exact=True).check()
            wait_result(3, turbine_type.lower())
        checks.append("Repeated axial/radial switches refresh every stage and return to the final requested type")

        open_section("td-section-speed")
        speed = page.locator('input[id*=\'"key":"rotational_speed_value"\']')
        factor = accepted_result()["overall_performance"]["rotational_speed"] / accepted_result()["overall_performance"]["specific_speed"]
        precise_speed = .8123456789
        speed.fill(str(precise_speed))
        page.get_by_label("Actual", exact=True).check()
        expected_rpm = precise_speed * factor
        wait_speed("actual", expected_rpm)
        expect(page.locator("#td-speed-label")).to_have_text("Shaft speed [rpm]")
        assert abs(float(speed.input_value()) - expected_rpm) < 1e-6
        checks.append("Speed conversion preserves the latest unblurred value while a previous result may be pending")

        for mode in ("Specific", "Actual", "Specific", "Actual", "Specific"):
            page.get_by_label(mode, exact=True).check()
        wait_speed("specific", precise_speed)
        expect(page.locator("#td-speed-label")).to_have_text("Specific speed [−]")
        page.wait_for_timeout(700)
        assert abs(float(speed.input_value()) - precise_speed) < 1e-9
        assert abs(accepted_result()["overall_performance"]["rotational_speed"] - expected_rpm) < 1e-6
        checks.append("Rapid shaft-speed toggles preserve the canonical units and physical RPM")

        # A blank draft must remain visibly incomplete across structural edits,
        # then a repaired value must resume plotting for the displayed count.
        search("High pressure degree of reaction")
        reaction.fill("")
        open_section("td-section-setup")
        page.locator("#stage_plus").click()
        expect(page.locator("#stage_count_label")).to_have_text("4", timeout=15000)
        expect(reaction).to_have_value("")
        expect(page.locator("#td-calculation-status")).to_contain_text("finite", timeout=15000)
        search("High pressure degree of reaction")
        reaction.fill("0.456789123")
        reaction.blur()
        wait_result(4, "axial")
        expect(page.locator("#td-calculation-status")).to_have_text("")
        checks.append("Structural edits retain blank drafts and recover live plots after the invalid value is repaired")

        # Delay delivery of a structural response while another structural
        # control changes. Both intents must be present in the final design.
        page.evaluate("""() => {
            const fetch = window.fetch.bind(window);
            window.fetch = (...args) => {
                const body = args[1] && args[1].body;
                let delay = 0;
                if (window.tdTestDelayStructure && typeof body === 'string' &&
                    body.includes('td-stage-transaction.data')) {
                    delay = window.tdTestDelayStructure; window.tdTestDelayStructure = 0;
                }
                return fetch(...args).then(response => delay ?
                    new Promise(resolve => setTimeout(() => resolve(response), delay)) : response);
            };
        }""")
        open_section("td-section-setup")
        page.evaluate("window.tdTestDelayStructure = 350")
        page.locator("#stage_plus").click()
        page.wait_for_function("window.tdTestDelayStructure === 0")
        page.get_by_label("Radial", exact=True).check()
        expect(page.locator("#stage_count_label")).to_have_text("5", timeout=15000)
        wait_result(5, "radial")
        page.wait_for_timeout(500)
        assert all(stage["radius_ratio_12"] == .85 for stage in accepted_result()["inputs"]["stages"])
        page.evaluate("window.tdTestDelayStructure = 350")
        page.get_by_label("Axial", exact=True).check()
        page.wait_for_function("window.tdTestDelayStructure === 0")
        page.locator("#stage_minus").click()
        expect(page.locator("#stage_count_label")).to_have_text("4", timeout=15000)
        wait_result(4, "axial")
        page.wait_for_timeout(500)
        assert all(stage["radius_ratio_12"] == 1 for stage in accepted_result()["inputs"]["stages"])
        checks.append("Interleaved stage counts and turbine types preserve both intents under delayed responses")

        page.evaluate("window.tdTestDelayStructure = 1000")
        page.locator("#stage_plus").click()
        page.wait_for_function("window.tdTestDelayStructure === 0")
        search("High pressure degree of reaction")
        reaction.fill("0.42345678901")
        reaction.blur()
        expect(page.locator("#stage_count_label")).to_have_text("5", timeout=15000)
        expect(reaction).to_have_js_property("valueAsNumber", .42345678901)
        wait_result(5, "axial")
        assert accepted_result()["inputs"]["stages"][0]["degree_reaction"] == .42345678901
        checks.append("A stage value committed while Add is pending survives the delayed editor rebuild")

        config["inputs"]["stages"][0]["name"] = "New high pressure"
        config["inputs"]["stages"][1]["name"] = "New low pressure"
        config["inputs"]["stages"][1]["degree_reaction"] = .4712345678
        page.evaluate("window.tdTestDelayStructure = 1000")
        page.locator('#load_button input[type="file"]').set_input_files({
            "name": "pending-stage-load.yaml", "mimeType": "text/yaml", "buffer": yaml.safe_dump(config).encode()})
        page.wait_for_function("window.tdTestDelayStructure === 0")
        open_section("td-section-setup")
        page.locator("#stage_plus").click()
        expect(page.locator("#stage_count_label")).to_have_text("3", timeout=15000)
        wait_result(3, "axial")
        page.wait_for_timeout(1200)
        loaded = accepted_result()["inputs"]["stages"]
        assert [item["name"] for item in loaded] == ["New high pressure", "New low pressure", "stage_3"]
        assert loaded[2]["degree_reaction"] == .4712345678
        checks.append("Adding during a pending YAML load uses the new file's stages and keeps the added stage")
        assert not errors, errors
        return {"checks": checks, "errors": errors, "results": len(results)}
    finally:
        (output / "callbacks.json").write_text(json.dumps(callbacks, indent=2), encoding="utf-8")
        page.screenshot(path=output / "stage_stress.png", full_page=True)


if __name__ == "__main__":
    main()
