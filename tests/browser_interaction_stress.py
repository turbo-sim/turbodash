"""Optional real-browser stress checks for model/figure synchronization.

Run with the repository Python and Playwright installed. The test starts and
stops its own server; every browser artifact stays inside the repository.
"""

import argparse
import base64
import json
from browser_picker_helpers import choose_path
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

import numpy as np
import yaml
from playwright.sync_api import expect, sync_playwright


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chromium", default="C:/Program Files/Google/Chrome/Application/chrome.exe")
    parser.add_argument("--output", type=Path, default=ROOT / "development/app_design/output_stress_interaction")
    args = parser.parse_args()
    args.output = args.output.resolve()
    for folder in ("temp", "profile", "downloads", "cache"):
        (args.output / folder).mkdir(parents=True, exist_ok=True)
    os.environ.update(TEMP=str(args.output / "temp"), TMP=str(args.output / "temp"), PYTHONDONTWRITEBYTECODE="1")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    address = f"http://127.0.0.1:{port}"
    code = f"from turbodash.apps.app_turbine_optimization.app import app; app.run(port={port}, debug=False)"
    with (args.output / "server.log").open("w", encoding="utf-8") as log:
        server = subprocess.Popen([sys.executable, "-B", "-c", code], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                  creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            deadline = time.monotonic() + 60
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
                    str(args.output / "profile"), executable_path=args.chromium, headless=True,
                    viewport={"width": 1700, "height": 1100},
                    downloads_path=str(args.output / "downloads"), accept_downloads=True,
                    args=["--disable-background-networking", "--no-first-run", "--disable-component-update",
                          "--disable-crash-reporter", f"--disk-cache-dir={args.output / 'cache'}"],
                )
                try:
                    report = exercise(context, address, args.output)
                finally:
                    context.close()
            (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            summary = dict(report, failures=[{key: failure[key] for key in ("check", "error")}
                                             for failure in report["failures"]])
            print(json.dumps(summary, indent=2))
            assert not report["failures"], summary["failures"]
        finally:
            server.terminate()
            server.wait(timeout=10)


def exercise(context, address, output):
    page = context.new_page()
    errors, bodies, checks, failures, exchanges = [], [], [], [], []
    request_order = {}
    page.on("pageerror", lambda error: errors.append(str(error)))

    def remember_request(request):
        if "_dash-update-component" in request.url:
            request_order[request] = len(request_order) + 1

    page.on("request", remember_request)

    def flatten(items):
        for item in items:
            if isinstance(item, list):
                yield from flatten(item)
            else:
                yield item

    def capture(response):
        if "_dash-update-component" not in response.url:
            return
        if response.status >= 400:
            errors.append(f"Callback HTTP {response.status}: {response.request.post_data[:150]}")
        if response.status == 200:
            try:
                body = response.json().get("response", {})
            except Exception:
                return
            if "result_store" in body:
                bodies.append(body)
                payload = response.request.post_data_json
                inputs = {}
                for item in flatten(payload["inputs"]):
                    identity = item["id"]
                    if isinstance(identity, str):
                        try:
                            identity = json.loads(identity)
                        except ValueError:
                            continue
                    if isinstance(identity, dict) and identity.get("scope") == "overall":
                        inputs[identity["key"]] = item["value"]
                exchanges.append({
                    "request_order": request_order.get(response.request),
                    "changed": payload["changedPropIds"],
                    "request_overall": inputs,
                    "result_inputs": body["result_store"]["data"]["inputs"],
                    "result_signature": body.get("result_input_signature", {}).get("data"),
                })

    page.on("response", capture)
    page.goto(address)
    page.wait_for_selector("#perf_table .dash-table-container", timeout=90000)

    def accepted_result():
        # Network completion order is not Dash's accepted-result order. Read
        # the renderer's current stores without modifying any application state.
        return page.evaluate("""() => {
            const state = window.dash_stores[0].getState();
            const props = id => {
                const path = state.paths.strs[id];
                return path ? path.reduce((node, key) => node[key], state.layout).props : {};
            };
            return {inputs: (props('result_store').data || {}).inputs,
                    signature: props('result_input_signature').data};
        }""")

    def accepted_snapshot():
        accepted = accepted_result()
        for body in reversed(bodies):
            if body.get("result_input_signature", {}).get("data") == accepted["signature"]:
                return body
        raise AssertionError(f"No captured figure matches accepted signature {accepted['signature']}")

    def diagnostic_state():
        return page.evaluate("""() => {
            const state = window.dash_stores[0].getState();
            const props = id => {
                const path = state.paths.strs[id];
                return path ? path.reduce((node, key) => node[key], state.layout).props : {};
            };
            return {
                controls: Array.from(document.querySelectorAll('input[id*="overall"]')).map(input => ({id:input.id, value:input.value})),
                accepted_inputs: (props('result_store').data || {}).inputs,
                accepted_signature: props('result_input_signature').data,
                accepted_first_x: ((props('meridional_plot').figure || {}).data || [{}])[0].x,
                plotted_first_x: Array.from(document.querySelector('#meridional_plot .js-plotly-plot').data[0].x || [])
            };
        }""")

    def run(name, action):
        try:
            action()
            checks.append(name)
        except Exception as error:
            failures.append({"check": name, "error": str(error), "status": page.locator("#td-calculation-status").inner_text(),
                             "client": diagnostic_state(), "recent_exchanges": exchanges[-5:]})
            page.screenshot(path=output / f"failure_{len(failures)}.png", full_page=True)
        print(json.dumps({"check": name, "passed": name in checks}), flush=True)

    def field(key, stage=None):
        scope = "overall" if stage is None else "stage"
        selector = f'input[id*=\'"scope":"{scope}"\'][id*=\'"key":"{key}"\']'
        if stage is not None:
            selector += f'[id*=\'"stage":{stage}\']'
        return page.locator(selector)

    def reveal(key, stage=None):
        control = field(key, stage)
        control.evaluate("el => { for (let p = el.parentElement; p; p = p.parentElement) if(p.tagName === 'DETAILS') p.open = true; }")
        return control

    def wait_result(key, value, stage=None):
        deadline = time.monotonic() + 20
        actual = None
        while time.monotonic() < deadline:
            inputs = accepted_result().get("inputs")
            if inputs:
                actual = inputs.get(key) if stage is None else inputs["stages"][stage].get(key)
                if actual == value:
                    page.wait_for_timeout(400)
                    inputs = accepted_result()["inputs"]
                    actual = inputs.get(key) if stage is None else inputs["stages"][stage].get(key)
                    if actual == value:
                        return
            page.wait_for_timeout(100)
        raise AssertionError(f"Accepted result never reached {key}={value}; latest={actual!r}")

    def assert_accepted(**values):
        inputs = accepted_result()["inputs"]
        for key, value in values.items():
            assert inputs[key] == value, f"Accepted {key} reverted to {inputs[key]!r}; expected {value!r}"

    def array(value):
        if isinstance(value, dict) and "bdata" in value:
            return np.frombuffer(base64.b64decode(value["bdata"]), dtype=value["dtype"])
        return np.asarray(value)

    def compare_figures(snapshot=None):
        assert bodies, "No result callbacks captured"
        snapshot = snapshot or accepted_snapshot()
        for identity in ("meridional_plot", "blades_plot"):
            expected = snapshot[identity]["figure"]["data"]
            actual = page.locator(f"#{identity} .js-plotly-plot").evaluate("el => el.data.map(t => ({x:Array.from(t.x || []), y:Array.from(t.y || [])}))")
            assert len(expected) == len(actual), f"{identity}: trace count {len(actual)} != {len(expected)}"
            for index, (left, right) in enumerate(zip(expected, actual)):
                for axis in ("x", "y"):
                    np.testing.assert_allclose(right[axis], array(left.get(axis, [])), equal_nan=True,
                                               err_msg=f"{identity} trace {index} {axis}")

    def ordinary_change():
        control = reveal("mass_flow_rate")
        control.fill("12.34567")
        control.press("Tab")
        wait_result("mass_flow_rate", 12.34567)
        compare_figures()
        expect(control).to_have_value("12.34567")

    run("Ordinary numeric edit updates result and both real Plotly coordinate arrays", ordinary_change)

    def rapid_changes():
        mass = reveal("mass_flow_rate")
        ratio = reveal("blade_velocity_ratio")
        for index in range(8):
            mass.fill(str(10 + index * .4))
            mass.press("Tab")
            ratio.fill(str(.58 + index * .005))
            ratio.press("Tab")
            page.locator(f'.td-theme-toggle [data-theme="{"dark" if index % 2 == 0 else "light"}"]').click()
            workspace = "geometry" if index % 2 == 0 else "meanline"
            page.locator(f'.td-workspace-rail [data-workspace="{workspace}"]').click()
        page.locator(".td-brand").click()
        wait_result("mass_flow_rate", 12.8)
        wait_result("blade_velocity_ratio", .615)
        assert float(mass.input_value()) == 12.8
        assert float(ratio.input_value()) == .615
        assert_accepted(mass_flow_rate=12.8, blade_velocity_ratio=.615)
        compare_figures()
        assert page.locator("#td-calculation-status").inner_text() == ""

    run("Rapid model edits interleaved with theme and workspace changes keep graph coordinates synchronized", rapid_changes)

    def keyboard_change():
        mass = reveal("mass_flow_rate")
        mass.click()
        mass.press("Control+a")
        mass.press_sequentially("13.7654321", delay=20)
        mass.press("Enter")
        wait_result("mass_flow_rate", 13.7654321)
        compare_figures()

    run("Keyboard text entry and Enter commit exact floating point values", keyboard_change)

    def invalid_then_valid():
        mass = reveal("mass_flow_rate")
        before = len(bodies)
        mass.fill("")
        mass.press("Tab")
        expect(page.locator("#td-calculation-status")).to_contain_text("last valid design")
        assert len(bodies) == before
        mass.fill("14.25")
        mass.press("Tab")
        wait_result("mass_flow_rate", 14.25)
        compare_figures()
        expect(mass).to_have_value("14.250")

    run("Blank draft retains valid plots and correcting it resumes updates", invalid_then_valid)

    def save_during_edit():
        mass = reveal("mass_flow_rate")
        mass.fill("15.987654")
        mass.press("Tab")
        wait_result("mass_flow_rate", 15.987654)
        with page.expect_download(timeout=20000) as download:
            page.locator("#save_button").click()
        target = output / "downloads" / "interaction-save.yaml"
        download.value.save_as(target)
        saved = yaml.safe_load(target.read_text(encoding="utf-8"))
        assert saved["inputs"]["mass_flow_rate"] == float(mass.input_value()) == 15.987654
        assert saved["inputs"]["blade_velocity_ratio"] == float(field("blade_velocity_ratio").input_value())

    run("Saved design matches final visible precise control values", save_during_edit)

    def delayed_response():
        # Delayed network delivery emulates a slow calculation/connection while
        # the user continues editing. Keep a real server calculation throughout.
        page.evaluate("""() => {
            const fetch = window.fetch.bind(window);
            window.fetch = (...args) => {
                const body = args[1] && args[1].body;
                let delay = 0;
                if (window.tdTestDelayNextSolve && typeof body === 'string' &&
                    body.includes('result_input_signature.data')) {
                    delay = window.tdTestDelayNextSolve;
                    window.tdTestDelayNextSolve = 0;
                }
                // Delay the first request dispatch too, so the older network
                // body reliably arrives after the newer accepted result.
                const request = delay ? new Promise(resolve => setTimeout(() => resolve(fetch(...args)), 700)) : fetch(...args);
                return request.then(response => delay ?
                    new Promise(resolve => setTimeout(() => resolve(response), delay)) : response);
            };
            window.tdTestDelayNextSolve = 2500;
        }""")
        mass = reveal("mass_flow_rate")
        mass.fill("16.125")
        mass.press("Tab")
        page.wait_for_function("window.tdTestDelayNextSolve === 0")
        mass.fill("17.875")
        mass.press("Tab")
        wait_result("mass_flow_rate", 17.875)
        page.wait_for_timeout(3000)
        assert_accepted(mass_flow_rate=17.875)
        assert float(mass.input_value()) == 17.875
        compare_figures()
        assert page.locator("#td-calculation-status").inner_text() == ""

    run("Out-of-order valid calculation responses retain the latest design figures", delayed_response)

    def delayed_invalid_response():
        page.evaluate("window.tdTestDelayNextSolve = 2500")
        mass = reveal("mass_flow_rate")
        mass.fill("")
        mass.press("Tab")
        page.wait_for_function("window.tdTestDelayNextSolve === 0")
        mass.fill("18.125")
        mass.press("Tab")
        wait_result("mass_flow_rate", 18.125)
        page.wait_for_timeout(3000)
        assert_accepted(mass_flow_rate=18.125)
        compare_figures()
        assert page.locator("#td-calculation-status").inner_text() == "", "A delayed invalid draft must not overwrite the latest valid status"
        assert "last" not in page.locator("#td-results-label").inner_text().lower()

    run("Out-of-order invalid draft responses cannot mark a corrected design stale", delayed_invalid_response)

    def delayed_success_status():
        mass = reveal("mass_flow_rate")
        mass.fill("18.25")
        mass.press("Tab")
        wait_result("mass_flow_rate", 18.25)
        retained = accepted_snapshot()
        page.evaluate("window.tdTestDelayNextSolve = 2500")
        mass.fill("19.125")
        mass.press("Tab")
        page.wait_for_function("window.tdTestDelayNextSolve === 0")
        mass.fill("")
        mass.press("Tab")
        expect(page.locator("#td-calculation-status")).to_contain_text("last valid design")
        page.wait_for_timeout(3000)
        try:
            expect(page.locator("#td-calculation-status")).to_contain_text("last valid design")
            assert "last" in page.locator("#td-results-label").inner_text().lower()
            assert_accepted(mass_flow_rate=18.25)
            compare_figures(retained)
        finally:
            mass.fill("19.5")
            mass.press("Tab")
            wait_result("mass_flow_rate", 19.5)

    run("Out-of-order successful responses cannot hide the current invalid-draft warning", delayed_success_status)

    def pair_change_during_solve():
        pressure = reveal("inlet_property_1")
        page.evaluate("window.tdTestDelayNextSolve = 2500")
        pressure.fill("550123.456789")
        pressure.press("Tab")
        page.wait_for_function("window.tdTestDelayNextSolve === 0")
        pair = page.locator("#td-inlet-pair")
        pair.click()
        pair.locator("input").fill("HmassP")
        pair.locator("input").press("Enter")
        expect(page.locator("#td-inlet-label-1")).to_contain_text("enthalpy")
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if float(field("inlet_property_2").input_value()) == 550123.456789:
                break
            page.wait_for_timeout(100)
        assert float(field("inlet_property_2").input_value()) == 550123.456789, "Changing property pair discarded a recent valid pressure edit"
        wait_result("inlet_property_pair", "HmassP_INPUTS")
        wait_result("inlet_property_2", 550123.456789)
        page.wait_for_timeout(3000)
        assert_accepted(inlet_property_pair="HmassP_INPUTS", inlet_property_2=550123.456789)
        compare_figures()

    run("Property-pair change during an in-flight calculation preserves the current inlet state", pair_change_during_solve)

    def invalid_pair_change():
        first = reveal("inlet_property_1")
        first.fill("")
        first.press("Tab")
        expect(page.locator("#td-calculation-status")).to_contain_text("last valid design")
        pair = page.locator("#td-inlet-pair")
        pair.click()
        pair.locator("input").fill("PT")
        pair.locator("input").press("Enter")
        expect(page.locator("#td-inlet-label-1")).to_have_text("Inlet pressure [Pa]")
        expect(first).to_have_value("")
        expect(field("inlet_property_2")).to_have_value("")
        expect(page.locator("#td-fluid-error")).to_contain_text("Enter both inlet values")

    run("Changing property pair from an invalid inlet requests both new values", invalid_pair_change)

    def add_during_editor_load():
        def open_variables():
            section = page.locator("#opt-panel")
            if section.get_attribute("open") is None:
                section.locator(":scope > summary").click()
            page.get_by_role("tab", name="Design variables", exact=True).click()

        open_variables()
        config = yaml.safe_load((ROOT / "development/optimization/case_axial.yaml").read_text(encoding="utf-8"))
        config["optimization"]["design_variables"] = {
            "blade_velocity_ratio": {"lower_value": .5, "upper_value": .75, "initial_value": .6}
        }
        config["optimization"]["constraints"] = []
        page.evaluate("""() => {
            const fetch = window.fetch.bind(window);
            window.tdTestEditorRequests = [];
            window.tdTestEditorLoadPending = false;
            window.fetch = (...args) => {
                const body = args[1] && args[1].body;
                let delay = 0;
                if (typeof body === 'string' && body.includes('opt-editor-state.data')) {
                    const payload = JSON.parse(body);
                    window.tdTestEditorRequests.push(payload);
                    if (payload.changedPropIds.some(id => ['loaded_cfg_store.data', 'opt-editor-load.data'].includes(id))) {
                        window.tdTestEditorLoadPending = true;
                        delay = 350;
                    }
                }
                return fetch(...args).then(response => delay ?
                    new Promise(resolve => setTimeout(() => resolve(response), delay)) : response);
            };
        }""")
        page.locator('#load_button input[type="file"]').set_input_files({
            "name": "pending-editor-load.yaml", "mimeType": "text/yaml",
            "buffer": yaml.safe_dump(config).encode(),
        })
        page.wait_for_function("window.tdTestEditorLoadPending")
        open_variables()
        choose_path(page, "variable", "rotational_speed")
        page.wait_for_function("window.tdTestEditorRequests.some(p => p.changedPropIds.includes('opt-selection-request.data'))")
        requests = page.evaluate("window.tdTestEditorRequests")
        clicked = [request for request in requests if "opt-selection-request.data" in request["changedPropIds"]]

        selection = next(item["value"] for item in flatten(clicked[-1]["inputs"]) if item["id"] == "opt-selection-request")
        assert any(operation.get("path") == "rotational_speed" and operation["selected"]
                   for operation in selection["operations"]), "The checkbox action must reach the editor callback"
        expect(page.locator("#opt-variable-rows .td-opt-row")).to_have_count(2, timeout=15000)
        wait_result("inlet_property_pair", "PT_INPUTS")
        compare_figures()

    run("Add during a pending YAML editor load retains the loaded row and the new row", add_during_editor_load)
    compare_figures()
    if errors:
        failures.append({"check": "Browser and callback errors", "error": errors})
    (output / "exchanges.json").write_text(json.dumps(exchanges, indent=2), encoding="utf-8")
    return {"checks": checks, "failures": failures, "browser_errors": errors, "result_updates": len(bodies)}


if __name__ == "__main__":
    main()
