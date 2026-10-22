"""Exercise the integrated modeller in Chrome against a running workbench.

Run: python tests/browser_blade_modeller.py --url http://127.0.0.1:8067
Artifacts are written to development/blade_parametrization/output_integration.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[1]


def check_geometry_controls(page, output):
    """Verify workspace isolation, scoped search, help, and slider endpoints."""
    from playwright.sync_api import expect

    expect(page.locator("#td-inputs-title")).to_have_text("Geometry inputs")
    expect(page.locator("#geometry_stage .Select-value-label")).to_have_count(1, timeout=30000)
    assert page.locator('[data-controls-workspace="meanline"] input:visible').count() == 0
    headings = page.locator("#blade_modeller_controls > details > summary").all_text_contents()
    assert headings == ["01Meridional channel", "02Camberline and turning", "03Thickness distribution", "04Flow domain"]
    page.screenshot(path=str(output / "geometry_controls_collapsed.png"))
    search = page.locator("#td-parameter-search")
    search.fill("inlet pressure")
    expect(page.locator("#td-search-results")).to_contain_text("No matching parameters")
    search.fill("relief")
    expect(page.locator(".td-search-result")).to_have_count(1)
    page.locator(".td-search-result").click()
    expect(page.locator("#blade-help-leading_edge-relief_strength")).to_be_visible()
    page.locator("#blade-help-leading_edge-relief_strength").hover()
    expect(page.locator(".tooltip-inner")).to_contain_text("Remove some thickness")
    page.mouse.move(900, 100)
    search.fill("base sampling")
    page.locator(".td-search-result").click()
    expect(page.locator("#blade-help-numerics-number_of_points")).to_be_visible()
    search.fill("turning concentration")
    page.locator(".td-search-result").click()
    slider = page.locator('[data-parameter-aliases="camberline.turning_concentration"] .td-blade-slider')
    expect(slider.locator(".rc-slider-mark-text")).to_have_text(["1", "3"])
    page.locator("#td-collapse-all").click()
    page.locator("#blade-section-3 > summary").click()
    page.screenshot(path=str(output / "geometry_controls_thickness.png"))
    page.locator('.td-workspace-rail [data-workspace="meanline"]').click()
    expect(page.locator("#td-inputs-title")).to_have_text("Meanline inputs")
    search.fill("turning concentration")
    expect(page.locator("#td-search-results")).to_contain_text("No matching parameters")
    page.locator('.td-workspace-rail [data-workspace="mesh"]').click()
    expect(page.locator("#td-inputs-title")).to_have_text("Mesh inputs")
    assert page.locator('[data-controls-workspace="meanline"] input:visible').count() == 0
    page.locator('.td-workspace-rail [data-workspace="geometry"]').click()
    search.fill("")
    search.press("Escape")
    print("PASS geometry sections, workspace isolation, scoped search, tooltips and endpoints", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8067")
    args = parser.parse_args()
    output = ROOT / "development/blade_parametrization/output_integration"
    for name in ("temp", "profile", "downloads"):
        (output / name).mkdir(parents=True, exist_ok=True)
    os.environ.update(TEMP=str(output / "temp"), TMP=str(output / "temp"))
    from playwright.sync_api import Error, sync_playwright

    observed, errors, calls, checks = {}, [], [], []
    closing = False
    passed = False
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch_persistent_context(
            str(output / "profile"), headless=True,
            executable_path="C:/Program Files/Google/Chrome/Application/chrome.exe",
            viewport={"width": 1680, "height": 1080}, accept_downloads=True,
            args=["--disable-background-networking", "--no-first-run", "--disable-crash-reporter"],
        )
        try:
            page = browser.new_page()
            page.on("pageerror", lambda error: errors.append(str(error)))

            def capture(response):
                if "_dash-update-component" not in response.url:
                    return
                request = response.request.post_data_json
                request["response_status"] = response.status
                calls.append(request)
                if response.status >= 400:
                    errors.append(f"Callback HTTP {response.status}: {request.get('output')}")
                if response.status == 200:
                    try:
                        payload = response.json()
                    except Error:
                        if closing:
                            return
                        raise
                    for identity, props in payload.get("response", {}).items():
                        observed.setdefault(identity, {}).update(props)

            page.on("response", capture)

            def wait(predicate, label, timeout=60):
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    page.wait_for_timeout(100)
                    if predicate():
                        print(f"PASS {label}", flush=True)
                        return
                raise AssertionError(f"Timed out: {label}; errors={errors}; notice={observed.get('blade_modeller_error')}")

            def store_data(identity):
                # Responses may arrive after Dash has superseded them. Read
                # the accepted browser store, not the last network response.
                return page.evaluate("""id => {
                    const state = window.store?.getState();
                    const path = state?.paths?.strs?.[id];
                    if (!path) return {};
                    const component = path.reduce((node, key) => node[key], state.layout);
                    return component.props.data || {};
                }""", identity)

            def state():
                return store_data("geometry_state")

            def committed():
                return store_data("geometry_committed")

            def row(index=0):
                return state()["rows"][index]["parameters"]

            def solver_calls():
                return sum("td-calculation-state.data" in item.get("output", "") for item in calls)

            def field(path):
                control = page.locator('input[id=' + json.dumps(json.dumps({"path": path, "type": "blade-parameter"}, separators=(",", ":"))) + ']')
                control.evaluate("node => { for (let p=node.parentElement; p; p=p.parentElement) if(p.tagName==='DETAILS') p.open=true; }")
                return control

            page.goto(args.url)
            wait(lambda: len(state().get("rows", [])) == 2, "initial geometry baseline")
            page.wait_for_function("document.querySelector('#blades_plot .js-plotly-plot')?.data?.some(t => t.x?.length > 600)", timeout=60000)
            initial_outline = page.locator("#blades_plot .js-plotly-plot").evaluate(
                "graph => JSON.stringify(graph.data.filter(t => t.x?.length > 600).map(t => [t.x,t.y]))")
            page.locator('.td-workspace-rail [data-workspace="geometry"]').click()
            check_geometry_controls(page, output)
            page.wait_for_function("document.querySelector('#geometry_profile .js-plotly-plot')?.data?.some(t => t.x?.length > 600)", timeout=60000)
            page.locator("#blade_modeller_preview").scroll_into_view_if_needed()
            page.wait_for_function("document.querySelector('#geometry-scene canvas') && window.dash_clientside.studio?.scenes.geometry?.actors.some(a => a.props.actor.visibility && a.props.children[0].props.points.length > 1800)", timeout=60000)
            assert page.locator("#blade_modeller_preview .js-plotly-plot").count() == 0
            vtk_mount = page.evaluate("window.dash_clientside.studio.roots.geometry.id")
            checks.append("rendered high-resolution 2D traces and shared TurboMoC VTK 3D surfaces")
            initial_epoch = state()["_epoch"]
            initial_solves = solver_calls()
            initial_geometry = deepcopy(state())

            def vtk_control(kind, **extra):
                identifier = {"type": "vtk-row-" + kind, "pane": "geometry", "row": "stage_1_stator", **extra}
                return page.locator('[id=' + json.dumps(json.dumps(identifier, sort_keys=True, separators=(",", ":"))) + ']')

            def visible_blades():
                return page.evaluate("window.dash_clientside.studio.scenes.geometry.actors.filter(a => a.props.actor.visibility && a.props.id.endsWith('-blade')).length")

            page.wait_for_function("window.dash_clientside.studio.scenes.geometry.fitted", timeout=60000)
            page.wait_for_timeout(500)
            geometry_key = page.evaluate("window.dash_clientside.studio.scenes.geometry.dataKey")
            fit_before = page.evaluate("window.dash_clientside.studio.scenes.geometry.viewProps.triggerResetCamera")
            count = vtk_control("copies")
            assert int(count.get_attribute("max")) == row()["source"]["cascade"]["blade_count"]
            count.fill("2")
            count.press("Tab")
            wait(lambda: visible_blades() == 2, "VTK displayed blade count")
            copy_slider = vtk_control("slider").locator('[role="slider"]')
            wait(lambda: copy_slider.get_attribute("aria-valuenow") == "2", "VTK count reflected by slider")
            copy_slider.press("ArrowRight")
            copy_slider.press("Tab")
            wait(lambda: visible_blades() == 3 and int(count.input_value()) == 3, "VTK copy slider and number synchronization")
            picker = vtk_control("picker", surface="blade").locator('input[type="color"]')
            picker.evaluate("node => { node.value = '#12ab34'; node.dispatchEvent(new Event('input', {bubbles: true})); }")
            page.wait_for_function("window.dash_clientside.studio.scenes.geometry.actors.some(a => a.props.actor.visibility && a.props.id.endsWith('-blade') && Math.abs(a.props.property.color[0]-18/255) < 1e-8)", timeout=60000)
            vtk_control("preset", action="surfaces").click()
            page.wait_for_function("window.dash_clientside.studio.scenes.geometry.actors.filter(a => a.props.actor.visibility).length === 21", timeout=60000)
            vtk_control("surface", surface="shroud").locator("input").uncheck()
            page.wait_for_function("!window.dash_clientside.studio.scenes.geometry.actors.some(a => a.props.actor.visibility && a.props.id.endsWith('-shroud'))", timeout=60000)
            vtk_control("preset", action="blade").click()
            page.wait_for_function("window.dash_clientside.studio.scenes.geometry.actors.filter(a => a.props.actor.visibility).length === 3", timeout=60000)
            assert page.evaluate("window.dash_clientside.studio.scenes.geometry.dataKey") == geometry_key
            assert page.evaluate("window.dash_clientside.studio.scenes.geometry.viewProps.triggerResetCamera") == fit_before
            canvas = page.locator("#geometry-scene canvas")
            canvas.scroll_into_view_if_needed()
            before_camera = canvas.screenshot()
            rectangle = canvas.bounding_box()
            center_x, center_y = rectangle["x"] + rectangle["width"] / 2, rectangle["y"] + rectangle["height"] / 2
            page.mouse.move(center_x, center_y)
            page.mouse.down()
            page.mouse.move(center_x + 60, center_y + 25, steps=12)
            page.mouse.up()
            assert canvas.screenshot() != before_camera
            page.locator("#geometry-vtk-fit").click()
            wait(lambda: page.evaluate("window.dash_clientside.studio.scenes.geometry.viewProps.triggerResetCamera") != fit_before, "VTK rotate and fit view")
            vtk_control("preset", action="reset").click()
            page.wait_for_function("window.dash_clientside.studio.scenes.geometry.actors.filter(a => a.props.actor.visibility).length === 6", timeout=60000)
            assert state() == initial_geometry and solver_calls() == initial_solves
            assert page.evaluate("window.dash_clientside.studio.roots.geometry.id") == vtk_mount
            checks.append("TurboMoC VTK copies, slider, surface colors/visibility/presets, rotate and fit without geometry changes")
            precise = 1.61234567890123
            concentration = field("camberline.turning_concentration")
            concentration.fill(str(precise))
            concentration.press("Tab")
            wait(lambda: row()["overrides"].get("camberline.turning_concentration") == precise, "precise typed edit")
            wait(lambda: committed().get("revision") == state().get("revision"), "typed 3D commit")
            assert float(concentration.input_value()) == precise
            checks.append("precise numeric input and committed geometry")

            # Merely focusing a slider must not quantize the precise number.
            slider = page.locator('[id=' + json.dumps(json.dumps({"path": "camberline.turning_concentration", "type": "blade-slider"}, separators=(",", ":"))) + '] [role="slider"]')
            slider.focus()
            slider.press("Tab")
            field("thickness.trailing_edge_fullness").fill("-0.123456789")
            field("thickness.trailing_edge_fullness").press("Tab")
            wait(lambda: row()["overrides"].get("thickness.trailing_edge_fullness") == -0.123456789, "next precise edit")
            assert row()["overrides"]["camberline.turning_concentration"] == precise
            checks.append("slider focus preserves typed precision and the next edit")

            saved = deepcopy(state())
            for tab in ("meanline", "geometry", "mesh", "geometry"):
                page.locator(f'.td-workspace-rail [data-workspace="{tab}"]').click()
                page.wait_for_timeout(250)
            assert state() == saved
            assert solver_calls() == initial_solves
            assert page.evaluate("window.dash_clientside.studio.roots.geometry.id") == vtk_mount
            checks.append("tabs preserve geometry without a meanline solve")

            # Hold a real pointer drag long enough for multiple lightweight updates.
            def committed_vtk_ready():
                proposal = observed.get("blade_modeller_3d_proposal", {}).get("data", {})
                return (proposal.get("token", [None] * 4)[1:3] == [committed().get("_epoch"), committed().get("revision")]
                        and page.evaluate("window.dash_clientside.studio.scenes.geometry.dataKey") == proposal.get("data", {}).get("key"))

            wait(committed_vtk_ready, "previous committed VTK geometry rendered")
            slider.scroll_into_view_if_needed()
            bounds = slider.bounding_box()
            x, y = bounds["x"] + bounds["width"] / 2, bounds["y"] + bounds["height"] / 2
            before_revision, committed_revision = state()["revision"], committed()["revision"]
            before_vtk = page.evaluate("window.dash_clientside.studio.scenes.geometry.dataKey")
            page.mouse.move(x, y)
            page.mouse.down()
            page.mouse.move(x + 15, y, steps=8)
            wait(lambda: state()["revision"] > before_revision, "live slider update")
            assert committed()["revision"] == committed_revision
            assert page.evaluate("window.dash_clientside.studio.scenes.geometry.dataKey") == before_vtk
            page.mouse.move(x + 28, y, steps=8)
            page.wait_for_timeout(350)
            page.mouse.up()
            wait(lambda: committed()["revision"] == state()["revision"], "3D refresh after release")
            assert solver_calls() == initial_solves
            checks.append("live 2D drag and 3D deferred until release")

            # Invalid fullness combinations retain the valid state.
            field("thickness.leading_edge_fullness").fill("0.5")
            field("thickness.leading_edge_fullness").press("Tab")
            wait(lambda: row()["overrides"].get("thickness.leading_edge_fullness") == .5, "valid LE fullness")
            valid_state = deepcopy(state())
            field("thickness.trailing_edge_fullness").fill("-0.5")
            field("thickness.trailing_edge_fullness").press("Tab")
            wait(lambda: "Edit not applied" in page.locator("#blade_modeller_error").inner_text(), "inadmissible pair diagnostic")
            assert state()["rows"] == valid_state["rows"] and state()["_epoch"] == valid_state["_epoch"]
            valid_state = deepcopy(state())
            checks.append("invalid rational-fullness pair preserves geometry")

            # Separate row ownership, YAML round trip and geometric crosscuts.
            page.locator("#geometry_row label").filter(has_text="Rotor").click()
            wait(lambda: page.locator("#blade_modeller_context").get_attribute("data-row") == "stage_01/rotor", "rotor selection")
            field("camberline.turning_concentration").fill("1.5123456789")
            # Opening the dialog includes the numeric edit blurred by its button.
            page.locator("#blade_yaml_open").click()
            page.locator("#blade_yaml_text").wait_for(state="visible")
            with page.expect_download() as download_info:
                page.locator("#blade_modeller_save").click()
            wait(lambda: row(1)["overrides"].get("camberline.turning_concentration") == 1.5123456789, "independent rotor edit")
            assert row() == valid_state["rows"][0]["parameters"]
            import yaml
            saved_row = yaml.safe_load(Path(download_info.value.path()).read_text(encoding="utf-8"))
            assert saved_row["overrides"]["camberline.turning_concentration"] == 1.5123456789
            yaml_before = deepcopy(state()["rows"])
            page.locator("#blade_yaml_text").fill("source: [broken")
            page.locator("#blade_yaml_apply").click()
            wait(lambda: "Could not import" in page.locator("#blade_yaml_feedback").inner_text(), "invalid YAML explained in dialog")
            assert state()["rows"] == yaml_before
            saved_row["overrides"]["thickness.trailing_edge_fullness"] = -.15
            page.locator('#blade_modeller_upload input[type="file"]').set_input_files({
                "name": "rotor.yaml", "mimeType": "text/yaml", "buffer": json.dumps(saved_row).encode(),
            })
            wait(lambda: "Loaded rotor.yaml" in page.locator("#blade_yaml_file_notice").inner_text(), "YAML file fills draft")
            assert state()["rows"] == yaml_before
            page.context.grant_permissions(["clipboard-read", "clipboard-write"])
            page.locator("#blade_yaml_copy").click()
            wait(lambda: page.evaluate("navigator.clipboard.readText()") == page.locator("#blade_yaml_text").input_value(), "copy YAML")
            page.locator("#blade_yaml_apply").click()
            wait(lambda: row(1)["overrides"].get("thickness.trailing_edge_fullness") == -.15, "row YAML import")
            page.locator("#blade_yaml_close").click()
            page.locator("#blade_yaml_text").wait_for(state="hidden")
            # Restore applies only to the selected row, then exercise pasted YAML.
            page.locator("#blade_modeller_reset").click()
            wait(lambda: not row(1)["overrides"], "restore selected row preset defaults")
            assert row() == valid_state["rows"][0]["parameters"]
            page.locator("#blade_yaml_open").click()
            page.locator("#blade_yaml_text").fill(json.dumps(saved_row))
            page.locator("#blade_yaml_apply").click()
            wait(lambda: row(1)["overrides"].get("thickness.trailing_edge_fullness") == -.15, "pasted YAML applied")
            page.locator("#blade_yaml_close").click()
            page.locator("#blade_yaml_text").wait_for(state="hidden")
            assert row() == valid_state["rows"][0]["parameters"]
            checks.append("independent row edits, immediate YAML save and round trip")
            page.locator("#blade_modeller_crosscuts input").check()
            wait(lambda: "mm" in page.locator("#blade_modeller_passage_note").inner_text(), "passage crosscuts")
            page.locator("#blade_modeller_crosscuts input").uncheck()
            page.wait_for_function("() => !document.querySelector('#geometry_profile .js-plotly-plot').data.some(t => t.name === 'Passage crosscuts')", timeout=60000)
            checks.append("optional passage crosscuts")

            # Dash queues callbacks that read their own outstanding outputs as
            # State. Record the client journal while the first reply is held,
            # then verify that the queued edit retains both operations.
            pending = []
            capture_edits = True
            page.evaluate("""() => {
                window.bladeTestEvents = [];
                window.bladeOriginalSetProps = window.dash_clientside.set_props;
                window.dash_clientside.set_props = function(id, props) {
                    if (id === 'blade_modeller_events') window.bladeTestEvents.push(JSON.parse(JSON.stringify(props.data)));
                    return window.bladeOriginalSetProps.apply(this, arguments);
                };
            }""")

            def delay_edits(route):
                request = route.request.post_data_json
                if capture_edits and "blade_modeller_edit_proposal.data" in request.get("output", ""):
                    pending.append(route)
                else:
                    route.continue_()

            page.route("**/_dash-update-component", delay_edits)
            field("camberline.turning_concentration").fill("1.623456789")
            field("camberline.turning_concentration").press("Tab")
            wait(lambda: len(pending) >= 1, "first delayed edit request")
            field("camberline.turning_location").fill("0.487654321")
            field("camberline.turning_location").press("Tab")
            wait(lambda: page.evaluate("window.bladeTestEvents.length") >= 2, "two edits recorded during pending response")
            actions = page.evaluate("window.bladeTestEvents.at(-1).actions")
            assert any(action.get("path") == "camberline.turning_concentration" and action.get("value") == 1.623456789 for action in actions)
            assert any(action.get("path") == "camberline.turning_location" and action.get("value") == .487654321 for action in actions)
            page.evaluate("() => { window.dash_clientside.set_props = window.bladeOriginalSetProps; }")
            capture_edits = False
            for route in reversed(pending):
                route.fulfill(response=route.fetch())
            page.unroute("**/_dash-update-component", delay_edits)
            wait(lambda: row(1)["overrides"].get("camberline.turning_concentration") == 1.623456789
                 and row(1)["overrides"].get("camberline.turning_location") == .487654321,
                 "both queued edits retained")
            assert solver_calls() == initial_solves
            checks.append("rapid edits retained while previous response is pending")

            valid_state = deepcopy(state())
            wait(lambda: observed.get("blade_modeller_2d_proposal", {}).get("data", {}).get("token", [None] * 4)[2] == state()["revision"], "latest 2D proposal")
            wait(lambda: observed.get("blade_modeller_3d_proposal", {}).get("data", {}).get("token", [None] * 4)[2] == state()["revision"], "latest 3D proposal")
            page.screenshot(path=str(output / "geometry.png"), full_page=True)
            page.locator("#blade_modeller_preview").scroll_into_view_if_needed()
            page.screenshot(path=str(output / "preview3d.png"), full_page=True)
            page.locator('.td-workspace-rail [data-workspace="meanline"]').click()
            page.wait_for_function("before => JSON.stringify(document.querySelector('#blades_plot .js-plotly-plot').data.filter(t => t.x?.length > 600).map(t => [t.x,t.y])) !== before",
                                   arg=initial_outline, timeout=60000)
            checks.append("meanline blade plot refreshes to edited geometry when opened")
            setup = page.locator('#td-section-operating')
            if setup.count():
                setup.evaluate("node => node.open = true")
            else:
                page.locator('.td-section').nth(1).evaluate("node => node.open = true")
            pressure = page.locator('input[id=' + json.dumps(json.dumps({"key": "exit_pressure", "scope": "overall"}, separators=(",", ":"))) + ']')
            previous_pressure = pressure.input_value()
            pressure.fill("-1")
            pressure.press("Tab")
            wait(lambda: observed.get("td-calculation-state", {}).get("data", {}).get("valid") is False, "failed calculation")
            if state() != valid_state:
                (output / "failed_calculation_before.json").write_text(json.dumps(valid_state, indent=2), encoding="utf-8")
                (output / "failed_calculation_after.json").write_text(json.dumps(state(), indent=2), encoding="utf-8")
            assert state() == valid_state
            checks.append("failed meanline calculation preserves last geometry")
            pressure.fill(previous_pressure)
            pressure.press("Tab")
            wait(lambda: state().get("_epoch") != initial_epoch, "fresh baseline after successful recalculation")
            assert all(not record["parameters"]["overrides"] for record in state()["rows"])
            wait(lambda: observed.get("blade_modeller_3d_proposal", {}).get("data", {}).get("token", [None] * 4)[1] == state()["_epoch"], "fresh committed VTK baseline")
            checks.append("successful recalculation resets all overrides")
            page.locator('#load_button input[type="file"]').set_input_files(
                str(ROOT / "demos/basic_demo/demo_radial_inflow_turbine.yaml"))
            wait(lambda: bool(state().get("rows")) and all(
                record["parameters"]["source"]["geometry"]["type"] == "radial"
                for record in state()["rows"]), "radial meanline baseline")
            page.locator('.td-workspace-rail [data-workspace="geometry"]').click()
            page.locator("#blade_modeller_preview").scroll_into_view_if_needed()
            page.wait_for_function("document.querySelector('#geometry-scene canvas') && window.dash_clientside.studio?.scenes.geometry?.active", timeout=60000)
            wait(lambda: "Radial" in page.locator("#geometry_message").inner_text(), "radial preview updated")
            wait(lambda: observed.get("blade_modeller_3d_proposal", {}).get("data", {}).get("data", {}).get("rows", [{}])[0].get("geometry_type") == "radial", "radial VTK surface data")
            page.locator("#blade_modeller_crosscuts input").check()
            wait(lambda: "axial" in page.locator("#blade_modeller_passage_note").inner_text(), "explicit radial crosscut limitation")
            correction = page.locator('[id=' + json.dumps(json.dumps({"path": "correction.preserve_physical_thickness", "type": "blade-parameter"}, separators=(",", ":"))) + '] input[type="checkbox"]')
            correction.locator("xpath=ancestor::details").evaluate_all("nodes => nodes.forEach(node => node.open = true)")
            correction.check()
            wait(lambda: row(1)["overrides"].get("correction.preserve_physical_thickness") is True, "radial physical thickness correction")
            wait(lambda: observed.get("blade_modeller_3d_proposal", {}).get("data", {}).get("token", [None] * 4)[1:3] == [state()["_epoch"], state()["revision"]], "corrected radial 3D proposal")
            page.locator("#blade_modeller_preview").scroll_into_view_if_needed()
            page.screenshot(path=str(output / "radial3d.png"), full_page=True)
            assert page.locator("#geometry_download_button").count() == 0
            checks.append("radial meanline/2D/3D workflow and explicit export/crosscut limits")
            assert not errors, errors
            passed = True
        finally:
            closing = True
            page.remove_listener("response", capture)
            (output / "observed.json").write_text(json.dumps(observed), encoding="utf-8")
            (output / "calculation_requests.json").write_text(json.dumps([
                {"changed": item.get("changedPropIds"), "status": item.get("response_status"), "inputs": item.get("inputs")}
                for item in calls if "td-calculation-state.data" in item.get("output", "")
            ], indent=2), encoding="utf-8")
            (output / "report.json").write_text(json.dumps({"passed": passed, "checks": checks, "errors": errors}, indent=2), encoding="utf-8")
            browser.close()
    print(json.dumps({"passed": passed, "checks": checks, "errors": errors}, indent=2))


if __name__ == "__main__":
    main()
