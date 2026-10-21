"""Real-browser workbench regression and visual captures (optional Playwright).

Run with an existing server: python tests/browser_workbench.py --url http://127.0.0.1:8062
Browser profiles, downloads and temporary files remain inside the repository.
"""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path

import yaml
from browser_picker_helpers import choose_path


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8062")
    parser.add_argument("--browser", default="C:/Program Files/Google/Chrome/Application/chrome.exe")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = ROOT / "development/app_design/implemented"
    temporary = output / "output_browser"
    for folder in ("temp", "profile", "downloads"):
        (temporary / folder).mkdir(parents=True, exist_ok=True)
    os.environ.update(TEMP=str(temporary / "temp"), TMP=str(temporary / "temp"), PYTHONDONTWRITEBYTECODE="1")
    from playwright.sync_api import sync_playwright, expect

    errors, results, checks, requests = [], [], [], []
    with sync_playwright() as playwright:
        context = playwright.chromium.launch_persistent_context(
            str(temporary / "profile"), executable_path=args.browser, headless=True,
            viewport={"width": 1680, "height": 1080}, device_scale_factor=1,
            downloads_path=str(temporary / "downloads"), accept_downloads=True,
            args=["--disable-background-networking", "--no-first-run", "--disable-component-update", "--disable-crash-reporter"],
        )
        try:
            page = context.new_page()
            page.on("pageerror", lambda error: errors.append(str(error)))

            def capture(response):
                if "_dash-update-component" not in response.url:
                    return
                requests.append(response.request.post_data_json)
                if response.status >= 400:
                    errors.append(f"Callback HTTP {response.status}: {response.request.post_data[:150]}")
                if response.status == 200:
                    try:
                        body = response.json().get("response", {})
                    except Exception:
                        return
                    if "result_store" in body:
                        results.append(body["result_store"]["data"])

            page.on("response", capture)
            page.goto(args.url)
            page.evaluate("localStorage.removeItem('turbodash.workbench.theme'); localStorage.removeItem('turbodash.workbench.inspector-width')")
            page.emulate_media(color_scheme="light")
            page.reload()
            page.wait_for_selector("#perf_table .dash-table-container", timeout=90000)
            page.wait_for_function("window.TurbodashWorkbench && document.querySelector('#meridional_plot .js-plotly-plot').layout.font.color === '#18314d'")
            expect(page.locator(".td-inspector .td-section")).to_have_count(5)
            expect(page.locator(".td-inspector .td-section[open]")).to_have_count(0)
            expect(page.locator("#td-inlet-pair input")).to_have_attribute("aria-label", "Inlet property pair")
            assert page.locator(".td-brand").inner_text() == "turbodash"
            assert page.evaluate("getComputedStyle(document.getElementById('design-controls')).backgroundColor") == "rgb(248, 250, 253)"

            def assert_optimization_available():
                button = page.locator(".td-optimization-dock #opt-start")
                expect(button).to_be_visible()
                expect(button).to_be_in_viewport()
                assert button.evaluate("""button => {
                    const rect = button.getBoundingClientRect();
                    const hit = document.elementFromPoint(rect.x + rect.width / 2, rect.y + rect.height / 2);
                    return rect.bottom <= innerHeight && rect.top >= 0 && hit && button.contains(hit);
                }"""), "The optimization action must remain visible at the bottom of the inspector"
                assert button.evaluate("button => !!button.closest('#design-controls')")

            def assert_geometry_presentation():
                for graph_id in ("meridional_plot", "blades_plot"):
                    assert page.locator(f"#{graph_id} .js-plotly-plot").evaluate(
                        "graph => graph.layout.showlegend === false && graph._fullLayout.showlegend === false"
                    )
                    expect(page.locator(f"#{graph_id} .legend")).to_have_count(0)
                assert page.locator("#meridional_plot .js-plotly-plot").evaluate("""graph => {
                    const walls = graph.data.filter(trace => ["Hub", "Shroud"].includes(trace.name));
                    return walls.length === 2 && walls.every(trace => trace.line.dash === "solid");
                }""")

            graph_ids = ["meridional_plot", "blades_plot", "efficiency_trends_plot", "loss_plot"]
            for group in ("td-geometry-plots", "td-efficiency-plots", "td-triangle-plots"):
                expect(page.locator(f"#{group}")).to_have_js_property("open", True)
            assert page.locator("#td-geometry-plots .dash-graph, #td-efficiency-plots .dash-graph").evaluate_all(
                "graphs => graphs.map(graph => graph.id)"
            ) == graph_ids
            for graph_id in graph_ids:
                page.wait_for_function(
                    "id => { const graph = document.querySelector('#' + id + ' .js-plotly-plot'); return graph && graph.data.length > 0; }",
                    arg=graph_id,
                )
            positions = [page.locator(f"#{graph_id}").bounding_box() for graph_id in graph_ids]
            assert abs(positions[0]["y"] - positions[1]["y"]) < 2
            assert abs(positions[2]["y"] - positions[3]["y"]) < 2
            assert positions[0]["x"] < positions[1]["x"]
            assert positions[2]["x"] < positions[3]["x"]
            assert positions[2]["y"] > positions[0]["y"] + positions[0]["height"]
            assert_geometry_presentation()
            for graph_id in ("efficiency_trends_plot", "loss_plot"):
                expect(page.locator(f"#{graph_id} .legend")).to_be_visible()
            page.wait_for_timeout(400)
            before_disclosures = len(results)
            for group in ("td-geometry-plots", "td-efficiency-plots", "td-triangle-plots"):
                page.locator(f"#{group} > summary").click()
                expect(page.locator(f"#{group}")).to_have_js_property("open", False)
                for graph in page.locator(f"#{group} .dash-graph").all():
                    expect(graph).to_be_hidden()
                assert_optimization_available()
                page.locator(f"#{group} > summary").click()
                for graph in page.locator(f"#{group} .dash-graph").all():
                    expect(graph).to_be_visible()
            page.locator("#tables_container > summary").click()
            expect(page.locator("#td-report-tabs")).to_be_hidden()
            page.locator("#tables_container > summary").click()
            expect(page.locator("#td-report-tabs")).to_be_visible()
            page.wait_for_timeout(400)
            assert len(results) == before_disclosures, "Plot and report disclosures must not rerun the solver"
            page.locator(".td-canvas").evaluate("element => { element.scrollTop = 0; }")
            assert_optimization_available()
            page.screenshot(path=output / "workbench_light.png", full_page=True)
            checks.append("Five closed sections; approved light inspector, blade-ring identity and left workspace rail")
            checks.append("Three independent plot disclosures place efficiency/losses below geometry and velocity triangles above reports without solving")
            checks.append("Solid hub and shroud lines; geometry legends hidden while efficiency and loss legends remain")
            checks.append("Optimization action stays at the bottom of the left inspector below closed parameter sections")
            # Unit selection redraws cached results, never the numerical model.
            triangles = page.locator("#triangles_plot_0 .js-plotly-plot")
            page.wait_for_function("document.querySelector('#triangles_plot_0 .js-plotly-plot')?.data.length > 0")
            before_units = len(results)
            expect(page.locator("#triangles_container .dash-graph")).to_have_count(1)
            page.locator("#td-triangle-mode").get_by_text("Velocity [m/s]", exact=True).click()
            page.wait_for_function("document.querySelector('#triangles_plot_0 .js-plotly-plot').layout.xaxis.title.text.includes('m/s')")
            assert triangles.evaluate("""graph => graph.data.filter(trace => trace.mode.includes('lines')).every(
                trace => trace.marker.symbol === 'arrow' && trace.marker.angleref === 'previous' && trace.marker.size[1] >= 10)""")
            page.locator("#td-triangle-mode").get_by_text("Mach numbers", exact=True).click()
            page.wait_for_function("!document.querySelector('#triangles_plot_0 .js-plotly-plot').layout.xaxis.title.text.includes('m/s')")
            page.wait_for_timeout(300)
            assert len(results) == before_units
            loss = page.locator("#loss_plot .js-plotly-plot")
            assert loss.evaluate("graph => graph.data[0].x.length === 3")
            assert loss.evaluate("graph => graph.data.some(trace => /exit kinetic/i.test(trace.name))")
            assert loss.evaluate("graph => graph.layout.yaxis.title.text.includes('[%]')")
            total_penalty = loss.evaluate("graph => graph.data.reduce((total, trace) => total + trace.y.reduce((sum, value) => sum + value, 0), 0)")
            assert abs(total_penalty - 100 * (1 - results[-1]["overall_performance"]["efficiency_ts"])) < 1e-8
            assert loss.evaluate("graph => graph.data.find(trace => trace.meta.theme_role === 'loss_profile').marker.color") == "#2462D1"
            trends = page.locator("#efficiency_trends_plot .js-plotly-plot")
            assert trends.evaluate("graph => graph.layout.yaxis.title.text.includes('[%]') && graph.data[0].line.color === '#1f77b4' && graph.data[1].line.color === '#ff7f0e'")
            page.locator("#td-efficiency-plots").scroll_into_view_if_needed()
            page.screenshot(path=output / "workbench_performance.png", full_page=True)
            checks.append("Velocity/Mach toggle redraws cached arrows; efficiency and loss budget use percent, with blue/orange app colors and final exit energy")
            page.locator(".td-canvas").evaluate("element => { element.scrollTop = 0; }")
            assert page.evaluate("document.querySelector('#meridional_plot .js-plotly-plot').data.filter(t=>t.meta && t.meta.theme_role === 'flow_guide').length") == 5
            assert page.evaluate("document.querySelector('#meridional_plot .js-plotly-plot').layout.yaxis.range[0]") == 0
            page.evaluate("Plotly.relayout(document.querySelector('#meridional_plot .js-plotly-plot'), {'yaxis.range':[40,180]})")
            assert page.evaluate("document.querySelector('#meridional_plot .js-plotly-plot').layout.yaxis.range[0]") == 40
            page.locator('.td-theme-toggle [data-theme="dark"]').click()
            page.wait_for_function("document.querySelector('#meridional_plot .js-plotly-plot').layout.font.color === '#e7eff9'")
            page.wait_for_function("document.querySelector('#efficiency_trends_plot .js-plotly-plot').data[0].line.color === '#69a9e2'")
            assert trends.evaluate("graph => graph.data[1].line.color === '#ffad5c'")
            assert loss.evaluate("graph => graph.data.find(trace => trace.meta.theme_role === 'loss_profile').marker.color") == "#88B7FF"
            assert page.evaluate("document.querySelector('#meridional_plot .js-plotly-plot').layout.yaxis.range[0]") == 40
            assert_geometry_presentation()
            page.locator('.td-theme-toggle [data-theme="light"]').click()
            page.wait_for_function("document.querySelector('#meridional_plot .js-plotly-plot').layout.font.color === '#18314d'")
            page.locator("#meridional_plot").hover()
            page.locator('#meridional_plot .modebar-btn[data-title="Reset axes"]').click()
            page.wait_for_function("document.querySelector('#meridional_plot .js-plotly-plot').layout.yaxis.range[0] === 0")
            checks.append("Extended channel guides, manual radial zoom preserved across themes, and reset to zero baseline")
            if args.smoke:
                print(json.dumps({"checks": checks, "errors": errors, "fields": page.evaluate("TurbodashWorkbench.indexFields().length")}, indent=2))
                assert not errors, errors
                return

            def search(query):
                page.keyboard.press("Control+k")
                page.locator("#td-parameter-search").fill(query)
                expect(page.locator(".td-search-result").first).to_be_visible()
                page.locator("#td-parameter-search").press("Enter")

            def field(key, stage=None):
                scope = "overall" if stage is None else "stage"
                return page.locator(f'input[id*=\'"scope":"{scope}"\'][id*=\'"key":"{key}"\']' +
                                    (f'[id*=\'"stage":{stage}\']' if stage is not None else ""))

            search("rotor maximum thickness meridional")
            expected = field("rotor.maximum_thickness_to_chord", 0)
            expect(expected).to_be_focused()
            expect(page.locator("#td-workbench")).to_have_attribute("data-blade-row", "rotor")
            expected.fill("0.3123456789")
            expected.blur()
            page.wait_for_timeout(1600)
            page.screenshot(path=output / "workbench_stage.png", full_page=True)
            checks.append("Keyboard typeahead reveals rotor field, opens disclosures and accepts precise floating point")

            before = len(results)
            page.locator('.td-theme-toggle [data-theme="dark"]').click()
            page.wait_for_function("document.querySelector('#meridional_plot .js-plotly-plot').layout.font.color === '#e7eff9'")
            expect(expected).to_have_value("0.3123456789")
            page.wait_for_timeout(500)
            assert len(results) == before, "Theme must not rerun the solver"
            page.screenshot(path=output / "workbench_dark.png", full_page=True)
            page.emulate_media(color_scheme="light")
            expect(page.locator("#td-workbench")).to_have_attribute("data-theme", "dark")
            assert page.evaluate("localStorage.getItem('turbodash.workbench.theme')") == "dark"
            checks.append("Explicit dark theme persists over OS changes, recolors real Plotly figures, leaves model untouched")

            # Documentation opens independently; the calculation remains untouched.
            documentation = page.locator(".td-header #td-documentation-link")
            expect(documentation).to_have_attribute("href", "/docs/")
            expect(documentation).to_have_attribute("target", "_blank")
            expect(page.locator("#td-documentation-pane, #td-documentation-back")).to_have_count(0)
            assert documentation.evaluate(
                "link => [link.previousElementSibling, link.nextElementSibling].some(neighbor => neighbor && /github\\.com/.test(neighbor.href))"
            ), "Documentation should be beside the GitHub link in the header"
            before_docs = len(results)
            model_before_docs = deepcopy(results[-1])
            fields_before_docs = page.locator('#design-controls input[type="number"]').evaluate_all(
                "fields => fields.map(field => [field.id, field.value])"
            )
            page.evaluate("document.getElementById('td-workbench')._browserRegressionIdentity = 'mounted-calculation'")
            with context.expect_page() as new_page:
                documentation.click()
            docs = new_page.value
            expect(docs.get_by_role("heading", name="Turbodash documentation", level=1)).to_be_visible(timeout=30000)
            expect(docs.get_by_role("link", name="Return to calculations", exact=True)).to_have_count(0)
            docs.screenshot(path=output / "workbench_documentation.png", full_page=True)
            docs.close()
            expect(page.locator("#meanline_pane")).to_be_visible()
            expect(expected).to_have_value("0.3123456789")
            page.wait_for_timeout(500)
            assert page.evaluate("document.getElementById('td-workbench')._browserRegressionIdentity") == "mounted-calculation"
            assert page.locator('#design-controls input[type="number"]').evaluate_all(
                "fields => fields.map(field => [field.id, field.value])"
            ) == fields_before_docs
            assert len(results) == before_docs, "Reading documentation must not rerun the solver"
            assert results[-1] == model_before_docs
            checks.append("Header documentation opens Sphinx in a new tab without a return button or any change to the calculation")

            page.locator('.td-workspace-rail [data-workspace="geometry"]').click()
            expect(page.locator("#geometry_pane")).to_be_visible()
            expect(page.locator("#opt-start")).to_be_hidden()
            expect(page.locator('[data-controls-workspace="geometry"]')).to_be_visible()
            page.locator('.td-workspace-rail [data-workspace="mesh"]').click()
            expect(page.locator("#mesh_pane")).to_be_visible()
            expect(page.locator("#opt-start")).to_be_hidden()
            page.locator(".td-brand").click()
            expect(page.locator("#meanline_pane")).to_be_visible()
            assert_optimization_available()
            expect(expected).to_have_value("0.3123456789")
            checks.append("Workspace rail and brand preserve edits; geometry and mesh controls remain available")

            # Named multistage load, independent edits, and exact round-trip serialization.
            config = yaml.safe_load((ROOT / "demos/optimization/case_axial.yaml").read_text(encoding="utf-8"))
            config["inputs"]["stages"][0]["name"] = "High pressure"
            config["inputs"]["stages"].append(deepcopy(config["inputs"]["stages"][0]))
            config["inputs"]["stages"][1]["name"] = "Low pressure"
            config["optimization"]["design_variables"] = {"blade_velocity_ratio": {"lower_value": 0.5, "upper_value": 0.65}}
            config["optimization"]["constraints"] = []
            config["optimization"]["solver_options"]["maxiter"] = 20
            page.locator('#load_button input[type="file"]').set_input_files({"name": "two-stage-review.yaml", "mimeType": "text/yaml", "buffer": yaml.safe_dump(config).encode()})
            expect(page.locator("#stage_count_label")).to_have_text("2")
            expect(page.locator(".td-inspector .td-section[open]")).to_have_count(0)
            search("low pressure rotor maximum thickness meridional")
            low = field("rotor.maximum_thickness_to_chord", 1)
            expect(low).to_be_focused()
            low.fill("0.3123456789")
            low.blur()
            search("high pressure rotor maximum thickness meridional")
            high = field("rotor.maximum_thickness_to_chord", 0)
            high.fill("0.287654321")
            high.blur()
            search("low pressure rotor maximum thickness meridional")
            expect(low).to_have_value("0.3123456789")
            page.wait_for_timeout(2000)
            page.locator("#td-triangle-mode").get_by_text("Velocity [m/s]", exact=True).click()
            page.wait_for_function("document.querySelector('#triangles_plot_1 .js-plotly-plot').layout.xaxis.title.text.includes('m/s')")
            with page.expect_download(timeout=20000) as downloaded:
                page.locator("#save_button").click()
            save = temporary / "downloads" / "roundtrip.yaml"
            downloaded.value.save_as(save)
            saved = yaml.safe_load(save.read_text(encoding="utf-8"))
            expect(page.locator("#triangles_container .dash-graph")).to_have_count(2)
            triangle_positions = [graph.bounding_box() for graph in page.locator("#triangles_container .dash-graph").all()]
            assert triangle_positions[1]["y"] >= triangle_positions[0]["y"] + triangle_positions[0]["height"]
            assert page.locator("#loss_plot .js-plotly-plot").evaluate("graph => graph.data[0].x.length === 5")
            page.locator("#td-triangle-plots").scroll_into_view_if_needed()
            for triangle in page.locator("#triangles_container .js-plotly-plot").all():
                assert triangle.evaluate("graph => graph.data.every(trace => trace.line.color === trace.marker.color)")
            page.screenshot(path=output / "workbench_velocity_triangles.png", full_page=True)
            page.locator(".td-canvas").evaluate("element => { element.scrollTop = 0; }")
            assert [stage["name"] for stage in saved["inputs"]["stages"]] == ["High pressure", "Low pressure"]
            assert saved["inputs"]["stages"][1]["blade_geometry"]["rotor"]["maximum_thickness_to_chord"] == 0.3123456789
            assert saved["inputs"]["stages"][0]["blade_geometry"]["rotor"]["maximum_thickness_to_chord"] == 0.287654321
            assert saved["optimization"]["design_variables"]
            page.screenshot(path=output / "workbench_multistage.png", full_page=True)
            page.locator('.td-theme-toggle [data-theme="light"]').click()
            page.wait_for_function("document.querySelector('#meridional_plot .js-plotly-plot').layout.font.color === '#18314d'")
            page.wait_for_timeout(200)  # Finish Bootstrap's tab-color transition before the visual capture.
            page.screenshot(path=output / "workbench_multistage_light.png", full_page=True)
            checks.append("Named multistage load and save retain independent precise blade edits and optimization settings")

            search("property pair")
            before_conditions = (float(field("inlet_property_1").input_value()), float(field("inlet_property_2").input_value()))
            page.locator("#td-inlet-pair").click()
            page.locator("#td-inlet-pair input").fill("HmassP")
            page.locator("#td-inlet-pair input").press("Enter")
            expect(page.locator("#td-inlet-label-1")).to_contain_text("enthalpy")
            assert float(field("inlet_property_1").input_value()) != before_conditions[0]
            assert abs(float(field("inlet_property_2").input_value()) - before_conditions[0]) < 1e-5
            for _ in range(40):
                if results[-1]["inputs"]["inlet_property_pair"] == "HmassP_INPUTS":
                    break
                page.wait_for_timeout(250)
            assert results[-1]["inputs"]["inlet_property_pair"] == "HmassP_INPUTS"
            page.wait_for_function("""() => Array.from(document.querySelectorAll('#triangles_container .js-plotly-plot'))
                .every(graph => graph.layout.xaxis.title.text.includes('m/s'))""")
            page.wait_for_timeout(500)
            checks.append("Property-pair conversion changes both values with matching units rather than reinterpreting them")

            page.locator("#td-collapse-all").click()
            assert_optimization_available()
            page.locator("#opt-panel > summary").click()
            page.get_by_role("tab", name="Design variables", exact=True).click()
            expect(page.locator("#opt-variable-rows .td-opt-row")).to_have_count(1)
            choose_path(page, "variable", "rotational_speed")
            expect(page.locator("#opt-variable-rows .td-opt-row")).to_have_count(2)
            page.get_by_role("button", name="Remove variable", exact=True).last.click()
            expect(page.locator("#opt-variable-rows .td-opt-row")).to_have_count(1)
            search("solver tolerance")
            expect(page.locator("#opt-tolerance")).to_be_focused()
            page.get_by_role("tab", name="Design variables", exact=True).click()
            page.locator("#opt-start").click()
            expect(page.locator("#opt-actions")).to_have_class("td-optimization-dock td-opt-state-success", timeout=120000)
            expect(page.locator("#opt-state-label")).to_contain_text("Applied")
            expect(page.locator("#opt-start")).to_have_class("td-opt-start btn btn-success")
            expect(page.locator("#opt-terminal")).to_have_js_property("open", True)
            checks.append("Optimization editor add/remove and real background solve complete through redesigned controls")
            checks.append("Parameter search reveals controls in inactive optimization tabs")

            width_before = page.locator("#design-controls").bounding_box()["width"]
            page.locator("#td-inspector-resize").focus()
            page.keyboard.press("ArrowRight")
            assert page.locator("#design-controls").bounding_box()["width"] > width_before
            checks.append("Inspector width is resizable with the keyboard")

            for tab in ("Geometry", "Flow stations", "Performance"):
                page.locator("#td-report-tabs").get_by_role("tab", name=tab, exact=True).click()
            page.locator('.td-theme-toggle [data-theme="light"]').click()
            page.locator("#td-collapse-all").click()
            page.set_viewport_size({"width": 390, "height": 844})
            expect(page.locator("#design-controls")).to_be_hidden()
            expect(page.locator("#opt-start")).to_be_hidden()
            page.locator("#td-inspector-toggle").click()
            expect(page.locator("#design-controls")).to_be_visible()
            assert_optimization_available()
            page.screenshot(path=output / "workbench_mobile.png", full_page=True)
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1"), "Narrow layout must not overflow the page"
            page.keyboard.press("Escape")
            expect(page.locator("#design-controls")).to_be_hidden()
            expect(page.locator("#opt-start")).to_be_hidden()
            checks.append("Complete report tabs and narrow keyboard-dismissible inspector drawer")
            checks.append("Optimization action is visible in the mobile inspector and hidden when its drawer closes")
            page.evaluate("localStorage.removeItem('turbodash.workbench.theme')")
            page.emulate_media(color_scheme="dark")
            page.reload()
            expect(page.locator("#td-workbench")).to_have_attribute("data-theme", "dark")
            checks.append("Fresh browser preference follows a dark operating-system theme")
            assert not errors, errors
            report = {"checks": checks, "browser_errors": errors, "successful_result_updates": len(results)}
            (output / "browser_checks.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(report, indent=2))
        except Exception:
            page.screenshot(path=temporary / "failure.png", full_page=True)
            (temporary / "failure.json").write_text(json.dumps({
                "checks": checks, "browser_errors": errors,
                "calculation": page.locator("#td-calculation-status").inner_text(),
                "optimization": page.locator("#opt-status").inner_text(),
                "recent_requests": requests[-8:],
            }, indent=2), encoding="utf-8")
            raise
        finally:
            context.close()


if __name__ == "__main__":
    main()
