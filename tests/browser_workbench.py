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
            page.screenshot(path=output / "workbench_light.png", full_page=True)
            checks.append("Five closed sections; approved light inspector, blade-ring identity and left workspace rail")
            assert page.evaluate("document.querySelector('#meridional_plot .js-plotly-plot').data.filter(t=>t.meta && t.meta.theme_role === 'flow_guide').length") == 5
            assert page.evaluate("document.querySelector('#meridional_plot .js-plotly-plot').layout.yaxis.range[0]") == 0
            page.evaluate("Plotly.relayout(document.querySelector('#meridional_plot .js-plotly-plot'), {'yaxis.range':[40,180]})")
            assert page.evaluate("document.querySelector('#meridional_plot .js-plotly-plot').layout.yaxis.range[0]") == 40
            page.locator('.td-theme-toggle [data-theme="dark"]').click()
            page.wait_for_function("document.querySelector('#meridional_plot .js-plotly-plot').layout.font.color === '#e7eff9'")
            assert page.evaluate("document.querySelector('#meridional_plot .js-plotly-plot').layout.yaxis.range[0]") == 40
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

            page.locator('.td-workspace-rail [data-workspace="geometry"]').click()
            expect(page.locator("#geometry_pane")).to_be_visible()
            expect(page.locator('[data-controls-workspace="geometry"]')).to_be_visible()
            page.locator('.td-workspace-rail [data-workspace="mesh"]').click()
            expect(page.locator("#mesh_pane")).to_be_visible()
            page.locator(".td-brand").click()
            expect(page.locator("#meanline_pane")).to_be_visible()
            expect(expected).to_have_value("0.3123456789")
            checks.append("Workspace rail and brand preserve edits; geometry and mesh controls remain available")

            # Named multistage load, independent edits, and exact round-trip serialization.
            config = yaml.safe_load((ROOT / "development/optimization/case_axial.yaml").read_text(encoding="utf-8"))
            config["inputs"]["stages"][0]["name"] = "High pressure"
            config["inputs"]["stages"].append(deepcopy(config["inputs"]["stages"][0]))
            config["inputs"]["stages"][1]["name"] = "Low pressure"
            config["optimization"]["design_variables"] = {"blade_velocity_ratio": {"lower_value": 0.5, "upper_value": 0.65}}
            config["optimization"]["constraints"] = []
            config["optimization"]["solver_options"]["max_iterations"] = 20
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
            with page.expect_download(timeout=20000) as downloaded:
                page.locator("#save_button").click()
            save = temporary / "downloads" / "roundtrip.yaml"
            downloaded.value.save_as(save)
            saved = yaml.safe_load(save.read_text(encoding="utf-8"))
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
            page.wait_for_timeout(500)
            checks.append("Property-pair conversion changes both values with matching units rather than reinterpreting them")

            page.locator("#td-collapse-all").click()
            page.locator("#opt-panel > summary").click()
            page.get_by_role("tab", name="Design variables", exact=True).click()
            expect(page.locator("#opt-variable-rows .td-opt-row")).to_have_count(1)
            page.locator("#opt-add-variable").click()
            expect(page.locator("#opt-variable-rows .td-opt-row")).to_have_count(2)
            page.get_by_role("button", name="Remove variable", exact=True).last.click()
            expect(page.locator("#opt-variable-rows .td-opt-row")).to_have_count(1)
            search("solver tolerance")
            expect(page.locator("#opt-tolerance")).to_be_focused()
            page.get_by_role("tab", name="Design variables", exact=True).click()
            page.locator("#opt-start").click()
            page.wait_for_function("/completed|success|failed/i.test(document.getElementById('opt-status').textContent)", timeout=120000)
            assert "failed" not in page.locator("#opt-status").inner_text().lower(), page.locator("#opt-status").inner_text()
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
            page.locator("#td-inspector-toggle").click()
            expect(page.locator("#design-controls")).to_be_visible()
            page.screenshot(path=output / "workbench_mobile.png", full_page=True)
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1"), "Narrow layout must not overflow the page"
            page.keyboard.press("Escape")
            expect(page.locator("#design-controls")).to_be_hidden()
            checks.append("Complete report tabs and narrow keyboard-dismissible inspector drawer")
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
