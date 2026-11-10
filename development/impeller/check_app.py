"""Browser regression for a running impeller app (default localhost:8052).

Uses an isolated headless browser. No user profile or external service is used.
"""

import argparse
import json
from pathlib import Path

import yaml


def main():
    from playwright.sync_api import sync_playwright, expect

    expect.set_options(timeout=45000)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8052")
    args = parser.parse_args()
    output = Path(__file__).parent / "output/app"
    output.mkdir(exist_ok=True, parents=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            channel="msedge",
            args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
        )
        try:
            page = browser.new_page(
                viewport={"width": 1440, "height": 1000}, accept_downloads=True
            )
            page.set_default_timeout(45000)
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(args.url)
            expect(page.locator("#impeller-status")).to_have_text("Geometry up to date")
            frame = page.locator("#impeller-3d-frame").element_handle().content_frame()
            frame.wait_for_function(
                "!!window.impellerReport && document.getElementById('impeller-viewer').dataset.fingerprint === window.impellerReport.fingerprint"
            )
            page.wait_for_function(
                "document.querySelector('#impeller-angle-plot .js-plotly-plot')?.data?.length===2"
            )
            original = frame.evaluate("window.impellerReport.fingerprint")
            frame.evaluate(
                "window.__testCanvas=window.impellerViewer.renderer.domElement;window.__testCamera=window.impellerViewer.camera.position.toArray()"
            )
            frame.locator("#blade-mode").select_option("single")

            def field(path):
                result = page.locator(f'[data-parameter="{path}"]')
                # Expand the same details container that holds this input.
                summary = result.locator("xpath=ancestor::details[1]").locator(
                    ":scope > summary"
                )
                if not result.locator("input[type=number]").is_visible():
                    summary.click()
                return result.locator("input[type=number]")

            def edit(path, value):
                box = field(path)
                box.fill(str(value))
                box.press("Tab")

            network = page.context.new_cdp_session(page)
            network.send("Network.enable")
            network.send(
                "Network.emulateNetworkConditions",
                {
                    "offline": False,
                    "latency": 300,
                    "downloadThroughput": -1,
                    "uploadThroughput": -1,
                },
            )
            edit("sections.0.camberline.inlet_metal_angle_deg", -34.125123456)
            edit("sections.-1.camberline.inlet_metal_angle_deg", -51.375987654)
            edit("sections.0.thickness.maximum_thickness_m", 7.123456789)
            expect(page.locator("#impeller-status")).to_have_text("Geometry up to date")
            page.wait_for_function(
                "Math.abs(document.querySelector('#impeller-angle-plot .js-plotly-plot').data[0].y[0]+34.125123456)<1e-9"
            )
            assert (
                abs(
                    page.evaluate(
                        "document.querySelector('#impeller-angle-plot .js-plotly-plot').data[1].y[0]"
                    )
                    + 51.375987654
                )
                < 1e-9
            )
            assert (
                abs(
                    float(
                        field("sections.0.thickness.maximum_thickness_m").input_value()
                    )
                    - 7.123456789
                )
                < 1e-10
            )
            frame.wait_for_function(
                "previous=>window.impellerReport.fingerprint!==previous", arg=original
            )
            assert frame.evaluate(
                "window.__testCanvas===window.impellerViewer.renderer.domElement"
            )
            assert frame.evaluate(
                "window.__testCamera.every((x,i)=>x===window.impellerViewer.camera.position.toArray()[i])"
            )
            assert frame.locator("#blade-mode").input_value() == "single"
            network.send(
                "Network.emulateNetworkConditions",
                {
                    "offline": False,
                    "latency": 0,
                    "downloadThroughput": -1,
                    "uploadThroughput": -1,
                },
            )

            # Saving captures exactly the accepted edited configuration.
            with page.expect_download() as download:
                page.locator("#impeller-yaml-download-button").click()
            saved = output / "browser_roundtrip.yaml"
            download.value.save_as(saved)
            config = yaml.safe_load(saved.read_text(encoding="utf-8"))
            assert (
                config["sections"][0]["camberline"]["inlet_metal_angle_deg"]
                == -34.125123456
            )
            assert (
                config["sections"][-1]["camberline"]["inlet_metal_angle_deg"]
                == -51.375987654
            )
            assert (
                abs(
                    config["sections"][0]["thickness"]["maximum_thickness_m"]
                    - 0.007123456789
                )
                < 1e-15
            )
            before_invalid = frame.evaluate("window.impellerReport.fingerprint")
            edit("dimensions.inlet_hub_radius", 400)
            expect(page.locator("#impeller-status")).to_have_text(
                "Invalid draft — showing last valid geometry"
            )
            expect(page.locator("#impeller-yaml-download-button")).to_be_disabled()
            assert frame.evaluate("window.impellerReport.fingerprint") == before_invalid
            assert page.locator("#impeller-error").inner_text()
            edit("dimensions.inlet_hub_radius", 95.56499997233045)
            expect(page.locator("#impeller-status")).to_have_text("Geometry up to date")

            page.locator("#impeller-reset").click()
            expect(page.locator("#impeller-status")).to_have_text("Geometry up to date")
            frame.wait_for_function(
                "expected=>window.impellerReport.fingerprint===expected", arg=original
            )
            page.locator("#impeller-yaml-upload input[type=file]").set_input_files(
                saved
            )
            page.wait_for_function(
                "Math.abs(document.querySelector('#impeller-angle-plot .js-plotly-plot').data[0].y[0]+34.125123456)<1e-9"
            )
            expect(page.locator("#impeller-status")).to_have_text("Geometry up to date")

            # Invalid upload does not change current geometry or inputs.
            imported_fingerprint = frame.evaluate("window.impellerReport.fingerprint")
            page.locator("#impeller-yaml-upload input[type=file]").set_input_files(
                {
                    "name": "invalid.yaml",
                    "mimeType": "application/yaml",
                    "buffer": b"dimensions: []",
                }
            )
            expect(page.locator("#impeller-error")).not_to_be_empty()
            assert (
                frame.evaluate("window.impellerReport.fingerprint")
                == imported_fingerprint
            )
            assert not errors, errors
            page.screenshot(path=str(output / "checked-app.png"))
            summary = {
                "browser": browser.version,
                "page_errors": errors,
                "checks": [
                    "rapid independent edits under latency",
                    "full typed precision",
                    "shared hub/shroud plots",
                    "persistent 3D camera and canvas",
                    "display settings retained",
                    "invalid draft retains geometry and blocks save",
                    "recovery",
                    "reset",
                    "YAML download and reload",
                    "invalid import rejected",
                ],
            }
            (output / "validation.json").write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps(summary, indent=2))
        finally:
            browser.close()


if __name__ == "__main__":
    main()
