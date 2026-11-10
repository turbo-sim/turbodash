"""Exercise the generated offline report in an isolated headless browser.

Requires Playwright and Chromium, Edge, or Chrome. No user browser profile is
opened. Screenshots and a validation summary are saved beside the HTML.
"""

import argparse
import json
from pathlib import Path


def main():
    from playwright.sync_api import sync_playwright

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--html",
        type=Path,
        default=Path(__file__).parent / "output/reversible_impeller/geometry.html",
    )
    args = parser.parse_args()
    path = args.html.resolve()
    output = path.parent / "viewer_checks"
    output.mkdir(exist_ok=True)
    with sync_playwright() as playwright:
        browser = None
        for channel in ("msedge", "chrome", None):
            try:
                browser = playwright.chromium.launch(
                    headless=True,
                    channel=channel,
                    timeout=15000,
                    args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
                )
                break
            except Exception:
                continue
        if browser is None:
            raise RuntimeError(
                "Install Chromium with python -m playwright install chromium."
            )
        try:
            page = browser.new_page(
                viewport={"width": 1440, "height": 1080}, device_scale_factor=1
            )
            errors, network = [], []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on(
                "request",
                lambda request: (
                    network.append(request.url)
                    if request.url.startswith(("http:", "https:"))
                    else None
                ),
            )
            page.goto(path.as_uri(), wait_until="load")
            page.wait_for_function(
                "document.getElementById('impeller-viewer').dataset.ready",
                timeout=30000,
            )
            assert (
                page.locator("#impeller-viewer").get_attribute("data-ready") == "true"
            ), page.locator("#viewer-status").inner_text()
            page.wait_for_function(
                "window.impellerViewer.renderer.info.render.calls > 0"
            )
            assert not errors, errors
            assert not network, network
            count = page.evaluate("window.impellerReport.blade_count")
            assert (
                page.evaluate(
                    "window.impellerViewer.bladeGroup.children.filter(x=>x.visible).length"
                )
                == count
            )
            page.screenshot(path=str(output / "assembly.png"), full_page=True)
            page.locator("#blade-mode").select_option("single")
            assert (
                page.evaluate(
                    "window.impellerViewer.bladeGroup.children.filter(x=>x.visible).length"
                )
                == 1
            )
            page.locator("#show-disk").uncheck()
            assert page.evaluate("window.impellerViewer.disk.visible") is False
            page.locator("#show-shroud").check()
            assert page.evaluate("window.impellerViewer.shroud.visible") is True
            page.locator("#show-edges").check()
            assert page.evaluate("window.impellerViewer.edgeGroup.visible") is True
            page.locator("#show-disk").check()
            page.locator("#disk-opacity").fill("0.35")
            assert (
                abs(page.evaluate("window.impellerViewer.disk.material.opacity") - 0.35)
                < 1e-10
            )
            page.screenshot(path=str(output / "single_blade.png"), full_page=False)
            page.locator("#show-shroud").uncheck()
            page.locator("#disk-opacity").fill("1")
            page.locator("#blade-mode").select_option("row")
            page.locator("#show-edges").uncheck()
            page.locator("#reset-view").click()
            home = page.evaluate("window.impellerViewer.camera.position.toArray()")
            # Exercise an actual visible cube face, followed by mouse rotation.
            face = page.locator(
                '.orientation-face[data-kind="face"][aria-hidden="false"]'
            ).first
            assert face.count() == 1
            face.click()
            page.wait_for_function(
                "previous => window.impellerViewer.camera.position.toArray().some((x,i)=>Math.abs(x-previous[i])>1e-5)",
                arg=home,
            )
            before_drag = page.evaluate(
                "window.impellerViewer.camera.position.toArray()"
            )
            box = page.locator("#impeller-viewer canvas").bounding_box()
            x, y = box["x"] + 0.55 * box["width"], box["y"] + 0.5 * box["height"]
            page.mouse.move(x, y)
            page.mouse.down()
            page.mouse.move(x + 90, y + 50, steps=10)
            page.mouse.up()
            page.wait_for_function(
                "previous => window.impellerViewer.camera.position.toArray().some((x,i)=>Math.abs(x-previous[i])>1e-5)",
                arg=before_drag,
            )
            page.locator("#reset-view").click()
            restored = page.evaluate("window.impellerViewer.camera.position.toArray()")
            assert max(abs(a - b) for a, b in zip(home, restored)) < 1e-10
            page.locator("#fullscreen-view").click()
            page.wait_for_function("!!document.fullscreenElement")
            page.evaluate("document.exitFullscreen()")
            page.wait_for_function("!document.fullscreenElement")
            page.set_viewport_size({"width": 390, "height": 844})
            page.wait_for_function(
                "window.impellerViewer.renderer.domElement.clientWidth < 400"
            )
            assert page.evaluate(
                "document.documentElement.scrollWidth <= window.innerWidth + 1"
            )
            page.locator("#impeller-viewer").screenshot(
                path=str(output / "mobile_viewer.png")
            )
            assert not errors, errors
            assert not network, network
            summary = {
                "browser": browser.version,
                "page_errors": errors,
                "external_requests": network,
                "checks": [
                    "offline file loading",
                    "full row",
                    "single blade",
                    "disk visibility and opacity",
                    "shroud and edge overlays",
                    "cube alignment",
                    "mouse rotation",
                    "reset view",
                    "fullscreen",
                    "mobile resize",
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
