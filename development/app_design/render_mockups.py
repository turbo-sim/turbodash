"""Render and inspect the offline concept boards with an installed Playwright.

Optional --browser points to an existing Chromium executable; no install or network.
Browser profile, downloads, TEMP and TMP are confined to app_design/output_render/.
"""

import argparse
import json
import os
from pathlib import Path

from build_mockups import CONCEPTS, ROOT, main as build


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--browser", help="Path to existing Chromium/Chrome executable")
    args = parser.parse_args()
    output = ROOT / "output_render"
    output.mkdir(exist_ok=True)
    for name in ("temp", "downloads", "profile"):
        (output / name).mkdir(exist_ok=True)
    os.environ.update(TEMP=str(output / "temp"), TMP=str(output / "temp"),
                      PYTHONDONTWRITEBYTECODE="1")
    from playwright.sync_api import sync_playwright

    build()
    report = {"scope": "Offline design mockups only", "views": []}
    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir=str(output / "profile"), headless=True,
            viewport={"width": 1600, "height": 1160}, device_scale_factor=2,
            downloads_path=str(output / "downloads"),
            args=["--disable-background-networking", "--disable-component-update",
                  "--no-first-run", "--disable-crash-reporter"],
            **({"executable_path": args.browser} if args.browser else {}),
        )
        try:
            page = context.new_page()
            errors = []
            external_requests = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            def local_only(route):
                if route.request.url.startswith(("http://", "https://")):
                    external_requests.append(route.request.url)
                    route.abort()
                else:
                    route.continue_()
            context.route("**/*", local_only)
            for c in CONCEPTS:
                page.goto((ROOT / (c["slug"] + ".html")).as_uri())
                page.evaluate("document.fonts.ready")
                assert page.locator("details[open]").count() == 0
                assert page.locator(".control-group").count() == 7
                page.screenshot(path=ROOT / "png" / (c["slug"] + "_collapsed.png"), full_page=True)
                target = page.locator("#group-" + c["expanded"])
                target.locator(":scope > summary").focus()
                page.keyboard.press("Enter")
                assert target.get_attribute("open") is not None
                page.evaluate("""(key) => {
                    const target = document.querySelector('#group-' + key);
                    const scroll = document.querySelector('.inspector-scroll');
                    scroll.scrollTop = Math.max(0, target.offsetTop - scroll.offsetTop - 8);
                    document.querySelector('#state-label').textContent = 'INTERACTION EXAMPLE · ' + key.toUpperCase() + ' EXPANDED';
                }""", c["expanded"])
                page.locator(".app-bar").click(position={"x": 300, "y": 10})
                page.screenshot(path=ROOT / "png" / (c["slug"] + "_expanded.png"), full_page=True)
                overflow = page.evaluate("""() => [...document.querySelectorAll('body,.proposal,.inspector,.group-body,.canvas,.plot-card,.app-bar')].filter(e=>e.scrollWidth>e.clientWidth+2).map(e=>e.className)""")
                assert not overflow, (c["slug"], overflow)
                page.locator(".collapse-button").click()
                assert page.locator("details[open]").count() == 0
                report["views"].append({"concept": c["slug"], "default_collapsed": True,
                    "keyboard_expand": True, "collapse_all": True, "horizontal_overflow": overflow})
            page.set_viewport_size({"width": 1700, "height": 1160})
            page.goto((ROOT / "sidebar_comparison.html").as_uri())
            page.evaluate("document.fonts.ready")
            page.screenshot(path=ROOT / "sidebar_comparison.png", full_page=True)
            assert page.locator('.control-group[open]').count() == 4
            page.set_viewport_size({"width": 1700, "height": 1600})
            page.goto((ROOT / "index.html").as_uri())
            page.evaluate("document.body.classList.add('overview')")
            page.wait_for_function("[...document.images].every(i=>i.complete && i.naturalWidth>0)")
            report["gallery_image_count"] = page.locator("img").count()
            height = page.evaluate("Math.ceil(document.querySelector('.gallery-footer').getBoundingClientRect().bottom)")
            page.set_viewport_size({"width":1700,"height":height})
            page.screenshot(path=ROOT / "overview.png", full_page=True)
            report["page_errors"] = errors
            report["external_requests"] = external_requests
            assert not errors and not external_requests
        finally:
            context.close()
    (output / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
