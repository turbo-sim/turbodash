"""Capture the review and check its promised interactions in an existing browser.

No server, installs or network. Browser profile, temporary files and downloaded
test feedback stay in this folder's ignored output_capture/ directory.
"""
import argparse
import json
import os
from pathlib import Path

from build_review import ROOT, main as build


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--browser", help="Existing Chrome/Chromium executable")
    args = parser.parse_args()
    output = ROOT / "output_capture"
    for folder in ("temp", "profile", "downloads"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    (ROOT / "png").mkdir(exist_ok=True)
    os.environ.update(TEMP=str(output / "temp"), TMP=str(output / "temp"), PYTHONDONTWRITEBYTECODE="1")
    from playwright.sync_api import sync_playwright, expect
    build()
    checks = []
    errors, remote_requests = [], []
    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            str(output / "profile"), headless=True,
            viewport={"width":1600,"height":1080}, device_scale_factor=2,
            downloads_path=str(output / "downloads"), accept_downloads=True,
            args=["--disable-background-networking", "--no-first-run", "--disable-component-update", "--disable-crash-reporter"],
            **({"executable_path":args.browser} if args.browser else {}),
        )
        try:
            def intercept(route):
                if route.request.url.startswith(("http://", "https://")):
                    remote_requests.append(route.request.url)
                    route.abort()
                else:
                    route.continue_()
            context.route("**/*", intercept)
            page = context.new_page()
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto((ROOT / "index.html").as_uri())
            page.evaluate("localStorage.clear()")
            page.reload()
            expect(page.locator('.inspector details[open]')).to_have_count(0)
            expect(page.locator('.section')).to_have_count(5)
            assert page.locator('#brand').inner_text() == 'turbodash'
            page.screenshot(path=ROOT / "png" / "review_light.png", full_page=True)
            checks.append("Five numbered sections initially closed; branding has no Studio")
            page.locator('#example-count').select_option('2')
            page.locator('#parameter-search').fill('low pressure rotor maximum thickness')
            expect(page.locator('.search-result').first).to_contain_text('Maximum thickness / chord')
            page.screenshot(path=ROOT / "png" / "review_search.png", full_page=True)
            page.locator('#parameter-search').press('Enter')
            expect(page.locator('#stage-select')).to_have_value('1')
            expect(page.locator('[data-row="rotor"]')).to_have_attribute('aria-pressed','true')
            expect(page.locator('#field-maximum_thickness_to_chord')).to_be_focused()
            page.locator('#field-maximum_thickness_to_chord').fill('0.3123456789')
            page.locator('#stage-select').select_option('0')
            page.locator('#field-maximum_thickness_to_chord').fill('0.287654321')
            page.locator('#stage-select').select_option('1')
            expect(page.locator('#field-maximum_thickness_to_chord')).to_have_value('0.3123456789')
            checks.append("Typeahead selects the correct stage/row and preserves independent precise values")
            page.keyboard.press('Control+k')
            page.locator('#parameter-search').fill('low pressure trailing edge wedge')
            page.locator('#parameter-search').press('Enter')
            expect(page.locator('#stage-edges')).to_have_attribute('open','')
            expect(page.locator('#field-trailing_edge_wedge_angle')).to_be_focused()
            checks.append("Keyboard search opens nested disclosures and focuses target")
            page.locator('#collapse-all').click()
            page.locator('#parameter-search').fill('property pair')
            page.locator('#parameter-search').press('Enter')
            page.locator('#property-pair').select_option('HmassP_INPUTS')
            expect(page.locator('#property-1')).to_have_value('')
            expect(page.locator('#property-2')).to_have_value('')
            expect(page.locator('#property-1-label')).to_have_text('Inlet enthalpy')
            expect(page.locator('#property-1-unit')).to_have_text('J/kg')
            expect(page.locator('#pair-message')).to_be_visible()
            page.locator('#property-1').fill('450000.123456789')
            page.locator('#property-2').fill('500000')
            expect(page.locator('#pair-message')).to_be_hidden()
            checks.append("Property-pair fallback clears values rather than reinterpreting them, with matching labels/units")
            assert page.locator('#meridional .passage').count() == 2
            assert page.locator('#meridional .rotor-passage').count() == 2
            assert page.locator('#meridional .hub').count() == 1
            assert page.locator('#meridional .shroud').count() == 1
            assert page.locator('#meridional .arrow').count() == 2
            assert 'starts at zero' in page.locator('#meridional svg').get_attribute('aria-label')
            checks.append("Single/two-stage schematic surfaces and nominal flow arrows; zero radial baseline")
            expected = {"0":{"value":"0.00","unit":"W"},"999.995":{"value":"1.00","unit":"kW"},"999995":{"value":"1.00","unit":"MW"},"-1350000":{"value":"-1.35","unit":"MW"}}
            for value, result in expected.items():
                assert page.evaluate('(v)=>reviewAPI.formatPower(v)', float(value)) == result
            assert page.evaluate('reviewAPI.formatPower(null).value') == '—'
            checks.append("Adaptive power units, rounding thresholds, negative values and unavailable value")
            before = page.evaluate('reviewAPI.getStages()')
            page.locator('[data-theme="dark"]').click()
            assert page.evaluate('reviewAPI.getStages()') == before
            expect(page.locator('#brand-logo')).to_have_attribute('src','assets/blade-ring-dark.svg')
            page.locator('#parameter-search').fill('')
            page.locator('#collapse-all').click()
            page.locator('#stage > summary').click()
            page.locator('#stage-select').select_option('0')
            page.locator('[data-row="stator"]').click()
            page.locator('#field-solidity_chord').scroll_into_view_if_needed()
            page.screenshot(path=ROOT / "png" / "review_dark.png", full_page=True)
            page.locator('[data-report="geometry"]').click()
            expect(page.locator('#report-table')).to_contain_text('Aspect ratio')
            page.locator('[data-report="flow"]').click()
            expect(page.locator('#report-table')).to_contain_text('Exit static pressure')
            checks.append("Theme preserves stage data; both logo variants; three report views")
            page.locator('#open-decisions').click()
            expect(page.locator('.decision')).to_have_count(10)
            page.screenshot(path=ROOT / "png" / "review_choices.png", full_page=True)
            page.locator('#choice-D01').select_option('Midnight Blueprint: deep navy with pale blue')
            page.locator('#notes-D01').fill('Browser-test note; exported locally only.')
            with page.expect_download() as download:
                page.locator('#export-feedback').click()
            feedback_path=output / 'downloads' / 'feedback_check.json'
            download.value.save_as(feedback_path)
            data=json.loads(feedback_path.read_text(encoding='utf-8'))
            assert data['decisions'][0]['choice'].startswith('Midnight')
            assert all(d['choice'] is None for d in data['decisions'][1:])
            page.reload()
            page.locator('#open-decisions').click()
            expect(page.locator('#notes-D01')).to_have_value('Browser-test note; exported locally only.')
            checks.append("Decision placeholders, notes, local persistence and feedback JSON export")
            page.locator('#close-decisions').click()
            page.evaluate('localStorage.clear()')
            page.reload()
            page.set_viewport_size({"width":390,"height":844})
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1')
            checks.append("Narrow review has no horizontal page overflow")
            page.set_viewport_size({"width":1400,"height":1000})
            page.goto((ROOT / 'logo_studies.html').as_uri())
            page.wait_for_function('[...document.images].every(i=>i.complete && i.naturalWidth>0)')
            page.screenshot(path=ROOT / 'png' / 'logo_studies.png', full_page=True)
            assert not errors, errors
            assert not remote_requests, remote_requests
        finally:
            context.close()
    report={'checks':checks,'page_errors':errors,'remote_requests':remote_requests}
    (output / 'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
