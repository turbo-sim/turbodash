"""Verify the running explorer with an isolated real browser.

Start app.py first, then run this script with its --url. Requires Playwright
and Microsoft Edge (or --channel chrome). Screenshots and YAML stay in output/.
"""

import argparse
from pathlib import Path
import yaml
from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parents[1] / "output" / "browser_check"
STATE_JS = """async () => {
 const {figureState}=await import('/_mixed_flow_walls/assets/figure.js');
 return figureState('wall-editor-viewport');
}"""


def wait_ready(page):
    page.wait_for_function(
        "!document.getElementById('wall-apply').disabled", timeout=30000
    )


def apply_edits(page):
    with page.expect_response(
        lambda r: r.url.endswith("/_dash-update-component")
        and "wall-notice.children" in (r.request.post_data_json or {}).get("output", "")
    ) as response:
        page.locator("#wall-apply").click()
    result = response.value.json()["response"]["wall-notice"]["children"]
    assert result.startswith("Wall offsets applied"), result
    wait_ready(page)


def save(page, name):
    with page.expect_download() as download:
        page.locator("#save-yaml").click()
    path = OUT / name
    download.value.save_as(path)
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8065")
    parser.add_argument(
        "--channel", default="msedge", help="Chromium channel (msedge or chrome)"
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    run(args)


def run(args):
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            channel=args.channel,
            args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
        )
        page = browser.new_page(
            viewport={"width": 1600, "height": 1300}, accept_downloads=True
        )
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(args.url)
        page.wait_for_selector("#wall-editor-viewport canvas")
        wait_ready(page)
        initial = page.evaluate(STATE_JS)
        point = page.evaluate("""async () => {
          const {figureState}=await import('/_mixed_flow_walls/assets/figure.js');
          const {planarFrame}=await import('/_mixed_flow_walls/assets/display.js');
          const T=await import('/_mixed_flow_walls/assets/vendor/three.module.js');
          const s=figureState('wall-editor-viewport'), c=s.camera;
          const canvas=document.querySelector('#wall-editor-viewport canvas'), b=canvas.getBoundingClientRect();
          const rect=planarFrame(b.width,b.height), aspect=1100/640;
          const fw=Math.min(rect.width,rect.height*aspect), fh=fw/aspect;
          rect.left+=(rect.width-fw)/2;rect.top+=(rect.height-fh)/2;rect.width=fw;rect.height=fh;
          const half=c.ortho_half, a=rect.width/rect.height;
          const camera=new T.OrthographicCamera(-half*a,half*a,half,-half,c.near,c.far);
          camera.position.fromArray(c.position);camera.up.fromArray(c.up);camera.zoom=c.zoom;
          camera.lookAt(new T.Vector3(...c.target));camera.updateProjectionMatrix();camera.updateMatrixWorld();
          const p=s.objects.hub.geometry.parameters.control_points.values;
          const v=new T.Vector3(p[0][3],p[1][3],p[2]?.[3]||0).project(camera);
          return {x:b.left+rect.left+(v.x+1)*rect.width/2,y:b.top+rect.top+(1-v.y)*rect.height/2};
        }""")
        before = initial["objects"]["hub"]["geometry"]["parameters"]["control_points"][
            "values"
        ]
        page.mouse.move(point["x"], point["y"])
        page.mouse.down()
        page.mouse.move(point["x"] + 18, point["y"] - 12, steps=6)
        page.mouse.up()
        page.wait_for_function(
            """async before => {try {
          const {figureState}=await import('/_mixed_flow_walls/assets/figure.js');
          const s=figureState('wall-editor-viewport');
          return s.objects.hub.geometry.parameters.control_points.values[0][3]!==before;
        } catch {return false;}}""",
            arg=before[0][3],
            timeout=30000,
        )
        apply_edits(page)
        wait_ready(page)
        edited = save(page, "browser_edited.yaml")
        assert set(edited["control_point_displacements"]["hub"]) == {3}, edited
        apply_edits(page)
        wait_ready(page)
        assert save(page, "browser_noop.yaml") == edited
        page.locator("#wall-clear").click()
        page.wait_for_function(
            "document.getElementById('wall-notice').textContent.startsWith('Wall offsets cleared')"
        )
        wait_ready(page)
        assert "control_point_displacements" not in save(page, "browser_cleared.yaml")
        page.locator("#yaml-upload input[type=file]").set_input_files(
            OUT / "browser_edited.yaml"
        )
        page.wait_for_function(
            "document.getElementById('file-notice').textContent.startsWith('Loaded browser_edited.yaml')"
        )
        wait_ready(page)
        assert save(page, "browser_reloaded.yaml") == edited
        axial = page.locator(
            '[data-parameter="dimensions.axial_length"] input.mc-number'
        )
        axial.fill("65")
        axial.press("Tab")
        page.wait_for_function(
            "document.querySelector('[data-parameter=\"dimensions.axial_length\"] input.mc-number').value==='65'"
        )
        wait_ready(page)
        changed = save(page, "browser_parameters.yaml")
        assert changed["dimensions"]["axial_length"] == 0.065, changed
        assert (
            changed["control_point_displacements"]
            == edited["control_point_displacements"]
        )
        page.wait_for_function(
            """async () => {try { const {figureState}=await import('/_mixed_flow_walls/assets/figure.js'); return figureState('wall-editor-viewport').objects.hub.geometry.parameters.control_points.values[0][6]===69; } catch {return false;} }"""
        )
        page.get_by_label("Figure object").select_option("shroud")
        page.wait_for_function(
            """async () => {try { const {figureState}=await import('/_mixed_flow_walls/assets/figure.js'); return figureState('wall-editor-viewport').active==='shroud'; } catch { return false; }}"""
        )
        page.wait_for_function(
            "document.querySelector('#wall-editor [aria-label=\"Geometry colour\"]').value === '#d16b27'"
        )
        page.locator("#wall-editor").get_by_role(
            "button", name="Open editing controls", exact=True
        ).click()
        selector = page.get_by_label("Selected control point")
        selector.select_option("3")
        coordinate = page.get_by_label("X coordinate")
        value = float(coordinate.input_value())
        coordinate.fill(str(value + 0.5))
        coordinate.press("Enter")
        page.wait_for_function(
            """async () => {try {const {figureState}=await import('/_mixed_flow_walls/assets/figure.js'); return figureState('wall-editor-viewport').objects.shroud.geometry.parameters.control_points.values[0][3] > 44.7; }catch{return false;} }"""
        )
        apply_edits(page)
        wait_ready(page)
        both = save(page, "browser_both.yaml")
        assert (
            both["control_point_displacements"]["hub"]
            == edited["control_point_displacements"]["hub"]
        )
        assert set(both["control_point_displacements"]["shroud"]) == {3}
        page.locator("#wall-editor").get_by_role(
            "button", name="Open editing controls", exact=True
        ).click()
        page.get_by_label("Selected control point").select_option("3")
        coordinate = page.get_by_label("X coordinate")
        coordinate.focus()
        coordinate.press("Enter")
        coordinate.press("Tab")
        apply_edits(page)
        wait_ready(page)
        assert save(page, "browser_numeric_noop.yaml") == both
        page.locator(".mc-wall-editor-card").screenshot(path=OUT / "editor_dragged.png")
        assert not errors, errors
        print(
            "PASS: real pointer drag, apply, no-op, clear, YAML reload, parameter change, wall switching, numeric edit and untouched Enter; no browser errors."
        )
        browser.close()


if __name__ == "__main__":
    main()
