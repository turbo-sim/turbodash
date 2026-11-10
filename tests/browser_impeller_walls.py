"""Check native wall dragging and its full impeller update in a running app.

Run the app first, then pass its --url. An isolated headless browser is used.
"""

import argparse
import json
from pathlib import Path

from playwright.sync_api import expect, sync_playwright
import yaml

STATE = """async () => {
 const {figureState}=await import('/_impeller_walls/assets/figure.js');
 return figureState('wall-editor-viewport');
}"""


def state(page):
    page.wait_for_function("""async () => { try {
      const {figureState}=await import('/_impeller_walls/assets/figure.js');
      return !!figureState('wall-editor-viewport');
    } catch {return false;} }""")
    return page.evaluate(STATE)


def point_value(snapshot, name="hub", axis=0, index=3):
    return snapshot["objects"][name]["geometry"]["parameters"]["control_points"][
        "values"
    ][axis][index]


def wait_geometry(page):
    expect(page.locator("#impeller-status")).to_have_text(
        "Geometry up to date", timeout=45000
    )
    expect(page.locator("#wall-apply")).to_be_enabled(timeout=30000)
    page.wait_for_function(
        "document.querySelector('#impeller-3d-frame').contentDocument.querySelector('#impeller-viewer')?.dataset.ready === 'true'",
        timeout=30000,
    )


def fingerprint(page):
    return page.evaluate(
        "document.querySelector('#impeller-3d-frame').contentDocument.querySelector('#impeller-viewer').dataset.fingerprint"
    )


def drag(page, index=3, dx=8, dy=-4):
    page.locator("#wall-editor-viewport canvas").scroll_into_view_if_needed()
    point = page.evaluate(
        """async index => {
      const {figureState}=await import('/_impeller_walls/assets/figure.js');
      const {planarFrame}=await import('/_impeller_walls/assets/display.js');
      const T=await import('/_impeller_walls/assets/vendor/three.module.js');
      const s=figureState('wall-editor-viewport'),c=s.camera;
      const canvas=document.querySelector('#wall-editor-viewport canvas'),b=canvas.getBoundingClientRect();
      const rect=planarFrame(b.width,b.height),aspect=s.rendering.width/s.rendering.height;
      const fw=Math.min(rect.width,rect.height*aspect),fh=fw/aspect;
      rect.left+=(rect.width-fw)/2;rect.top+=(rect.height-fh)/2;rect.width=fw;rect.height=fh;
      const half=c.ortho_half,a=rect.width/rect.height;
      const camera=new T.OrthographicCamera(-half*a,half*a,half,-half,c.near,c.far);
      camera.position.fromArray(c.position);camera.up.fromArray(c.up);camera.zoom=c.zoom;
      camera.lookAt(new T.Vector3(...c.target));camera.updateProjectionMatrix();camera.updateMatrixWorld();
      const p=s.objects.hub.geometry.parameters.control_points.values;
      const v=new T.Vector3(p[0][index],p[1][index],p[2]?.[index]||0).project(camera);
      return {x:b.left+rect.left+(v.x+1)*rect.width/2,y:b.top+rect.top+(1-v.y)*rect.height/2};
    }""",
        index,
    )
    page.mouse.move(point["x"], point["y"])
    page.mouse.down()
    page.mouse.move(point["x"] + dx, point["y"] + dy, steps=6)
    page.mouse.up()


def download(page, output):
    with page.expect_download() as downloaded:
        page.locator("#impeller-yaml-download-button").click()
    downloaded.value.save_as(output)
    return yaml.safe_load(output.read_text(encoding="utf-8"))


def run(url, output):
    output.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            channel="msedge",
            args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
        )
        page = browser.new_page(
            viewport={"width": 1600, "height": 1300}, accept_downloads=True
        )
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        try:
            page.goto(url)
            page.wait_for_selector("#wall-editor-viewport canvas", timeout=45000)
            wait_geometry(page)
            initial = state(page)
            initial_fingerprint = fingerprint(page)
            initial_channel = page.locator(
                "#impeller-channel-plot .js-plotly-plot"
            ).evaluate("g=>JSON.stringify(g.data)")
            drag(page)
            page.wait_for_function(
                """async previous=>{try {
              const {figureState}=await import('/_impeller_walls/assets/figure.js');
              return figureState('wall-editor-viewport').objects.hub.geometry.parameters.control_points.values[0][3] !== previous;
            } catch {return false;}}""",
                arg=point_value(initial),
            )
            assert (
                fingerprint(page) == initial_fingerprint
            ), "Unapplied wall edits changed accepted 3D geometry."
            page.locator("#wall-apply").click()
            expect(page.locator("#wall-notice")).to_have_text(
                "Wall edits applied to the complete impeller.", timeout=30000
            )
            page.wait_for_function(
                "previous=>document.querySelector('#impeller-3d-frame').contentDocument.querySelector('#impeller-viewer').dataset.fingerprint!==previous",
                arg=initial_fingerprint,
                timeout=45000,
            )
            wait_geometry(page)
            assert (
                page.locator("#impeller-channel-plot .js-plotly-plot").evaluate(
                    "g=>JSON.stringify(g.data)"
                )
                != initial_channel
            )
            edited = download(page, output / "dragged.yaml")
            assert "3" in {str(i) for i in edited["control_point_displacements"]["hub"]}
            assert edited["rake"]["mode"] == "trailing_edge"
            page.locator("#impeller-wall-editor").screenshot(
                path=str(output / "dragged.png")
            )

            # A no-op Apply preserves float64 offsets, including JSON key conversion.
            page.locator("#wall-apply").click()
            wait_geometry(page)
            assert download(page, output / "noop.yaml") == edited
            page.locator("#wall-clear").click()
            expect(page.locator("#wall-notice")).to_have_text(
                "Wall refinements cleared.", timeout=30000
            )
            page.wait_for_function(
                "expected=>document.querySelector('#impeller-3d-frame').contentDocument.querySelector('#impeller-viewer').dataset.fingerprint===expected",
                arg=initial_fingerprint,
                timeout=45000,
            )
            wait_geometry(page)
            assert (
                download(page, output / "cleared.yaml")["control_point_displacements"]
                == {}
            )

            # Reset and Clear must also discard local edits that were never applied.
            for button, notice in (
                (
                    "impeller-reset",
                    "Example restored, including the original wall control points.",
                ),
                ("wall-clear", "Wall refinements cleared."),
            ):
                current = state(page)
                drag(page)
                page.wait_for_function(
                    """async previous=>{try {
                  const {figureState}=await import('/_impeller_walls/assets/figure.js');
                  return figureState('wall-editor-viewport').objects.hub.geometry.parameters.control_points.values[0][3] !== previous;
                } catch {return false;}}""",
                    arg=point_value(current),
                )
                page.locator("#" + button).click()
                expect(page.locator("#wall-notice")).to_have_text(notice, timeout=30000)
                page.wait_for_function(
                    """async expected=>{try {
                  const {figureState}=await import('/_impeller_walls/assets/figure.js');
                  return figureState('wall-editor-viewport').objects.hub.geometry.parameters.control_points.values[0][3] === expected;
                } catch {return false;}}""",
                    arg=point_value(initial),
                    timeout=30000,
                )
                wait_geometry(page)

            # The native point inspector must reject locked engineering stations.
            editor = page.locator("#wall-editor")
            editor.get_by_role(
                "button", name="Open editing controls", exact=True
            ).click()
            page.get_by_label("Selected control point", exact=True).select_option("0")
            coordinate = page.get_by_label("X coordinate", exact=True)
            coordinate.fill(str(point_value(initial, index=0) + 1))
            coordinate.press("Enter")
            expect(
                page.locator("#wall-editor-viewport .nrb-status.nrb-error")
            ).to_contain_text("fixed", timeout=30000)
            assert point_value(state(page), index=0) == point_value(initial, index=0)
            assert fingerprint(page) == initial_fingerprint
            assert not errors, errors
            summary = {
                "checks": [
                    "actual pointer drag",
                    "unapplied edits retain accepted 3D",
                    "apply updates engineering plot and 3D",
                    "float64 no-op apply",
                    "clear offsets",
                    "reset unapplied edits",
                    "clear unapplied edits",
                    "locked station rejection",
                ],
                "page_errors": errors,
            }
            (output / "validation.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
            print(json.dumps(summary, indent=2))
        except Exception:
            page.screenshot(path=str(output / "failure.png"), full_page=True)
            print("Page errors:", errors)
            raise
        finally:
            browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8052")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("development/impeller/output/app_wall_checks"),
    )
    args = parser.parse_args()
    run(args.url, args.output)
