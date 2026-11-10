"""Exercise rounded-nose controls and real STL downloads in the running app."""

import argparse
import json
from pathlib import Path
import struct
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from playwright.sync_api import sync_playwright, expect
    from turbodash.geometry.impeller.output import SurfaceMesh
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8052")
    args = parser.parse_args()
    output = Path(__file__).parent / "output/app_stl_checks"
    output.mkdir(parents=True, exist_ok=True)
    checks, errors, downloads = [], [], []
    expect.set_options(timeout=45000)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True, channel="msedge",
            args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
        )
        try:
            page = browser.new_page(viewport={"width": 1440, "height": 1100}, accept_downloads=True)
            page.set_default_timeout(45000)
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on("download", lambda download: downloads.append(download.suggested_filename))
            page.goto(args.url)
            expect(page.locator("#impeller-status")).to_have_text("Geometry up to date")
            frame = page.locator("#impeller-3d-frame").element_handle().content_frame()
            frame.wait_for_function("!!window.impellerReport && !!window.impellerViewer")

            def field(path):
                row = page.locator(f'[data-parameter="{path}"]')
                box = row.locator('input[type=number]')
                if not box.is_visible():
                    row.locator("xpath=ancestor::details[1]").locator(":scope > summary").click()
                return box

            def edit(path, value):
                box = field(path)
                box.fill(str(value))
                box.press("Tab")

            def ready():
                expect(page.locator("#impeller-status")).to_have_text("Geometry up to date")

            def tip():
                return frame.evaluate("window.impellerReport.disk.vertices.reduce((z,p)=>Math.min(z,p[2]),Infinity)")

            expect(field("disk.nose_length_m")).to_have_value("35")
            expect(field("disk.bore_radius_m")).to_have_value("0")
            expect(field("disk.bore_radius_m")).to_be_disabled()
            initial_tip = tip()
            original_fingerprint = frame.evaluate("window.impellerReport.fingerprint")
            poles = frame.evaluate("window.impellerReport.disk.vertices.filter(p=>p[0]===0 && p[1]===0).length")
            assert poles == 2
            checks.append("rounded nose and solid centre visible in preview")
            button = page.locator("#impeller-stl-download-button")
            expect(button).to_be_enabled()
            page.locator(".impeller-assembly-panel").screenshot(path=str(output / "rounded_nose.png"))

            with page.expect_download(timeout=120000) as pending:
                button.click()
                expect(button).to_be_disabled()
                expect(button).to_contain_text("Preparing STL")
            download = pending.value
            expected_name = f"impeller_{original_fingerprint[:12]}_mm.stl"
            assert download.suggested_filename == expected_name
            destination = output / expected_name
            download.save_as(str(destination))
            expect(page.locator("#impeller-stl-status")).to_contain_text("STL ready")
            expect(button).to_be_enabled()
            data = destination.read_bytes()
            count = struct.unpack("<I", data[80:84])[0]
            assert len(data) == 84 + 50 * count and b"length_units=mm" in data[:80]
            records = np.frombuffer(data, offset=84, dtype=[("normal", "<f4", (3,)), ("points", "<f4", (3, 3)), ("attribute", "<u2")])
            vertices, inverse = np.unique(records["points"].reshape(-1, 3), axis=0, return_inverse=True)
            faces = inverse.reshape(-1, 3)
            mesh = SurfaceMesh(vertices.astype(float) / 1000, faces)
            quality = mesh.validate(closed=True, positive_volume=True)
            edges = np.concatenate((faces[:, :2], faces[:, 1:], faces[:, [2, 0]]))
            graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(vertices), len(vertices))).tocsr()
            assert connected_components(graph, directed=False, return_labels=False) == 1
            assert len(vertices) - 3 * len(faces) // 2 + len(faces) == 2
            assert abs(float(np.ptp(vertices[:, 0])) - 637.1) < .001
            assert abs(float(vertices[:, 2].min()) / 1000 - initial_tip) < 1e-6
            checks.append("real browser STL download is full-size mm, connected, watertight and bore-free")

            edit("disk.nose_length_m", 45)
            ready()
            frame.wait_for_function("window.impellerReport.fingerprint!==" + json.dumps(original_fingerprint))
            assert abs(tip() - initial_tip + .01) < 1e-10
            checks.append("nose control changes actual preview geometry")
            with page.expect_download() as pending:
                page.locator("#impeller-yaml-download-button").click()
            yaml_file = output / "edited_disk.yaml"
            pending.value.save_as(str(yaml_file))
            disk = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))["disk"]
            assert disk["nose_length_m"] == .045 and disk["bore_radius_m"] == 0
            checks.append("saved YAML retains nose and solid-centre settings")

            before_downloads = len(downloads)
            button.click()
            expect(button).to_be_disabled()
            edit("disk.nose_length_m", 50)
            ready()
            expect(page.locator("#impeller-stl-status")).to_contain_text("old STL was discarded", timeout=120000)
            assert len(downloads) == before_downloads
            expect(button).to_be_enabled()
            checks.append("changing design during export suppresses stale STL download")

            edit("disk.nose_length_m", -1)
            expect(page.locator("#impeller-status")).to_contain_text("Invalid draft")
            expect(button).to_be_disabled()
            edit("disk.nose_length_m", 0)
            ready()
            expect(field("disk.bore_radius_m")).to_be_enabled()
            edit("disk.bore_radius_m", 5)
            ready()
            edit("disk.bore_radius_m", 0)
            ready()
            edit("disk.nose_length_m", 35)
            ready()
            expect(button).to_be_enabled()
            checks.append("invalid geometry blocks export; flat/bored and rounded/solid controls recover")
            page.set_viewport_size({"width": 390, "height": 844})
            button.scroll_into_view_if_needed()
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            page.screenshot(path=str(output / "mobile_export.png"))
            checks.append("mobile export controls fit without horizontal scrolling")
            assert not errors, errors
            record = {"checks": checks, "javascript_errors": errors, "download": expected_name, "surface_quality": quality}
            (output / "validation.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(record, indent=2))
        finally:
            browser.close()


if __name__ == "__main__":
    main()
