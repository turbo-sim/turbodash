"""Run either audit section using this directory's scripts and repository models."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SECTIONS = {
    "individual": HERE / "individual_functions",
    "stage": HERE / "overall_stage",
}


def check_bundle():
    """Check human navigation and locally supplied reproduction scripts."""
    failures, checked_links = [], 0
    for section, directory in SECTIONS.items():
        expected_scripts = ["scripts/run.py", "scripts/render_saved.py"] if section == "individual" else ["study.py", "report.py"]
        for filename in expected_scripts:
            if not (directory / filename).is_file():
                failures.append(f"Missing {section} script: {filename}")
        if section == "individual":
            prefixes = {p.name[:3] for p in (directory / "report").glob("*.md")}
            if not {f"{i:02d}_" for i in range(1, 12)}.issubset(prefixes):
                failures.append(f"Missing numbered individual-function reports in {directory.name}")
        else:
            for name in ("readme.md", "comparison.md", "sensitivity_kacker_okapuu/readme.md", "sensitivity_benner/readme.md", "sensitivity_isentropic/readme.md"):
                if not (directory / name).is_file():
                    failures.append(f"Missing air-turbine report: {name}")
    for page in HERE.rglob("*.md"):
        if page.name != page.name.lower():
            failures.append(f"Uppercase Markdown filename: {page.relative_to(HERE)}")
        content = re.sub(r"```.*?```", "", page.read_text(encoding="utf-8-sig"), flags=re.S)
        for match in re.finditer(r"!?\[[^\]]*\]\((<[^>]+>|[^)]+)\)", content):
            target = match.group(1).strip().strip("<>")
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            target = unquote(target.split("#", 1)[0])
            checked_links += 1
            resolved = (page.parent / target).resolve()
            if not resolved.exists() and resolved != HERE / "validation.json":
                failures.append(f"Broken link in {page.relative_to(HERE)}: {target}")
    copied_model_names = {p.name for p in (REPO / "turbodash/loss_models").glob("*.py")} - {"__init__.py"}
    copied_model_names |= {"core_turbine.py", "math.py", "_numerics.py"}
    for file in HERE.rglob("*.py"):
        if file.name in copied_model_names:
            failures.append(f"Unexpected production-source copy: {file.relative_to(HERE)}")
    result = {"passed": not failures, "checked_links": checked_links, "failures": failures,
              "model_source": "../../turbodash", "production_source_copies": False}
    (HERE / "validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return not failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", choices=("all", *SECTIONS), default="all")
    parser.add_argument("--render-only", action="store_true", help="Redraw existing data without repeating the simulations.")
    parser.add_argument("--check", action="store_true", help="Only check navigation and script packaging.")
    args = parser.parse_args()
    if not (REPO / "turbodash/core/turbine.py").is_file():
        parser.error("Keep loss_model_audit inside loss_model_analysis in the Turbodash repository.")
    if args.check:
        return 0 if check_bundle() else 1
    for name, section in SECTIONS.items():
        if args.section not in ("all", name):
            continue
        if name == "individual":
            script = section / "scripts" / ("render_saved.py" if args.render_only else "run.py")
            command = [sys.executable, "-B", str(script)]
        else:
            command = [sys.executable, "-B", str(section / "study.py")]
            command += ["--render-only"] if args.render_only else ["--points", "241"]
        print(f"Running {section.name}: {'redraw saved data' if args.render_only else 'calculate and plot'}", flush=True)
        subprocess.run(command, cwd=HERE, check=True)
    return 0 if check_bundle() else 1


if __name__ == "__main__":
    raise SystemExit(main())
