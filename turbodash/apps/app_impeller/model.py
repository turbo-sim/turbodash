"""Validated, UI-independent operations for the impeller workbench."""

import base64
import binascii
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from turbodash.geometry.impeller import build_impeller, load_config, normalize_config
from turbodash.geometry.impeller.config import config_fingerprint
from turbodash.geometry.impeller.disk import build_disk
from turbodash.geometry.impeller.printing import export_printable_stl

DEFAULT_CASE = (
    Path(__file__).resolve().parents[3]
    / "development"
    / "impeller"
    / "examples"
    / "reversible_impeller_zero_rake.yaml"
)


def default_config(path=None):
    path = Path(path) if path is not None else DEFAULT_CASE
    if not path.is_file():
        raise ValueError(
            "Supply --config with an impeller YAML, or run from the Turbodash checkout containing development/impeller/examples."
        )
    return load_config(path)


def validate_config(config):
    """Apply the same full blade checks as scripts, with bounded app sampling."""
    config = normalize_config(config)
    for key, maximum in (
        ("streamwise_points", 641),
        ("spanwise_points", 65),
        ("integration_points", 10001),
    ):
        if config["numerics"][key] > maximum:
            raise ValueError(
                f"Interactive {key} must be <= {maximum}; use the scripts for larger studies."
            )
    if config["blade_count"] > 120:
        raise ValueError("The interactive viewer supports up to 120 main blades.")
    geometry = build_impeller(config)
    disk = build_disk(geometry, stream_samples=41, angular_samples=32)
    disk_config = config.get("disk")
    if disk_config is None or any(
        disk_config.get(key) is None
        for key in ("bore_radius_m", "back_thickness_m", "nose_length_m")
    ):
        # Resolve legacy/automatic disk dimensions once so every control has an
        # explicit value and the downloaded YAML reproduces the visible body.
        config["disk"] = {
            key: disk.metadata.get(key, 0.0)
            for key in ("bore_radius_m", "back_thickness_m", "nose_length_m")
        }
        geometry = build_impeller(config)
    return geometry


def uploaded_config(contents):
    """Parse and fully validate a local YAML before replacing any app state."""
    try:
        encoded = contents.split(",", 1)[1]
        if len(encoded) > 1_400_000:
            raise ValueError("Choose a YAML file smaller than 1 MB.")
        text = base64.b64decode(encoded, validate=True).decode("utf-8-sig")
        return validate_config(yaml.safe_load(text)).config
    except (IndexError, AttributeError, binascii.Error, UnicodeError) as error:
        raise ValueError("Could not read the uploaded YAML file.") from error
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML: {error}") from error


def evaluate(config, revision):
    """Return one coherent plot/3D packet, or an error without replacement data."""
    from . import plots, viewer

    try:
        geometry = validate_config(config)
        disk = build_disk(geometry, stream_samples=81, angular_samples=80)
        return {
            "valid": True,
            "revision": revision,
            "config": geometry.config,
            "fingerprint": geometry.fingerprint,
            "figures": plots.build_figures(geometry, disk=disk),
            "metrics": plots.metric_cards(geometry),
            "viewer": viewer.payload(geometry, disk=disk),
            "warnings": geometry.diagnostics["warnings"],
        }
    except (ValueError, TypeError, KeyError, IndexError, OverflowError) as error:
        return {"valid": False, "revision": revision, "error": str(error)}


def downloadable_yaml(accepted, revision):
    """Never label the previous valid configuration as an invalid/new draft."""
    if not accepted or accepted.get("revision") != revision:
        raise ValueError("Wait for a valid current geometry before saving YAML.")
    return yaml.safe_dump(normalize_config(accepted["config"]), sort_keys=False)


def downloadable_stl(accepted, revision):
    """Export exactly one accepted design, with full sampling and mm units."""
    if not accepted or accepted.get("revision") != revision:
        raise ValueError("Wait for a valid current geometry before exporting STL.")
    config = normalize_config(accepted["config"])
    fingerprint = config_fingerprint(config)
    if fingerprint != accepted.get("fingerprint"):
        raise ValueError("The accepted geometry changed; rebuild before exporting STL.")
    geometry = validate_config(config)
    if geometry.fingerprint != fingerprint:
        raise ValueError("Resolve the disk controls before exporting STL.")
    filename = f"impeller_{fingerprint[:12]}_mm.stl"
    with TemporaryDirectory(prefix="turbodash-impeller-stl-") as directory:
        destination = Path(directory) / filename
        record = export_printable_stl(geometry, destination)
        content = base64.b64encode(destination.read_bytes()).decode("ascii")
    return {
        "download": {
            "content": content,
            "filename": filename,
            "type": "model/stl",
            "base64": True,
        },
        "dimensions_mm": record["dimensions_mm"],
    }


def stl_export_packet(request):
    """Always finish the browser's export transaction, including failures."""
    packet = {
        "request_id": request.get("request_id"),
        "revision": request.get("revision"),
        "fingerprint": request.get("fingerprint"),
    }
    try:
        return {
            **packet,
            "valid": True,
            **downloadable_stl(request, packet["revision"]),
        }
    except (
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        OverflowError,
        ImportError,
        OSError,
        RuntimeError,
    ) as error:
        return {**packet, "valid": False, "error": str(error)}
