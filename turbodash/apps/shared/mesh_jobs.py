"""Isolated Gmsh jobs and opaque, expiring artifact handles for the Dash app.

Only JSON requests cross the process boundary. Artifact paths are resolved on
the server, never accepted from browser state. A shared local cache also works
with multiple Gunicorn workers on the same machine.
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from zipfile import ZIP_DEFLATED, ZipFile

from turbodash.apps.shared.paths import PROJECT_ROOT
from turbodash.apps.shared.serialization import json_default


CACHE_ROOT = Path(tempfile.gettempdir()) / "turbodash_mesh_jobs"
RETENTION_SECONDS = 24 * 60 * 60
JOB_TIMEOUT_SECONDS = 30 * 60
MAX_ACTIVE_JOBS = 2
_PROCESSES = {}


def fingerprint(results, stage_index=0, row="stator", options=None):
    """Identify the complete solved design and requested mesh settings."""
    payload = json.dumps(
        [results, stage_index, row, options or {}], sort_keys=True,
        separators=(",", ":"), default=json_default,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_json(path, value):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, default=json_default), encoding="utf-8")
    temporary.replace(path)


def _read_mapping(path):
    """Treat malformed cache records as unavailable artifacts, not server errors."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("Mesh artifacts are unavailable. Generate a new mesh.") from exc
    if not isinstance(value, dict):
        raise ValueError("Mesh artifacts are unavailable. Generate a new mesh.")
    return value


def _read_identity(folder):
    identity = _read_mapping(folder / "identity.json")
    created = identity.get("created")
    try:
        valid_created = not isinstance(created, bool) and isinstance(created, (int, float)) and math.isfinite(created)
    except OverflowError:
        valid_created = False
    if (not isinstance(identity.get("fingerprint"), str) or not identity["fingerprint"]
            or not valid_created):
        raise ValueError("Mesh artifacts are unavailable. Generate a new mesh.")
    return identity


def _read_status(folder):
    status = _read_mapping(folder / "status.json")
    if status.get("state") not in ("running", "complete", "failed"):
        raise ValueError("Mesh artifacts are unavailable. Generate a new mesh.")
    return status


def job_directory(handle):
    """Resolve an unguessable handle, checking its fingerprint and lifetime."""
    if (not isinstance(handle, dict) or not isinstance(handle.get("id"), str)
            or not re.fullmatch(r"[a-f0-9]{32}", handle["id"])):
        raise ValueError("Invalid mesh job handle. Generate a new mesh.")
    root = CACHE_ROOT.resolve()
    folder = (root / handle["id"]).resolve()
    if folder.parent != root or not folder.is_dir():
        raise ValueError("Mesh artifacts are unavailable. Generate a new mesh.")
    identity = _read_identity(folder)
    if identity["fingerprint"] != handle.get("fingerprint"):
        raise ValueError("Mesh job does not match this design.")
    if time.time() - identity["created"] > RETENTION_SECONDS:
        raise ValueError("Mesh artifacts expired after 24 hours. Generate a new mesh.")
    return folder


def _reap():
    for job_id, process in list(_PROCESSES.items()):
        if process.poll() is not None:
            path = CACHE_ROOT / job_id / "status.json"
            try:
                status = _read_status(path.parent)
                if status.get("state") == "running":
                    _write_json(path, {"state": "failed", "message": "Mesh worker stopped unexpectedly. Adjust settings and generate again."})
            except (OSError, ValueError):
                pass
            _PROCESSES.pop(job_id, None)


def _cleanup():
    """Remove only expired, terminal job folders directly inside our cache."""
    root = CACHE_ROOT.resolve()
    if not root.exists():
        return
    for item in root.iterdir():
        if not re.fullmatch(r"[a-f0-9]{32}", item.name) or not item.is_dir():
            continue
        target = item.resolve()
        if target.parent != root or item.is_symlink():
            continue
        try:
            status = _read_status(target)
            identity = _read_identity(target)
            if status.get("state") in ("complete", "failed") and time.time() - identity["created"] > RETENTION_SECONDS:
                shutil.rmtree(target)
        except (OSError, ValueError):
            continue


@contextmanager
def _launch_lock():
    """Serialize process admission across threads and local server workers."""
    lock = CACHE_ROOT / "launch.lock"
    deadline = time.monotonic() + 5
    while True:
        try:
            lock.mkdir()
            break
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > 120:
                    lock.rmdir()
                    continue
            except FileNotFoundError:
                continue
            if time.monotonic() >= deadline:
                raise ValueError("Another mesh job is starting. Try again shortly.")
            time.sleep(0.05)
    try:
        yield
    finally:
        lock.rmdir()


def start_job(results, stage_index=0, row="stator", options=None):
    """Start a numerical worker, immediately returning a JSON-safe handle."""
    from turbodash.meshing.turbomoc import build_blade_row

    # Validate before allocating a process or publishing a pending handle.
    build_blade_row(results, stage_index, row, options=options)
    _reap()
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    with _launch_lock():
        _cleanup()
        active = 0
        for identity_file in CACHE_ROOT.glob("*/identity.json"):
            try:
                handle = _read_mapping(identity_file)
                active += job_status(handle)["state"] == "running"
            except (ValueError, OSError):
                continue
        if active >= MAX_ACTIVE_JOBS:
            raise ValueError("Two mesh jobs are already running. Wait for one to finish before generating again.")
        return _start_job(results, stage_index, row, options)


def _start_job(results, stage_index, row, options):
    handle = {"id": uuid.uuid4().hex,
              "fingerprint": fingerprint(results, stage_index, row, options)}
    folder = CACHE_ROOT / handle["id"]
    folder.mkdir()
    created = time.time()
    _write_json(folder / "identity.json", {**handle, "created": created})
    request = {"results": results, "stage_index": stage_index, "row": row,
               "options": options or {}, "created": created}
    _write_json(folder / "request.json", request)
    _write_json(folder / "status.json", {"state": "running", "message": "Starting mesh worker…", "fraction": None})
    try:
        with (folder / "worker.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "turbodash.apps.shared.mesh_jobs", str(folder / "request.json")],
                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                cwd=str(PROJECT_ROOT),
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        _PROCESSES[handle["id"]] = process
    except OSError as exc:
        _write_json(folder / "status.json", {"state": "failed", "message": f"Cannot start mesh worker: {exc}"})
        raise ValueError(f"Cannot start mesh worker: {exc}") from exc
    return handle


def job_status(handle):
    folder = job_directory(handle)
    status = _read_status(folder)
    process = _PROCESSES.get(handle["id"])
    if status["state"] == "running":
        created = _read_identity(folder)["created"]
        exited = process is not None and process.poll() is not None
        if exited or time.time() - created > JOB_TIMEOUT_SECONDS + 30:
            if process is not None and process.poll() is None:
                process.kill()
            # Re-read after checking the process: it may just have finished.
            status = _read_status(folder)
            if status["state"] == "running":
                status = {"state": "failed", "message": "Mesh worker stopped or timed out. Adjust settings and generate again."}
                _write_json(folder / "status.json", status)
    if status["state"] == "complete":
        status["manifest"] = _read_mapping(folder / "manifest.json")
    _reap()
    return status


def cancel_job(handle):
    """Cooperatively cancel even when a different web worker receives the call."""
    folder = job_directory(handle)
    if job_status(handle)["state"] == "running":
        (folder / "cancel").touch()
    return job_status(handle)


def mesh_archive(handle):
    """Package a completed mesh, geometry and quality reports for download."""
    status = job_status(handle)
    if status["state"] != "complete":
        raise ValueError("Wait for a successful mesh before downloading.")
    folder = job_directory(handle)
    buffer = BytesIO()
    excluded = {"request.json", "identity.json", "status.json", "worker.log", "cancel"}
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        for path in sorted(folder.rglob("*")):
            if path.is_file() and not path.is_symlink() and path.name not in excluded and path.suffix != ".tmp":
                if folder.resolve() in path.resolve().parents:
                    archive.write(path, path.relative_to(folder).as_posix())
    return buffer.getvalue()


def _run_worker(request_path):
    import threading
    import traceback

    from turbodash.meshing.turbomoc import generate_mesh

    request_path = Path(request_path).resolve()
    folder = request_path.parent
    request = json.loads(request_path.read_text(encoding="utf-8"))
    # A hung native mesh must not occupy a server process indefinitely.
    finished = threading.Event()

    def watch():
        started = time.monotonic()
        while not finished.wait(0.5):
            cancelled = (folder / "cancel").exists()
            if cancelled or time.monotonic() - started > JOB_TIMEOUT_SECONDS:
                message = "Mesh job cancelled." if cancelled else "Mesh job exceeded the 30-minute time limit."
                _write_json(folder / "status.json", {"state": "failed", "message": message, "fraction": None})
                os._exit(124)

    watchdog = threading.Thread(target=watch, daemon=True)
    watchdog.start()

    def progress(message, fraction=None):
        _write_json(folder / "status.json", {"state": "running", "message": str(message), "fraction": fraction})

    try:
        manifest = generate_mesh(
            request["results"], folder, request["stage_index"], request["row"],
            options=request["options"], progress=progress,
        )
        _write_json(folder / "manifest.json", manifest)
        _write_json(folder / "status.json", {"state": "complete", "message": "Mesh generation complete.", "fraction": 1.0})
    except Exception as exc:
        traceback.print_exc()
        _write_json(folder / "status.json", {"state": "failed", "message": str(exc), "fraction": None})
    finally:
        finished.set()


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m turbodash.apps.shared.mesh_jobs REQUEST_JSON")
    _run_worker(sys.argv[1])


if __name__ == "__main__":
    main()
