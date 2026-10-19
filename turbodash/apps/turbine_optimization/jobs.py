"""Session-owned subprocess jobs with atomic progress records and PSV logs.

The web server never runs a solve or redirects stdout. Workers publish complete
physical snapshots, so future live geometry clients can consume the same files.
The local cache supports multiple server workers on a single host.
"""

from contextlib import contextmanager
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import tempfile
import threading
import time
import uuid

from turbodash.apps.shared.paths import PROJECT_ROOT
from turbodash.apps.shared.serialization import json_default
from .model import flat_config


CACHE_ROOT = Path(tempfile.gettempdir()) / "turbodash_optimization_jobs"
TIMEOUT_SECONDS = 600
RETENTION_SECONDS = 24 * 3600
MAX_ACTIVE_JOBS = 2
_PROCESSES = {}


def _write_json(path, value):
    """Publish atomically; Windows readers can briefly hold the destination."""
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, default=json_default), encoding="utf-8")
    for attempt in range(30):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 29:
                raise
            time.sleep(.02)


def _read_mapping(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            return value
    except (ValueError, OSError) as error:
        raise ValueError("Optimization data is unavailable. Start a new run.") from error
    raise ValueError("Optimization data is unavailable. Start a new run.")


def configure_sessions(server):
    """Persist a local signing key so sessions survive reloads/server workers."""
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    key_path = CACHE_ROOT / "session.key"
    try:
        with key_path.open("xb") as stream:
            stream.write(secrets.token_bytes(32))
    except FileExistsError:
        pass
    server.config.update(SECRET_KEY=os.environ.get("TURBODASH_SECRET_KEY") or key_path.read_bytes(),
                         SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax")


def session_owner():
    from flask import session
    if "optimization_owner" not in session:
        session["optimization_owner"] = uuid.uuid4().hex
    return session["optimization_owner"]


def fingerprint(config):
    def canonical(value):
        if isinstance(value, dict):
            return {key: canonical(item) for key, item in value.items()}
        if isinstance(value, list):
            return [canonical(item) for item in value]
        if isinstance(value, (float, int)) and not isinstance(value, bool):
            return float(value)
        return value
    payload = json.dumps(canonical(flat_config(config)["inputs"]), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def job_directory(handle, owner):
    job_id = handle.get("id") if isinstance(handle, dict) else None
    if not isinstance(job_id, str) or not re.fullmatch(r"[a-f0-9]{32}", job_id):
        raise ValueError("Invalid optimization job. Start a new run.")
    root = CACHE_ROOT.resolve()
    folder = (root / job_id).resolve()
    if folder.parent != root or not folder.is_dir():
        raise ValueError("Optimization job is unavailable. Start a new run.")
    identity = _read_mapping(folder / "identity.json")
    if identity.get("owner") != owner:
        raise ValueError("This optimization belongs to another session.")
    if time.time() - identity["created"] > RETENTION_SECONDS:
        raise ValueError("Optimization expired after 24 hours. Start a new run.")
    return folder


@contextmanager
def _launch_lock():
    lock = CACHE_ROOT / "launch.lock"
    deadline = time.monotonic() + 5
    while True:
        try:
            lock.mkdir()
            break
        except FileExistsError:
            if time.time() - lock.stat().st_mtime > 30:
                lock.rmdir()
                continue
            if time.monotonic() >= deadline:
                raise ValueError("Another optimization is starting. Try again shortly.")
            time.sleep(.05)
    try:
        yield
    finally:
        lock.rmdir()


def start_job(config, settings, owner):
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    with _launch_lock():
        active = 0
        for path in CACHE_ROOT.glob("*/identity.json"):
            try:
                identity = _read_mapping(path)
                expired = time.time() - identity["created"] > RETENTION_SECONDS
                status = (_read_mapping(path.parent / "status.json") if expired else
                          job_status({"id": path.parent.name}, identity["owner"]))
            except (ValueError, KeyError, OSError):
                # An interrupted server launch must not block later jobs.
                continue
            if status["state"] == "running":
                if identity["owner"] == owner:
                    raise ValueError("An optimization is already running in this session.")
                active += 1
            elif time.time() - identity["created"] > RETENTION_SECONDS:
                # Resolve and verify the exact cache child before removing it.
                import shutil
                target = path.parent.resolve()
                if target.parent == CACHE_ROOT.resolve() and not path.parent.is_symlink():
                    shutil.rmtree(target)
        if active >= MAX_ACTIVE_JOBS:
            raise ValueError("Two optimizations are running. Try again when one completes.")
        handle = {"id": uuid.uuid4().hex}
        folder = CACHE_ROOT / handle["id"]
        folder.mkdir()
        created = time.time()
        _write_json(folder / "identity.json", dict(owner=owner, created=created))
        _write_json(folder / "request.json", dict(config=flat_config(config), optimization=settings,
                                                 fingerprint=fingerprint(config)))
        _write_json(folder / "status.json", dict(state="running", message="Starting optimization worker…"))
        (folder / "heartbeat").touch()
        try:
            with (folder / "worker.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    [sys.executable, "-m", "turbodash.apps.turbine_optimization.jobs", str(folder / "request.json")],
                    cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            _PROCESSES[handle["id"]] = process
        except OSError as error:
            _write_json(folder / "status.json", dict(state="failed", message=str(error)))
            raise ValueError(f"Could not start optimization worker: {error}") from error
        return handle


def job_status(handle, owner):
    folder = job_directory(handle, owner)
    status = _read_mapping(folder / "status.json")
    process = _PROCESSES.get(handle["id"])
    if status["state"] == "running":
        exited = process is not None and process.poll() is not None
        stale = time.time() - (folder / "heartbeat").stat().st_mtime > 45
        if exited or stale:
            if stale and process is not None and process.poll() is None:
                process.kill()
            status = _read_mapping(folder / "status.json")
            if status["state"] == "running":
                status = dict(state="failed", message="Optimization worker stopped unexpectedly. Start a new run.")
                _write_json(folder / "status.json", status)
    if process is not None and process.poll() is not None:
        _PROCESSES.pop(handle["id"], None)
    return status


def read_job(handle, owner):
    folder = job_directory(handle, owner)
    status = job_status(handle, owner)
    log_path = folder / "solver.log"
    log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    # Bound browser payloads while keeping the full log in the local cache.
    if len(log) > 100000:
        log = "[Earlier output omitted]\n" + log[-100000:]
    progress_path = folder / "progress.json"
    progress = _read_mapping(progress_path) if progress_path.exists() else None
    result = _read_mapping(folder / "result.json") if status["state"] == "complete" else None
    if status["state"] == "failed":
        worker_log = folder / "worker.log"
        if worker_log.exists():
            log += worker_log.read_text(encoding="utf-8", errors="replace")[-10000:]
        log += "\nERROR: " + status["message"]
    return status, log, progress, result


def source_config(handle, owner):
    """Retrieve the submitted design for a failed application handoff."""
    folder = job_directory(handle, owner)
    return _read_mapping(folder / "request.json")["config"]


def _run_worker(request_path):
    from turbodash.optimization import optimize_turbine

    folder = Path(request_path).resolve().parent
    logger = logging.Logger(f"turbodash.optimization.{folder.name}", logging.INFO)
    handler = logging.FileHandler(folder / "solver.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    finished = threading.Event()

    def watch():
        started = time.monotonic()
        while not finished.wait(1):
            (folder / "heartbeat").touch()
            if time.monotonic() - started > TIMEOUT_SECONDS:
                logger.error("ERROR: Optimization exceeded the 10 minute limit.")
                _write_json(folder / "status.json", dict(state="failed", message="Optimization timed out after 10 minutes."))
                os._exit(1)

    threading.Thread(target=watch, daemon=True).start()
    try:
        request = _read_mapping(Path(request_path))
        logger.info("Optimizing turbine · scipy / SLSQP")
        logger.info("PSV Func-value is the signed, normalized minimized objective; progress below uses physical units.")

        def progress(snapshot):
            _write_json(folder / "progress.json", snapshot)
            logger.info("Update %d · physical objective %.8g · max violation %.3g",
                        snapshot["iteration"], snapshot["objective"], snapshot["max_constraint_violation"])

        result = optimize_turbine(request["config"], request["optimization"], callback=progress, logger=logger)
        payload = dict(config=result.config, turbine_results=result.turbine_results,
                       success=result.success, solver_success=result.solver_success, feasible=result.feasible,
                       message=result.message, objective=result.objective, initial_objective=result.initial_objective,
                       max_constraint_violation=result.max_constraint_violation, constraint_report=result.constraint_report,
                       history=result.history, fingerprint=request["fingerprint"])
        _write_json(folder / "result.json", payload)
        logger.info("%s · %s", "SUCCESS" if result.success else "NOT APPLIED", result.message)
        logger.info("Model evaluations: %d · elapsed %.2f s", result.n_evaluations, result.solver.elapsed_time)
        _write_json(folder / "status.json", dict(state="complete", message=result.message))
    except Exception as error:
        logger.exception("ERROR: %s", error)
        _write_json(folder / "status.json", dict(state="failed", message=str(error)))
    finally:
        finished.set()
        handler.close()
        logger.removeHandler(handler)


if __name__ == "__main__":
    _run_worker(sys.argv[1])
