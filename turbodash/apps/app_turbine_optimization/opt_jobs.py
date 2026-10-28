"""Run optimizations in background processes and deliver their results to the app.

Each run belongs to a browser session and receives a saved copy of its turbine
inputs and optimization settings. A separate Python process calls
``turbodash.optimization.optimize_turbine``, keeping the web interface responsive
while publishing the PySolverView frontend's convergence log, physical progress
snapshots, and the final result. The worker logger receives the frontend's
convergence history and cached final optimization report directly; the app
does not format a separate numerical report.

This module manages session ownership, concurrent-run limits, timeouts, and
expired jobs. Complete JSON snapshots are published through atomic file
replacement so that web requests can read progress while a worker is running.
The local file cache supports multiple web-server workers on the same machine.
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
from .opt_model import flat_config

CACHE_ROOT = Path(tempfile.gettempdir()) / "turbodash_optimization_jobs"
TIMEOUT_SECONDS = 600
RETENTION_SECONDS = 24 * 3600
MAX_ACTIVE_JOBS = 2
OPTIMIZATION_REPORT_WIDTH = 120
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
            time.sleep(0.02)


def _read_mapping(path):
    """Read one complete snapshot, allowing a brief Windows replacement lock."""
    try:
        for attempt in range(20):
            try:
                contents = path.read_text(encoding="utf-8")
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.02)
        value = json.loads(contents)
        if isinstance(value, dict):
            return value
    except (ValueError, OSError) as error:
        raise ValueError(
            "Optimization data is unavailable. Start a new run."
        ) from error
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
    server.config.update(
        SECRET_KEY=os.environ.get("TURBODASH_SECRET_KEY") or key_path.read_bytes(),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
    )


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

    payload = json.dumps(
        canonical(flat_config(config)["inputs"]), sort_keys=True, separators=(",", ":")
    )
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
            time.sleep(0.05)
    try:
        yield
    finally:
        lock.rmdir()


def start_job(config, settings, owner):
    optimizer_identity = _optimizer_identity(settings)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    with _launch_lock():
        active = 0
        for path in CACHE_ROOT.glob("*/identity.json"):
            try:
                identity = _read_mapping(path)
                expired = time.time() - identity["created"] > RETENTION_SECONDS
                status = (
                    _read_mapping(path.parent / "status.json")
                    if expired
                    else job_status({"id": path.parent.name}, identity["owner"])
                )
            except (ValueError, KeyError, OSError):
                # An interrupted server launch must not block later jobs.
                continue
            if status["state"] == "running":
                if identity["owner"] == owner:
                    raise ValueError(
                        "An optimization is already running in this session."
                    )
                active += 1
            elif time.time() - identity["created"] > RETENTION_SECONDS:
                # Resolve and verify the exact cache child before removing it.
                import shutil

                target = path.parent.resolve()
                if (
                    target.parent == CACHE_ROOT.resolve()
                    and not path.parent.is_symlink()
                ):
                    shutil.rmtree(target)
        if active >= MAX_ACTIVE_JOBS:
            raise ValueError(
                "Two optimizations are running. Try again when one completes."
            )
        handle = {"id": uuid.uuid4().hex, **optimizer_identity}
        folder = CACHE_ROOT / handle["id"]
        folder.mkdir()
        created = time.time()
        _write_json(folder / "identity.json", dict(owner=owner, created=created))
        _write_json(
            folder / "request.json",
            dict(
                config=flat_config(config),
                optimization=settings,
                fingerprint=fingerprint(config),
            ),
        )
        _write_json(
            folder / "status.json",
            dict(state="running", message="Starting optimization worker…", **optimizer_identity),
        )
        (folder / "heartbeat").touch()
        try:
            with (folder / "worker.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "turbodash.apps.app_turbine_optimization.opt_jobs",
                        str(folder / "request.json"),
                    ],
                    cwd=PROJECT_ROOT,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
            _PROCESSES[handle["id"]] = process
        except OSError as error:
            _write_json(
                folder / "status.json", dict(state="failed", message=str(error))
            )
            raise ValueError(f"Could not start optimization worker: {error}") from error
        return handle


def job_status(handle, owner):
    folder = job_directory(handle, owner)
    status = _read_mapping(folder / "status.json")
    if (folder / "cancel").exists():
        # The marker is authoritative even if completion raced with the click.
        status = dict(
            state="cancelled",
            message="Optimization stopped by user. Current design retained.",
        )
    process = _PROCESSES.get(handle["id"])
    if status["state"] == "running":
        exited = process is not None and process.poll() is not None
        stale = time.time() - (folder / "heartbeat").stat().st_mtime > 45
        if exited or stale:
            if stale and process is not None and process.poll() is None:
                process.kill()
            status = _read_mapping(folder / "status.json")
            if status["state"] == "running":
                status = dict(
                    state="failed",
                    message="Optimization worker stopped unexpectedly. Start a new run.",
                )
                _write_json(folder / "status.json", status)
    if process is not None and process.poll() is not None:
        _PROCESSES.pop(handle["id"], None)
    return status


def cancel_job(handle, owner):
    """Cancel only an owned run, including across web-server workers."""
    folder = job_directory(handle, owner)
    if job_status(handle, owner)["state"] != "running":
        return
    (folder / "cancel").touch()
    process = _PROCESSES.get(handle["id"])
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)
    # Another server process may own Popen; the worker's watchdog reads cancel.
    _write_json(
        folder / "status.json",
        dict(
            state="cancelled",
            message="Optimization stopped by user. Current design retained.",
        ),
    )


def read_job(handle, owner):
    folder = job_directory(handle, owner)
    status = job_status(handle, owner)
    log_path = folder / "solver.log"
    log = (
        log_path.read_text(encoding="utf-8", errors="replace")
        if log_path.exists()
        else ""
    )
    # Bound browser payloads while keeping the full log in the local cache.
    if len(log) > 100000:
        log = "[Earlier output omitted]\n" + log[-100000:]
    progress_path = folder / "progress.json"
    progress = _read_mapping(progress_path) if progress_path.exists() else None
    result = (
        _read_mapping(folder / "result.json") if status["state"] == "complete" else None
    )
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


def _optimizer_identity(settings):
    """Return the selected solver identity without changing legacy YAML defaults."""
    from turbodash.optimization import OPTIMIZER_LABELS, normalize_optimizer

    solver = normalize_optimizer(settings.get("solver"))
    return dict(solver=solver, solver_label=OPTIMIZER_LABELS[solver])


def _write_final_report(result, logger):
    """Log detached final data using PySolverView's pure table formatters.

    Custom SLSQP already has a cached report path. Other frontend report paths
    currently perform fresh assessments with a common native-derived tolerance;
    Pounce instead renders its authoritative returned-point certificate with the
    exact feasibility and optimality limits used to accept the run.
    """
    frontend = result.optimizer
    if frontend.library == "pysolver":
        frontend.print_optimization_report(
            include_design_variables=True,
            include_constraints=True,
            include_kkt_conditions=True,
            include_multipliers=True,
            to_console=False,
            width=OPTIMIZATION_REPORT_WIDTH,
        )
        return

    from pysolver_view.reporting import tables

    final = result.frontend_result
    assessment = final.get("kkt") or {}
    options = final.get("effective_options") or {}
    feasibility = assessment.get(
        "feasibility_tol", options.get("feasibility_tol", 1e-6)
    )
    optimality = assessment.get("optimality_tol", options.get("optimality_tol", 1e-6))
    enabled = bool(final.get("kkt_enabled", True))
    names = frontend.problem.variable_names
    bounds = frontend.problem.get_bounds_normalized()
    constraints = [
        dict(
            name=item["variable"].removeprefix("$")
            + ("" if item["index"] is None else f"[{item['index']}]"),
            type=(
                "="
                if item["type"] in ("=", "==")
                else ">" if item["type"] in (">", ">=") else "<"
            ),
            value=item["value"],
            target=item["target"],
            satisfied=item["normalized_violation"] <= feasibility,
        )
        for item in result.constraint_report
    ]
    sections = [
        tables.final_kkt_report(
            assessment,
            feasibility,
            optimality,
            enabled=enabled,
            width=OPTIMIZATION_REPORT_WIDTH,
        ),
        tables.variables_report(
            final.x_solver,
            bounds,
            names,
            normalized=True,
            width=OPTIMIZATION_REPORT_WIDTH,
        ),
        tables.variables_report(
            final.x,
            frontend.problem.get_bounds(),
            names,
            normalized=False,
            width=OPTIMIZATION_REPORT_WIDTH,
        ),
        tables.constraint_report(constraints, width=OPTIMIZATION_REPORT_WIDTH),
        tables.final_multipliers_report(
            assessment,
            constraints,
            names,
            bounds,
            enabled=enabled,
            width=OPTIMIZATION_REPORT_WIDTH,
        ),
    ]
    logger.info(
        "Native status: %s; verified user-KKT stop: %s",
        final.status,
        bool(final.get("stopped_on_kkt", False)),
    )
    logger.info("\n%s", "\n".join(sections))


def _run_worker(request_path):
    from turbodash.optimization import optimize_turbine, solver_diagnostics

    folder = Path(request_path).resolve().parent
    logger = logging.Logger(f"turbodash.optimization.{folder.name}", logging.INFO)
    handler = logging.FileHandler(folder / "solver.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    finished = threading.Event()
    identity = {}

    def watch():
        started = time.monotonic()
        while not finished.wait(1):
            (folder / "heartbeat").touch()
            if (folder / "cancel").exists():
                os._exit(0)
            if time.monotonic() - started > TIMEOUT_SECONDS:
                logger.error("ERROR: Optimization exceeded the 10 minute limit.")
                _write_json(
                    folder / "status.json",
                    dict(
                        state="failed",
                        message="Optimization timed out after 10 minutes.",
                    ),
                )
                os._exit(1)

    threading.Thread(target=watch, daemon=True).start()
    try:
        request = _read_mapping(Path(request_path))
        identity = _optimizer_identity(request["optimization"])
        logger.info("Optimizer: %s", identity["solver_label"])

        def progress(snapshot):
            _write_json(folder / "progress.json", dict(snapshot, **identity))

        result = optimize_turbine(
            request["config"], request["optimization"], callback=progress, logger=logger
        )
        # PySolverView renders its cached final assessment into the same logger
        # as the convergence history, including completed unsuccessful solves.
        # Finish the report before publishing completion to the polling client.
        _write_final_report(result, logger)
        payload = dict(
            config=result.config,
            turbine_results=result.turbine_results,
            success=result.success,
            solver_success=result.solver_success,
            feasible=result.feasible,
            message=result.message,
            objective=result.objective,
            initial_objective=result.initial_objective,
            max_constraint_violation=result.max_constraint_violation,
            constraint_report=result.constraint_report,
            history=result.history,
            fingerprint=request["fingerprint"],
            **identity,
            solver_status=result.solver.status,
            stopped_on_kkt=bool(result.frontend_result.get("stopped_on_kkt", False)),
            termination=result.frontend_result.get("termination", "native"),
            **solver_diagnostics(result.solver),
            domain_rejections=result.solver.get("domain_rejections", 0),
            derivative_rejections=result.solver.get("derivative_rejections", 0),
            recovery_events=result.solver.get("events", []),
        )
        _write_json(folder / "result.json", payload)
        _write_json(
            folder / "status.json",
            dict(state="complete", message=result.message, **identity),
        )
    except Exception as error:
        logger.exception("ERROR: %s", error)
        failure = error.to_dict() if hasattr(error, "to_dict") else None
        _write_json(
            folder / "status.json",
            dict(state="failed", message=str(error), failure=failure, **identity),
        )
    finally:
        finished.set()
        handler.close()
        logger.removeHandler(handler)


if __name__ == "__main__":
    _run_worker(sys.argv[1])
