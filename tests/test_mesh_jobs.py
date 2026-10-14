"""Job lifecycle and artifact access checks without invoking native meshing."""

from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zipfile import ZipFile

import numpy as np
import pytest

from turbodash import mesh_jobs as jobs


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "CACHE_ROOT", tmp_path / "mesh_cache")
    monkeypatch.setattr(jobs, "_PROCESSES", {})


def make_job(*, state="complete", age=0):
    handle = {"id": "a" * 32, "fingerprint": "test-fingerprint"}
    folder = jobs.CACHE_ROOT / handle["id"]
    folder.mkdir(parents=True)
    jobs._write_json(folder / "identity.json", {**handle, "created": time.time() - age})
    jobs._write_json(folder / "status.json", {"state": state, "message": "Test job"})
    jobs._write_json(folder / "manifest.json", {"mesh_file": "mesh_3d.cgns"})
    return handle, folder


def test_fingerprint_is_stable_for_key_order_and_numpy_results():
    first = {"stages": [{"r": np.float64(0.5)}], "values": np.array([1., 2.])}
    second = {"values": [1., 2.], "stages": [{"r": 0.5}]}
    assert jobs.fingerprint(first) == jobs.fingerprint(second)
    assert jobs.fingerprint(first, row="rotor") != jobs.fingerprint(first)
    assert jobs.fingerprint(first, options={"span_layers": 4}) != jobs.fingerprint(first)


@pytest.mark.parametrize("handle", [None, {}, {"id": "../outside"}, {"id": "A" * 32}, {"id": "a" * 33}, {"id": int("1" * 32)}])
def test_opaque_handle_rejects_malformed_or_path_based_access(handle):
    with pytest.raises(ValueError, match="Invalid mesh job handle"):
        jobs.job_directory(handle)


def test_handle_requires_existing_identity_and_matching_fingerprint():
    handle, folder = make_job()
    assert jobs.job_directory(handle) == folder.resolve()
    with pytest.raises(ValueError, match="does not match"):
        jobs.job_directory({**handle, "fingerprint": "another-design"})
    (folder / "identity.json").unlink()
    with pytest.raises(ValueError, match="unavailable"):
        jobs.job_directory(handle)


def test_expired_handle_cannot_read_or_download_artifacts():
    handle, _ = make_job(age=jobs.RETENTION_SECONDS + 1)
    with pytest.raises(ValueError, match="expired"):
        jobs.job_directory(handle)
    with pytest.raises(ValueError, match="expired"):
        jobs.mesh_archive(handle)


def test_archive_contains_only_export_artifacts():
    handle, folder = make_job()
    for filename in ("request.json", "worker.log", "unfinished.tmp"):
        (folder / filename).write_text("PRIVATE", encoding="utf-8")
    (folder / "mesh_3d.cgns").write_bytes(b"CGNS-test")
    (folder / "geometry").mkdir()
    (folder / "geometry" / "blade.stl").write_text("solid blade\nendsolid", encoding="utf-8")
    with ZipFile(BytesIO(jobs.mesh_archive(handle))) as archive:
        assert set(archive.namelist()) == {"manifest.json", "mesh_3d.cgns", "geometry/blade.stl"}
        assert archive.read("mesh_3d.cgns") == b"CGNS-test"


def test_archive_does_not_include_symlinks(tmp_path):
    handle, folder = make_job()
    outside = tmp_path / "outside.txt"
    outside.write_text("PRIVATE", encoding="utf-8")
    try:
        (folder / "linked.txt").symlink_to(outside)
    except OSError:
        pytest.skip("Creating symlinks requires privileges on this Windows installation.")
    with ZipFile(BytesIO(jobs.mesh_archive(handle))) as archive:
        assert "linked.txt" not in archive.namelist()


@pytest.mark.parametrize("state", ["running", "failed"])
def test_archive_requires_completed_job(state):
    handle, _ = make_job(state=state)
    with pytest.raises(ValueError, match="successful mesh"):
        jobs.mesh_archive(handle)


def test_cleanup_removes_only_expired_terminal_folders():
    _, expired = make_job(age=jobs.RETENTION_SECONDS + 1)
    running = jobs.CACHE_ROOT / ("b" * 32)
    running.mkdir()
    jobs._write_json(running / "identity.json", {"created": time.time() - jobs.RETENTION_SECONDS - 1})
    jobs._write_json(running / "status.json", {"state": "running"})
    unrelated = jobs.CACHE_ROOT / "user-data"
    unrelated.mkdir()
    jobs._cleanup()
    assert not expired.exists()
    assert running.exists()
    assert unrelated.exists()


def test_worker_completion_publishes_manifest_and_preserves_progress(monkeypatch):
    from turbodash import meshing

    _, folder = make_job(state="running")
    request = {"results": {"solved": True}, "stage_index": 1, "row": "rotor", "options": {}}
    jobs._write_json(folder / "request.json", request)
    captured = {}

    def generate(results, output, stage_index, row, *, options, progress):
        captured.update(results=results, output=output, stage_index=stage_index, row=row, options=options)
        progress("Sweeping section", 0.5)
        assert json.loads((output / "status.json").read_text(encoding="utf-8"))["fraction"] == 0.5
        return {"mesh_file": "volume.cgns", "cells": 42}

    monkeypatch.setattr(meshing, "generate_mesh", generate)
    jobs._run_worker(folder / "request.json")
    assert captured == {"results": {"solved": True}, "output": folder.resolve(),
                        "stage_index": 1, "row": "rotor", "options": {}}
    assert json.loads((folder / "status.json").read_text(encoding="utf-8"))["state"] == "complete"
    assert json.loads((folder / "manifest.json").read_text(encoding="utf-8"))["cells"] == 42


def test_worker_failure_is_published(monkeypatch):
    from turbodash import meshing

    handle, folder = make_job(state="running")
    jobs._write_json(folder / "request.json", {"results": {}, "stage_index": 0, "row": "stator", "options": {}})

    def fail(*args, **kwargs):
        raise ValueError("Invalid Jacobian in generated cells")

    monkeypatch.setattr(meshing, "generate_mesh", fail)
    jobs._run_worker(folder / "request.json")
    status = jobs.job_status(handle)
    assert status["state"] == "failed"
    assert "Invalid Jacobian" in status["message"]


def test_dead_native_worker_is_reported_as_failed():
    handle, folder = make_job(state="running")
    process = subprocess.Popen(
        [sys.executable, "-c", "raise SystemExit(1)"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    process.wait(timeout=15)
    jobs._PROCESSES[handle["id"]] = process
    assert jobs.job_status(handle)["state"] == "failed"
    assert handle["id"] not in jobs._PROCESSES
    # Another web process has no Popen handle and must see the durable failure.
    assert jobs.job_status(handle)["state"] == "failed"
    assert json.loads((folder / "status.json").read_text(encoding="utf-8"))["state"] == "failed"


def test_start_validates_before_allocating_worker(monkeypatch):
    from turbodash import meshing

    def reject(*args, **kwargs):
        raise ValueError("Unsupported radial turbine")

    monkeypatch.setattr(meshing, "build_blade_row", reject)
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Invalid request started a worker"))
    with pytest.raises(ValueError, match="Unsupported radial"):
        jobs.start_job({})
    assert not jobs.CACHE_ROOT.exists()


def test_admission_counts_jobs_from_other_server_processes(monkeypatch):
    from turbodash import meshing

    handle, _ = make_job(state="running")
    folder = jobs.CACHE_ROOT / ("b" * 32)
    folder.mkdir()
    jobs._write_json(folder / "identity.json", {**handle, "id": folder.name, "created": time.time()})
    jobs._write_json(folder / "status.json", {"state": "running"})
    monkeypatch.setattr(meshing, "build_blade_row", lambda *args, **kwargs: None)
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Server exceeded concurrent job limit"))
    with pytest.raises(ValueError, match="already running"):
        jobs.start_job({})
    assert not (jobs.CACHE_ROOT / "launch.lock").exists()


def test_cancellation_signal_is_shared_without_local_process_handle():
    handle, folder = make_job(state="running")
    assert not jobs._PROCESSES
    jobs.cancel_job(handle)
    assert (folder / "cancel").is_file()


def test_launch_lock_is_released_after_admission_failure():
    jobs.CACHE_ROOT.mkdir()
    with pytest.raises(RuntimeError, match="admission failed"):
        with jobs._launch_lock():
            assert (jobs.CACHE_ROOT / "launch.lock").is_dir()
            raise RuntimeError("admission failed")
    assert not (jobs.CACHE_ROOT / "launch.lock").exists()


def test_stale_launch_lock_recovers_after_server_restart():
    lock = jobs.CACHE_ROOT / "launch.lock"
    lock.mkdir(parents=True)
    old = time.time() - 121
    os.utime(lock, (old, old))
    with jobs._launch_lock():
        assert lock.is_dir()
        assert lock.stat().st_mtime > old
    assert not lock.exists()


def test_active_launch_lock_rejects_concurrent_admission(monkeypatch):
    lock = jobs.CACHE_ROOT / "launch.lock"
    lock.mkdir(parents=True)
    times = iter([0., 6.])
    monkeypatch.setattr(jobs.time, "monotonic", lambda: next(times))
    with pytest.raises(ValueError, match="Another mesh job is starting"):
        with jobs._launch_lock():
            pytest.fail("Concurrent admission acquired an active lock")
    assert lock.is_dir()


@pytest.mark.parametrize("identity", [
    None, [], {}, {"fingerprint": "test-fingerprint"},
    {"fingerprint": [], "created": 1}, {"fingerprint": "", "created": 1},
    {"fingerprint": "test-fingerprint", "created": "invalid"},
    {"fingerprint": "test-fingerprint", "created": True},
    {"fingerprint": "test-fingerprint", "created": float("nan")},
    {"fingerprint": "test-fingerprint", "created": float("inf")},
    {"fingerprint": "test-fingerprint", "created": 10 ** 1000},
])
def test_corrupted_identity_is_unavailable(identity):
    handle, folder = make_job()
    jobs._write_json(folder / "identity.json", identity)
    with pytest.raises(ValueError, match="unavailable"):
        jobs.job_directory(handle)


@pytest.mark.parametrize("status", [None, [], {}, {"state": "unknown"}, {"state": []}])
def test_corrupted_status_is_unavailable_and_dead_worker_is_reaped(status):
    from types import SimpleNamespace

    handle, folder = make_job()
    jobs._write_json(folder / "status.json", status)
    with pytest.raises(ValueError, match="unavailable"):
        jobs.job_status(handle)
    jobs._PROCESSES[handle["id"]] = SimpleNamespace(poll=lambda: 1)
    jobs._reap()
    assert handle["id"] not in jobs._PROCESSES


@pytest.mark.parametrize("filename,value", [
    ("identity.json", {}), ("identity.json", []),
    ("status.json", {}), ("status.json", []), ("manifest.json", []),
])
def test_corrupt_cache_entry_does_not_block_new_jobs(monkeypatch, filename, value):
    from turbodash import meshing

    _, folder = make_job()
    jobs._write_json(folder / filename, value)
    monkeypatch.setattr(meshing, "build_blade_row", lambda *args, **kwargs: None)
    monkeypatch.setattr(jobs, "_start_job", lambda *args: {"id": "new-job"})
    assert jobs.start_job({}) == {"id": "new-job"}
    assert not (jobs.CACHE_ROOT / "launch.lock").exists()
