"""Optional browser opening waits for the launched app and preserves server startup."""

from contextlib import nullcontext
import threading
from types import SimpleNamespace
from unittest.mock import Mock
import webbrowser

from dash import Dash, html
from flask import request
import pytest
from werkzeug.serving import make_server

from turbodash.apps.shared import cli


def test_plain_launch_does_not_start_a_browser_worker(monkeypatch):
    app = SimpleNamespace(run=Mock())
    worker = Mock()
    browser = Mock()
    monkeypatch.setattr(cli.threading, "Thread", worker)
    monkeypatch.setattr(webbrowser, "open", browser)

    cli.run_app(app)

    app.run.assert_called_once_with(debug=False)
    worker.assert_not_called()
    browser.assert_not_called()


def test_browser_opens_once_after_this_apps_layout_is_ready(monkeypatch):
    app = Dash(__name__, routes_pathname_prefix="/dashboard/")
    app.layout = html.Div("Ready")
    attempts = []
    opened = threading.Event()

    @app.server.before_request
    def temporarily_unavailable():
        if request.path == "/dashboard/_dash-layout":
            attempts.append(request.path)
            if len(attempts) == 1:
                return "Starting", 503

    server = make_server("127.0.0.1", 0, app.server)
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    monkeypatch.setenv("HOST", "0.0.0.0")
    monkeypatch.setenv("PORT", str(server.server_port))
    original_wsgi = app.server.wsgi_app

    def open_browser(url):
        assert len(attempts) >= 2
        opened.set()
        return True

    browser = Mock(side_effect=open_browser)
    monkeypatch.setattr(webbrowser, "open", browser)

    def run(**options):
        assert options == dict(debug=False, host="0.0.0.0", port=server.server_port)
        assert not opened.is_set()
        serving.start()
        assert opened.wait(5), "Browser did not open after the layout became ready"

    monkeypatch.setattr(app, "run", run)
    try:
        cli.run_app(app, open_browser=True)
    finally:
        if serving.ident is not None:
            server.shutdown()
            serving.join(timeout=5)
        server.server_close()

    browser.assert_called_once_with(f"http://127.0.0.1:{server.server_port}/dashboard/")
    assert app.server.wsgi_app == original_wsgi
    assert cli._READY_HEADER not in app.server.test_client().get("/dashboard/_dash-layout").headers


@pytest.mark.parametrize("status,headers", [(200, {}), (503, {cli._READY_HEADER: "token"})])
def test_browser_ignores_another_server_or_an_unhealthy_layout(monkeypatch, status, headers):
    response = SimpleNamespace(status=status, headers=headers)
    opener = Mock()
    opener.open.side_effect = lambda *_args, **_kwargs: nullcontext(response)
    monkeypatch.setattr(cli, "build_opener", lambda *_args: opener)
    browser = Mock()
    monkeypatch.setattr(webbrowser, "open", browser)
    logger = Mock()

    cli._open_when_ready("http://127.0.0.1/probe", "http://127.0.0.1/", "token",
                         threading.Event(), logger, timeout=.01)

    browser.assert_not_called()
    logger.warning.assert_called_once()


@pytest.mark.parametrize("error", [None, webbrowser.Error("Unavailable"), OSError("Unavailable")])
def test_unavailable_browser_does_not_retry_or_stop_the_server(monkeypatch, error):
    response = SimpleNamespace(status=200, headers={cli._READY_HEADER: "token"})
    opener = Mock()
    opener.open.side_effect = lambda *_args, **_kwargs: nullcontext(response)
    monkeypatch.setattr(cli, "build_opener", lambda *_args: opener)
    browser = Mock(return_value=False, side_effect=error)
    monkeypatch.setattr(webbrowser, "open", browser)
    logger = Mock()

    cli._open_when_ready("http://127.0.0.1/probe", "http://127.0.0.1/", "token",
                         threading.Event(), logger)

    browser.assert_called_once_with("http://127.0.0.1/")
    logger.warning.assert_called_once()


def test_failed_server_start_cancels_browser_worker_and_restores_wsgi(monkeypatch):
    app = Dash(__name__)
    app.layout = html.Div("Ready")
    original_wsgi = app.server.wsgi_app
    cancelled = threading.Event()

    def wait_for_cancellation(_probe, _url, _token, stopped, _logger):
        if stopped.wait(2):
            cancelled.set()

    monkeypatch.setattr(cli, "_open_when_ready", wait_for_cancellation)
    monkeypatch.setattr(app, "run", Mock(side_effect=OSError("Port in use")))
    with pytest.raises(OSError, match="Port in use"):
        cli.run_app(app, open_browser=True)

    assert cancelled.is_set()
    assert app.server.wsgi_app == original_wsgi


def test_unknown_command_line_option_does_not_start_a_server():
    main = Mock()
    with pytest.raises(SystemExit) as error:
        cli.run_cli(main, ["--unknown-option"])
    assert error.value.code == 2
    main.assert_not_called()
