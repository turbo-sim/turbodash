"""Command-line launch options shared by the independent Dash applications."""

import argparse
import os
import threading
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener
from uuid import uuid4
import webbrowser


_READY_HEADER = "X-Turbodash-Browser-Ready"


def run_cli(main, argv=None):
    """Parse launch options without consuming arguments in Python integrations."""
    parser = argparse.ArgumentParser(description="Start this Turbodash application.")
    parser.add_argument(
        "--open-browser", action="store_true",
        help="Open the app in your default browser once its server is ready.",
    )
    options = parser.parse_args(argv)
    return main(open_browser=options.open_browser)


def _open_when_ready(probe_url, browser_url, token, stopped, logger, timeout=30):
    """Wait for this server's layout, ignoring proxies and unrelated listeners."""
    opener = build_opener(ProxyHandler({}))
    request = Request(probe_url, headers={_READY_HEADER: token})
    deadline = time.monotonic() + timeout
    while not stopped.is_set() and time.monotonic() < deadline:
        ready = False
        try:
            with opener.open(request, timeout=.5) as response:
                ready = response.status == 200 and response.headers.get(_READY_HEADER) == token
        except (URLError, OSError):
            pass
        if ready and not stopped.is_set():
            try:
                opened = webbrowser.open(browser_url)
            except (webbrowser.Error, OSError):
                opened = False
            if not opened:
                logger.warning("Could not open a browser. Visit %s", browser_url)
            return
        stopped.wait(.1)
    if not stopped.is_set():
        logger.warning("Browser launch timed out waiting for the app. Visit %s", browser_url)


def run_app(app, *, open_browser=False):
    """Serve normally, optionally opening one browser tab after a ready response."""
    if not open_browser:
        return app.run(debug=False)

    host = os.environ.get("HOST") or "127.0.0.1"
    port = int(os.environ.get("PORT") or 8050)
    browser_host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(host, host).strip("[]")
    if ":" in browser_host:
        browser_host = f"[{browser_host}]"
    base_url = f"http://{browser_host}:{port}"
    browser_url = base_url + app.config.requests_pathname_prefix
    probe_url = base_url + app.config.routes_pathname_prefix + "_dash-layout"
    token = uuid4().hex
    stopped = threading.Event()
    original_wsgi = app.server.wsgi_app

    def identify_server(environ, start_response):
        # Only our own readiness request gets the marker. A different server
        # already listening on this port must never trigger a browser launch.
        if environ.get("HTTP_X_TURBODASH_BROWSER_READY") == token:
            def marked_response(status, headers, exc_info=None):
                return start_response(status, [*headers, (_READY_HEADER, token)], exc_info)

            return original_wsgi(environ, marked_response)
        return original_wsgi(environ, start_response)

    worker = threading.Thread(
        target=_open_when_ready,
        args=(probe_url, browser_url, token, stopped, app.server.logger),
        daemon=True,
    )
    app.server.wsgi_app = identify_server
    try:
        worker.start()
        return app.run(debug=False, host=host, port=port)
    finally:
        stopped.set()
        if worker.ident is not None:
            worker.join(timeout=1)
        app.server.wsgi_app = original_wsgi
