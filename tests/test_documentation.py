"""Exercise documentation delivery through the same server as the app."""

from pathlib import Path
import runpy

from dash import Dash, html
import pytest

from turbodash.apps.shared.documentation import register_documentation


def test_derivation_pages_preserve_complete_original_notes_and_figure():
    check = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "docs" / "development" / "verify_source_fidelity.py")
    )
    check["verify"]()


@pytest.fixture
def documentation_build(tmp_path):
    directory = tmp_path / "docs" / "_build" / "html"
    directory.mkdir(parents=True)
    (directory / "index.html").write_text("<h1>Turbodash documentation</h1>", encoding="utf-8")
    (directory / "_static").mkdir()
    (directory / "_static" / "theme.css").write_text("body { color: navy; }", encoding="utf-8")
    (directory / "searchindex.js").write_text("Search.setIndex({});", encoding="utf-8")
    (directory / ".doctrees").mkdir()
    (directory / ".doctrees" / "environment.pickle").write_bytes(b"private build state")
    (directory / "theory").mkdir()
    (directory / "theory" / "index.html").write_text("<h1>Meanline theory</h1>", encoding="utf-8")
    # These source files must never be exposed by the documentation route.
    (tmp_path / "docs" / "conf.py").write_text("private documentation settings", encoding="utf-8")
    (tmp_path / "app.py").write_text("private application source", encoding="utf-8")
    return directory


def make_app(routes_prefix="/", requests_prefix="/"):
    app = Dash(
        __name__,
        routes_pathname_prefix=routes_prefix,
        requests_pathname_prefix=requests_prefix,
    )
    app.layout = html.Div("Calculations")
    return app


@pytest.mark.parametrize(
    "routes_prefix, requests_prefix",
    [("/", "/"), ("/calc/", "/calc/"), ("/", "/proxy/"), ("/calc/", "/proxy/calc/")],
)
def test_documentation_respects_server_and_browser_prefixes(
    documentation_build, routes_prefix, requests_prefix,
):
    app = make_app(routes_prefix, requests_prefix)
    url = register_documentation(app, documentation_build)
    client = app.server.test_client()

    assert url == requests_prefix + "docs/"
    redirect = client.get(routes_prefix + "docs")
    assert redirect.status_code == 308
    assert redirect.headers["Location"] == url
    response = client.get(routes_prefix + "docs/")
    assert response.status_code == 200
    assert response.mimetype == "text/html"
    assert response.get_data(as_text=True) == "<h1>Turbodash documentation</h1>"
    assert client.get(routes_prefix + "_dash-layout").status_code == 200


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("_static/theme.css", "body { color: navy; }"),
        ("searchindex.js", "Search.setIndex({});"),
        ("theory/index.html", "<h1>Meanline theory</h1>"),
        ("theory/", "<h1>Meanline theory</h1>"),
    ],
)
def test_documentation_serves_nested_assets_and_pages(documentation_build, filename, expected):
    app = make_app()
    register_documentation(app, documentation_build)
    response = app.server.test_client().get("/docs/" + filename)

    assert response.status_code == 200
    assert response.get_data(as_text=True) == expected


@pytest.mark.parametrize(
    "filename",
    [
        "missing.html",
        "conf.py",
        "../../conf.py",
        "%2e%2e/%2e%2e/conf.py",
        "../../../app.py",
        "..%5c..%5cconf.py",
        "C:%5cWindows%5cwin.ini",
        ".doctrees/environment.pickle",
    ],
)
def test_documentation_does_not_expose_missing_files_or_sources(documentation_build, filename):
    app = make_app()
    register_documentation(app, documentation_build)
    response = app.server.test_client().get("/docs/" + filename)

    assert response.status_code == 404
    assert b"private" not in response.data


def test_documentation_rejects_symlinks_outside_build(documentation_build):
    link = documentation_build / "linked.html"
    try:
        link.symlink_to(documentation_build.parents[2] / "app.py")
    except OSError as error:
        pytest.skip(f"Creating symlinks is unavailable: {error}")
    app = make_app()
    register_documentation(app, documentation_build)

    response = app.server.test_client().get("/docs/linked.html")

    assert response.status_code == 404
    assert b"private" not in response.data


def test_missing_documentation_build_explains_build_and_can_be_built_later(tmp_path):
    app = make_app("/calc/", "/proxy/calc/")
    directory = tmp_path / "html"
    register_documentation(app, directory)
    client = app.server.test_client()

    response = client.get("/calc/docs/")

    assert response.status_code == 503
    assert response.mimetype == "text/html"
    assert "docs/build_docs.py" in response.get_data(as_text=True)
    assert "Return to calculations" not in response.get_data(as_text=True)
    assert not directory.exists()
    assert client.get("/calc/docs/../app.py").status_code == 404

    directory.mkdir()
    (directory / "index.html").write_text("<h1>Now available</h1>", encoding="utf-8")
    assert client.get("/calc/docs/").get_data(as_text=True) == "<h1>Now available</h1>"


def test_documentation_build_can_be_configured_by_environment(documentation_build, monkeypatch):
    monkeypatch.setenv("TURBODASH_DOCS_DIR", str(documentation_build))
    app = make_app()
    register_documentation(app)

    assert app.server.test_client().get("/docs/").status_code == 200


def test_explicit_documentation_build_takes_precedence(documentation_build, tmp_path, monkeypatch):
    monkeypatch.setenv("TURBODASH_DOCS_DIR", str(tmp_path / "missing"))
    app = make_app()
    register_documentation(app, documentation_build)

    assert app.server.test_client().get("/docs/").status_code == 200
