"""Browser-opening Python launchers; app imports remain lazy."""


def launch_app_stage():
    import os
    import webbrowser
    from threading import Timer
    from turbodash.apps.stage import app

    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        Timer(1, lambda: webbrowser.open("http://127.0.0.1:8050/")).start()

    app.run(debug=True)



def launch_app_turbine():
    import os
    import webbrowser
    from threading import Timer
    from turbodash.apps.turbine import app

    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        Timer(1, lambda: webbrowser.open("http://127.0.0.1:8050/")).start()

    app.run(debug=True)


def launch_app_experimental():
    """Launch the experimental meanline, geometry and meshing workspace."""
    import os
    import webbrowser
    from threading import Timer
    from turbodash.apps.turbine_experimental import app

    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        Timer(1, lambda: webbrowser.open("http://127.0.0.1:8050/")).start()

    app.run(debug=True)
