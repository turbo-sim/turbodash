def launch_app():
    import os
    import webbrowser
    from threading import Timer
    from turbodash.app_turbine import app

    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        Timer(1, lambda: webbrowser.open("http://127.0.0.1:8050/")).start()

    app.run(debug=True)



launch_app()
