"""Flask web frontend for the eye detection system.

Run:  python app.py      then open http://127.0.0.1:5000
Localhost only. Debug mode and the reloader are OFF on purpose: the reloader would start a
second process that opens the webcam twice.
"""
import atexit
import os

from flask import Flask, Response, jsonify, render_template, request

from src.web_camera_service import DetectionService

HOST = "127.0.0.1"
PORT = 5000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALLOWED_HOSTS = {f"{HOST}:{PORT}", f"localhost:{PORT}"}


def create_app(service=None):
    app = Flask(__name__,
                template_folder=os.path.join(BASE_DIR, "web", "templates"),
                static_folder=os.path.join(BASE_DIR, "web", "static"))
    app.config["SERVICE"] = service if service is not None else DetectionService()
    svc = app.config["SERVICE"]

    @app.before_request
    def reject_foreign_requests():
        # Stops other websites (and DNS-rebinding tricks) from driving the camera.
        host = request.host.lower()
        if app.config.get("ENFORCE_HOST", True) and host not in ALLOWED_HOSTS:
            return jsonify(error="unexpected Host header"), 403
        origin = request.headers.get("Origin")
        if request.method == "POST" and origin and origin.split("://", 1)[-1].lower() != host:
            return jsonify(error="cross-origin request refused"), 403
        # Browsers say who started a request. Another website embedding the stream or
        # posting to the API is "cross-site"/"same-site"; our own page is "same-origin",
        # and typing the address gives "none". Requests without the header (curl, tests) pass.
        fetch_site = request.headers.get("Sec-Fetch-Site")
        if (fetch_site and fetch_site not in ("same-origin", "none")
                and (request.method == "POST" or request.path == "/video_feed")):
            return jsonify(error="cross-site request refused"), 403
        return None

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/api/start")
    def api_start():
        return jsonify(svc.start())

    @app.post("/api/stop")
    def api_stop():
        return jsonify(svc.stop())

    @app.get("/api/status")
    def api_status():
        return jsonify(svc.status())

    @app.get("/video_feed")
    def video_feed():
        generation = svc.generation()                # taken at request time: ONE worker only

        def generate():
            last_seq = None
            while svc.is_active() and svc.generation() == generation:
                svc.mark_viewer()                    # keeps the 30 s inactivity timer fresh
                item = svc.wait_for_frame(last_seq, timeout=1.0)
                if item is None or svc.generation() != generation:
                    continue
                last_seq, jpeg = item
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
        response = Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")
        response.headers["Cache-Control"] = "no-store"
        return response

    return app


def main():
    app = create_app()
    service = app.config["SERVICE"]
    atexit.register(service.stop)                    # release the camera on any normal exit
    try:
        app.run(host=HOST, port=PORT, debug=False, use_reloader=False, threaded=True)
    finally:
        service.stop()


if __name__ == "__main__":
    main()
