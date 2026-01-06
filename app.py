from flask import Flask, render_template, request, jsonify, send_from_directory
from ultralytics import YOLO
import cv2
import time
import os
import threading
import mediapipe as mp
from datetime import datetime

app = Flask(__name__)

# ---------------- MODEL ----------------
model = YOLO("yolov8n.pt")

# ---------------- MEDIAPIPE ----------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

# ---------------- GLOBAL STATE ----------------
system_status = {
    "camera": "OFF",
    "mode": "NONE",
    "person_count": 0,
    "start_time": None,
    "paused": False,
    "alert": "",
    "alert_id": 0
}

stop_signal = False
COOLDOWN_SECONDS = 10
AUTO_STOP_MINUTES = 2
last_saved_time = 0

EVIDENCE_DIR = "evidence"
os.makedirs(EVIDENCE_DIR, exist_ok=True)


def set_alert(message):
    system_status["alert"] = message
    system_status["alert_id"] += 1


def camera_process(mode):
    global stop_signal, last_saved_time

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    system_status.update({
        "camera": "ON",
        "mode": mode.upper(),
        "start_time": time.time(),
        "paused": False
    })
    set_alert("Camera started")

    while not stop_signal:
        if system_status["paused"]:
            time.sleep(0.1)
            continue

        ret, frame = cap.read()
        if not ret:
            break

        # -------- AUTO STOP --------
        if time.time() - system_status["start_time"] > AUTO_STOP_MINUTES * 60:
            set_alert("Camera auto-stopped")
            break

        # -------- YOLO --------
        if mode == "person":
            results = model(frame, conf=0.5, classes=[0], verbose=False)
        else:
            results = model(frame, conf=0.5, classes=list(range(1, 80)), verbose=False)

        annotated = frame.copy()
        person_count = 0

        for box in results[0].boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label == "person":
                person_count += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )

        system_status["person_count"] = person_count

        # -------- CROWD EVIDENCE --------
        if mode == "person" and person_count >= 2:
            set_alert(f"Crowd detected ({person_count})")
            if time.time() - last_saved_time > COOLDOWN_SECONDS:
                ts = int(time.time())
                filename = f"crowd_{ts}.jpg"
                cv2.imwrite(os.path.join(EVIDENCE_DIR, filename), frame)
                last_saved_time = time.time()

        # -------- HEAD TURN --------
        if mode == "person":
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_results = face_mesh.process(rgb)
            if face_results.multi_face_landmarks:
                nose = face_results.multi_face_landmarks[0].landmark[1]
                if nose.x < 0.35 or nose.x > 0.65:
                    set_alert("Focus on Camera")
                    cv2.putText(
                        annotated,
                        "FOCUS ON CAMERA",
                        (150, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 0, 255),
                        4
                    )

        cv2.imshow("AI Surveillance System", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    system_status.update({
        "camera": "OFF",
        "mode": "NONE",
        "person_count": 0,
        "start_time": None,
        "paused": False
    })
    set_alert("Camera stopped")


# ---------------- ROUTES ----------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/start", methods=["POST"])
def start():
    global stop_signal
    stop_signal = False
    mode = request.form["mode"]
    threading.Thread(target=camera_process, args=(mode,), daemon=True).start()
    return "OK"


@app.route("/pause")
def pause():
    system_status["paused"] = True
    set_alert("Detection paused")
    return "OK"


@app.route("/resume")
def resume():
    system_status["paused"] = False
    set_alert("Detection resumed")
    return "OK"


@app.route("/stop")
def stop():
    global stop_signal
    stop_signal = True
    set_alert("Camera stopped")
    return "OK"


@app.route("/status")
def status():
    runtime = 0
    if system_status["start_time"]:
        runtime = int(time.time() - system_status["start_time"])

    return jsonify({
        "camera": system_status["camera"],
        "mode": system_status["mode"],
        "person_count": system_status["person_count"],
        "runtime": runtime,
        "alert": system_status["alert"],
        "alert_id": system_status["alert_id"]
    })


@app.route("/evidence/<path:filename>")
def evidence_file(filename):
    return send_from_directory(EVIDENCE_DIR, filename)


@app.route("/evidence")
def evidence():
    evidence_list = []

    for img in sorted(os.listdir(EVIDENCE_DIR), reverse=True):
        try:
            ts = int(img.split("_")[1].split(".")[0])
            readable_time = datetime.fromtimestamp(ts).strftime("%d-%m-%Y %H:%M:%S")
        except:
            readable_time = "Unknown time"

        evidence_list.append({
            "file": img,
            "time": readable_time
        })

    return render_template("evidence.html", images=evidence_list)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
