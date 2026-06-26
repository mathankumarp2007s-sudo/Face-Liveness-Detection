import time
import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, render_template, Response, jsonify, request

app = Flask(__name__)

# Video capture is initialized lazily per stream to avoid camera lock errors in Flask reloader.

# MediaPipe face mesh for landmark detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# Shared state for status endpoint
status_data = {
    "status": "Initializing...",
    "blink_count": 0,
    "head_movement": "Unknown",
    "spoof_score": 0,
    "detail": "Waiting for face"
}

# Eye landmark groups for EAR calculation
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
NOSE_TIP = 1

# Liveness thresholds
EAR_THRESHOLD = 0.25
CONSEC_FRAMES_BLINK = 2
HEAD_MOVE_THRESHOLD = 15
HEAD_MOVEMENT_WINDOW = 20
MIN_BLINKS = 1

blink_counter = 0
last_blink_time = 0.0
nose_positions = []


def eye_aspect_ratio(landmarks, eye_indices, image_width, image_height):
    coords = [np.array([landmarks[i].x * image_width, landmarks[i].y * image_height]) for i in eye_indices]
    A = np.linalg.norm(coords[1] - coords[5])
    B = np.linalg.norm(coords[2] - coords[4])
    C = np.linalg.norm(coords[0] - coords[3])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)


def calculate_head_movement():
    if len(nose_positions) < 2:
        return 0.0
    diffs = [np.linalg.norm(np.array(nose_positions[i]) - np.array(nose_positions[i - 1]))
             for i in range(1, len(nose_positions))]
    return float(np.mean(diffs)) if diffs else 0.0


def assess_liveness(blink_count, movement_value):
    movement_detected = movement_value > HEAD_MOVE_THRESHOLD
    if blink_count >= MIN_BLINKS and movement_detected:
        status = "Live person detected"
        detail = "Natural blink and head motion observed."
    elif blink_count >= MIN_BLINKS:
        status = "Likely live face"
        detail = "Blink detected, head motion minimal."
    elif movement_detected:
        status = "Likely live face"
        detail = "Head motion detected, no blink yet."
    else:
        status = "Possible spoof attempt"
        detail = "No blink or significant head motion detected."

    score = int(min(100, max(0, 15 * blink_count + (50 if movement_detected else 0) + 30)))
    return status, detail, score


def draw_status(frame, data):
    cv2.rectangle(frame, (8, 8), (420, 120), (0, 0, 0), -1)
    cv2.putText(frame, f"Status: {data['status']}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Blink Count: {data['blink_count']}", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Head Movement: {data['head_movement']}", (12, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Spoof Score: {data['spoof_score']}%", (12, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame


def generate_frames(camera_idx=0, backend=cv2.CAP_DSHOW):
    global blink_counter, last_blink_time, nose_positions, status_data

    if backend is not None:
        video_capture = cv2.VideoCapture(camera_idx, backend)
    else:
        video_capture = cv2.VideoCapture(camera_idx)
    if not video_capture.isOpened():
        status_data.update({"status": "Camera error", "detail": "Unable to open webcam.", "spoof_score": 0})
        return

    try:
        while True:
            success, frame = video_capture.read()
            if not success:
                status_data.update({"status": "Camera error", "detail": "Unable to read webcam.", "spoof_score": 0})
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)
            image_height, image_width = frame.shape[:2]

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0].landmark
                leftEAR = eye_aspect_ratio(face_landmarks, LEFT_EYE, image_width, image_height)
                rightEAR = eye_aspect_ratio(face_landmarks, RIGHT_EYE, image_width, image_height)
                avgEAR = (leftEAR + rightEAR) / 2.0

                if avgEAR < EAR_THRESHOLD:
                    blink_counter += 1
                else:
                    if blink_counter >= CONSEC_FRAMES_BLINK and time.time() - last_blink_time > 0.5:
                        status_data["blink_count"] += 1
                        last_blink_time = time.time()
                    blink_counter = 0

                nose = face_landmarks[NOSE_TIP]
                nose_xy = (nose.x * image_width, nose.y * image_height)
                nose_positions.append(nose_xy)
                if len(nose_positions) > HEAD_MOVEMENT_WINDOW:
                    nose_positions = nose_positions[-HEAD_MOVEMENT_WINDOW:]

                movement_value = calculate_head_movement()
                status_data["head_movement"] = f"{movement_value:.1f}px"
                status_data["status"], status_data["detail"], status_data["spoof_score"] = assess_liveness(
                    status_data["blink_count"], movement_value)

                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    results.multi_face_landmarks[0],
                    mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1))
            else:
                status_data.update({
                    "status": "No face detected",
                    "detail": "Please position your face in front of the camera.",
                    "spoof_score": 0,
                    "head_movement": "0px"
                })
                blink_counter = 0
                nose_positions = []

            draw_status(frame, status_data)
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        video_capture.release()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    camera_idx = request.args.get('index', default=0, type=int)
    backend_type = request.args.get('backend', default='DSHOW')
    
    if backend_type == 'MSMF':
        backend = cv2.CAP_MSMF
    elif backend_type == 'DEFAULT':
        backend = None
    else:
        backend = cv2.CAP_DSHOW
        
    return Response(generate_frames(camera_idx, backend), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status')
def status():
    return jsonify(status_data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
