import cv2
import numpy as np
import datetime
from flask import Flask, Response, jsonify, request, render_template
from flask_cors import CORS
from ultralytics import YOLO
from utils.deep_sort.deep_sort import DeepSort

app = Flask(__name__)
CORS(app)

# Load YOLOv8 model
model = YOLO("models/yolov8n.pt")

# Initialize DeepSORT tracker
tracker = DeepSort(model_path="models/deep_sort.ckpt")

# Open webcam
cap = cv2.VideoCapture(0)

# Global variables for motion detection and counting
motion_detected = False
tracking_enabled = True
motion_count = 0
unique_ids = set()

def generate_frames():
    global motion_detected, tracking_enabled, motion_count, unique_ids
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Mirror the frame horizontally
        frame = cv2.flip(frame, 1)

        # Perform detection
        results = model(frame, verbose=False)
        detections = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())

                if cls == 0:  # 'person' class
                    detections.append([x1, y1, x2, y2, conf])

        detections = np.array(detections) if detections else np.empty((0, 5))

        # Update tracker if enabled
        if tracking_enabled:
            tracked_objects = tracker.update(detections)
            motion_detected = len(tracked_objects) > 0

            for obj in tracked_objects:
                x1, y1, x2, y2, track_id = map(int, obj[:5])
                if track_id not in unique_ids:
                    unique_ids.add(track_id)
                    motion_count += 1
                    print(f"New motion detected. Total count: {motion_count}")

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Person {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Encode frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/motion-status')
def motion_status():
    return jsonify({"motion": motion_detected})

@app.route('/motion-count')
def get_motion_count():
    return jsonify({"motion_count": motion_count})

@app.route('/toggle-tracking', methods=['POST'])
def toggle_tracking():
    global tracking_enabled
    data = request.json
    tracking_enabled = data.get("tracking", True)
    return jsonify({"tracking": tracking_enabled})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
