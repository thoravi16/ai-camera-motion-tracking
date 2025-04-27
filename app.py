import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, render_template
from flask_cors import CORS
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

# Load YOLOv8 Nano model (small and fast)
model = YOLO("yolov8n.pt")  # Make sure 'yolov8n.pt' is present

# Tracking and motion flags
motion_detected = False
motion_count = 0
unique_ids = set()
tracking_enabled = True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    global motion_detected, motion_count, unique_ids, tracking_enabled

    if 'frame' not in request.files:
        return jsonify({'error': 'No frame uploaded'}), 400

    file = request.files['frame']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({'error': 'Invalid frame'}), 400

    frame = cv2.flip(frame, 1)

    # Resize frame for speed
    resized_frame = cv2.resize(frame, (416, 416))

    results = model.predict(resized_frame, verbose=False)

    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())

            if cls == 0:  # 0 is 'person' class
                detections.append([x1, y1, x2, y2, conf])

    motion_detected = False

    if tracking_enabled and len(detections) > 0:
        motion_detected = True
        for det in detections:
            x1, y1, x2, y2, conf = det
            cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Person {conf:.2f}"
            cv2.putText(resized_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        motion_count += 1

    # Encode the processed frame
    _, buffer = cv2.imencode('.jpg', resized_frame)
    return Response(buffer.tobytes(), mimetype='image/jpeg')

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
    app.run(host='0.0.0.0', port=5000)
