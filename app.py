import cv2
import numpy as np
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

    # Flip frame (optional)
    frame = cv2.flip(frame, 1)

    # Run YOLO detection
    results = model(frame, verbose=False)
    detections = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())

            if cls == 0:  # 'person'
                detections.append([x1, y1, x2, y2, conf])

    detections = np.array(detections) if detections else np.empty((0, 5))

    # DeepSORT tracking
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

    # Encode the processed frame to send back
    _, buffer = cv2.imencode('.jpg', frame)
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
