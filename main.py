import cv2
import torch
from ultralytics import YOLO
import numpy as np
from utils.deep_sort.deep_sort import DeepSort

# Load YOLOv8 model
model = YOLO("models/yolov8n.pt")  # Ensure this model is downloaded

# Initialize DeepSORT tracker
tracker = DeepSort(model_path="models/deep_sort.ckpt")

# Open webcam
cap = cv2.VideoCapture(0)  # 0 for default camera

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform person detection
    results = model(frame)

    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            conf = box.conf[0].item()  # Confidence score
            cls = int(box.cls[0].item())  # Class index

            if cls == 0:  # Class ID for 'person'
                detections.append([x1, y1, x2, y2, conf])

    # Convert detections into NumPy array
    if len(detections) > 0:
        detections = np.array(detections)

    # Update the tracker
    tracked_objects = tracker.update(detections)

    # Draw tracking info on the frame
    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = map(int, obj[:5])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Person {obj_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("AI Camera Motion Tracking", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
