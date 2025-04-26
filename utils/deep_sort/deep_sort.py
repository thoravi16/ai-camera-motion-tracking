import numpy as np
from filterpy.kalman import KalmanFilter

class DeepSort:
    def __init__(self, model_path=None):
        self.tracks = {}
        self.track_id = 0

    def update(self, detections):
        updated_tracks = []
        for det in detections:
            x1, y1, x2, y2, conf = det
            track_id = self.match_existing_track(x1, y1, x2, y2)
            if track_id is None:
                track_id = self.track_id
                self.tracks[track_id] = {"bbox": (x1, y1, x2, y2), "frames": 1}
                self.track_id += 1
            updated_tracks.append([x1, y1, x2, y2, track_id])
        return updated_tracks

    def match_existing_track(self, x1, y1, x2, y2):
        for track_id, track in self.tracks.items():
            tx1, ty1, tx2, ty2 = track["bbox"]
            iou = self.compute_iou((x1, y1, x2, y2), (tx1, ty1, tx2, ty2))
            if iou > 0.5:
                return track_id
        return None

    @staticmethod
    def compute_iou(box1, box2):
        x1, y1, x2, y2 = box1
        tx1, ty1, tx2, ty2 = box2
        xi1, yi1 = max(x1, tx1), max(y1, ty1)
        xi2, yi2 = min(x2, tx2), min(y2, ty2)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (tx2 - tx1) * (ty2 - ty1)
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0
