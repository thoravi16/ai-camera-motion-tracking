# api/motion.py
from flask import Flask, jsonify
import cv2  # or any other import you need

app = Flask(__name__)

@app.route("/api/motion", methods=["GET"])
def detect_motion():
    # You can place your motion detection logic here
    return jsonify({"status": "success", "message": "Motion detection endpoint hit!"})

if __name__ == "__main__":
    app.run()
