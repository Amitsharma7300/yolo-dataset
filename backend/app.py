from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import os
from datetime import datetime
import json
import torch
import time

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"], "allow_headers": "*"}})

# Check GPU availability
USE_GPU = torch.cuda.is_available()
print(f"GPU Available: {USE_GPU}")

# Load model from local workspace
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "best.pt")
model = YOLO(MODEL_PATH)

# Warmup the model to avoid slow first inference
print("Warming up model...")
dummy_img = np.zeros((320, 320, 3), dtype=np.uint8)
for _ in range(3):
    model(dummy_img, verbose=False, imgsz=320, half=USE_GPU)
print("Model warmed up!")

@app.route('/')
def health():
    return jsonify({
        'status': 'running',
        'gpu': USE_GPU,
        'model': 'loaded'
    })

# Class colors for drawing
CLASS_COLORS = {
    0: (34, 197, 94),    # part1 - Green
    1: (59, 130, 246),   # part2 - Blue
    2: (239, 68, 68),    # part3 - Red
    3: (234, 179, 8),    # part4 - Yellow
}

CLASS_NAMES = ['part1', 'part2', 'part3', 'part4']

def draw_detections(image, results):
    """Draw bounding boxes on image"""
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            color = CLASS_COLORS.get(cls, (255, 255, 255))
            # Convert RGB to BGR for OpenCV
            color_bgr = (color[2], color[1], color[0])

            # Draw box
            cv2.rectangle(image, (x1, y1), (x2, y2), color_bgr, 3)

            # Draw label background
            label = f"{CLASS_NAMES[cls]} {conf:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(image, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), color_bgr, -1)

            # Draw label text
            cv2.putText(image, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return image

def process_detections(results, scale=1):
    """Extract detection info from results"""
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(lambda x: int(x / scale), box.xyxy[0].tolist())
            detections.append({
                'class_id': int(box.cls[0]),
                'class_name': CLASS_NAMES[int(box.cls[0])],
                'confidence': round(float(box.conf[0]) * 100, 1),
                'bbox': {
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'width': x2 - x1, 'height': y2 - y1
                }
            })
    return detections

@app.route('/api/detect/base64', methods=['POST'])
def detect_base64():
    """Detect objects from base64 image"""
    try:
        data = request.json
        image_data = data.get('image', '')

        # Remove data URL prefix if present
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]

        # Decode base64 to image
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'success': False, 'error': 'Invalid image'}), 400

        # Run detection
        results = model(image, conf=0.5)

        # Draw detections
        result_image = draw_detections(image.copy(), results)

        # Encode result image to base64
        _, buffer = cv2.imencode('.jpg', result_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        result_base64 = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode()}"

        # Get detection info
        detections = process_detections(results)

        return jsonify({
            'success': True,
            'detections': detections,
            'total_detections': len(detections),
            'result_image': result_base64,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/detect/frame', methods=['POST'])
def detect_frame():
    """Fast detection for real-time video frames - optimized for speed"""
    start_time = time.time()
    try:
        data = request.json
        image_data = data.get('image', '')

        # Remove data URL prefix if present
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]

        # Decode base64 to image
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'success': False, 'error': 'Invalid image'}), 400

        decode_time = time.time()

        # Resize for faster inference
        h, w = image.shape[:2]
        target_size = 320
        scale = min(target_size / w, target_size / h)
        if scale < 1:
            new_w, new_h = int(w * scale), int(h * scale)
            image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            image_resized = image
            scale = 1

        resize_time = time.time()

        # Run detection with optimized settings
        results = model(image_resized, conf=0.25, verbose=False, imgsz=320, half=USE_GPU)

        inference_time = time.time()

        # Draw detections on original image
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Scale coordinates back to original size
                x1, y1, x2, y2 = map(lambda x: int(x / scale), box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                color = CLASS_COLORS.get(cls, (255, 255, 255))
                color_bgr = (color[2], color[1], color[0])

                cv2.rectangle(image, (x1, y1), (x2, y2), color_bgr, 2)
                label = f"{CLASS_NAMES[cls]} {conf:.0%}"
                cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)

        # Encode with lower quality for faster transfer
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 50])
        result_base64 = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode()}"

        # Get detection info
        detections = process_detections(results, scale)

        total_time = time.time() - start_time
        print(f"Frame: decode={decode_time-start_time:.3f}s, resize={resize_time-decode_time:.3f}s, inference={inference_time-resize_time:.3f}s, total={total_time:.3f}s, detections={len(detections)}")

        return jsonify({
            'success': True,
            'detections': detections,
            'total_detections': len(detections),
            'result_image': result_base64
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 50)
    print("YOLO Parts Detection Server")
    print(f"Model: {MODEL_PATH}")
    print("=" * 50)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
