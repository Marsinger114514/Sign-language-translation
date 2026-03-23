# app.py
import cv2
import torch
import torch.nn as nn
import numpy as np
import os
import threading
import logging
import base64
import json
import time
from PIL import Image
from flask import Flask, request, jsonify, g
from cvzone.HandTrackingModule import HandDetector
from torchvision import transforms, models

# 日志配置
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)

# 线程局部存储 + 锁保证 MediaPipe 的线程安全
thread_local = threading.local()
detector_lock = threading.Lock()

def get_detector():
    """每个线程独立初始化检测器，并使用锁确保创建过程线程安全"""
    if not hasattr(thread_local, 'detector'):
        with detector_lock:
            if not hasattr(thread_local, 'detector'):
                thread_local.detector = HandDetector(
                    maxHands=1,
                    detectionCon=0.5,
                    minTrackCon=0.5,
                    modelComplexity=0
                )
    return thread_local.detector

# --- 模型定义 (ResNet50) ---
class ResNetWrapper(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.backbone.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# --- 改进的骨架化处理：保持宽高比，居中绘制 ---
def skeleton_to_black(img, lmList):
    h, w, _ = img.shape
    points = np.array([[lm[0], lm[1]] for lm in lmList])
    min_x, min_y = points.min(axis=0)
    max_x, max_y = points.max(axis=0)
    padding = 20
    min_x = max(0, min_x - padding)
    min_y = max(0, min_y - padding)
    max_x = min(w, max_x + padding)
    max_y = min(h, max_y + padding)
    bbox_w = max_x - min_x
    bbox_h = max_y - min_y

    scale = 224 / max(bbox_w, bbox_h)
    new_w = int(bbox_w * scale)
    new_h = int(bbox_h * scale)

    black_img = np.zeros((224, 224, 3), dtype=np.uint8)
    offset_x = (224 - new_w) // 2
    offset_y = (224 - new_h) // 2

    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17)
    ]

    for start, end in connections:
        x1, y1 = lmList[start][0], lmList[start][1]
        x2, y2 = lmList[end][0], lmList[end][1]

        x1_bbox = (x1 - min_x) * scale
        y1_bbox = (y1 - min_y) * scale
        x2_bbox = (x2 - min_x) * scale
        y2_bbox = (y2 - min_y) * scale

        p1 = (int(x1_bbox + offset_x), int(y1_bbox + offset_y))
        p2 = (int(x2_bbox + offset_x), int(y2_bbox + offset_y))
        cv2.line(black_img, p1, p2, (255, 255, 255), 3)

    return black_img

# --- 分类器封装 ---
class HandGestureClassifier:
    def __init__(self, model_path, class_map_path, device='cpu'):
        with open(class_map_path, 'r') as f:
            self.class_map = json.load(f)
        self.num_classes = len(self.class_map)
        self.device = torch.device(device)
        self.model = ResNetWrapper(self.num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, img):
        img_t = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(img_t)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            conf, idx = torch.max(probs, 0)
        class_id = str(idx.item())
        return self.class_map[class_id], conf.item()

# 全局分类器实例
model_loaded = False
classifier = None

try:
    model_path = os.environ.get('MODEL_PATH', 'best_model.pth')
    class_map_path = os.environ.get('CLASS_MAP_PATH', 'class_map.json')
    if os.path.exists(model_path) and os.path.exists(class_map_path):
        classifier = HandGestureClassifier(
            model_path=model_path,
            class_map_path=class_map_path,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        model_loaded = True
        log.info(f"模型加载成功，使用设备: {classifier.device}")
    else:
        log.error(f"模型文件不存在: {model_path} 或 {class_map_path}")
except Exception as e:
    log.error(f"模型加载失败: {e}")

# --- 请求计时中间件 ---
@app.before_request
def before_request():
    g.start_time = time.time()

@app.after_request
def after_request(response):
    if hasattr(g, 'start_time'):
        duration = time.time() - g.start_time
        log.info(f"{request.method} {request.path} - status: {response.status_code}, duration: {duration:.3f}s")
    return response

# --- 预测端点 ---
@app.route('/api/predict/letter', methods=['POST', 'OPTIONS'])
def predict_letter():
    if request.method == 'OPTIONS':
        return '', 200

    if not model_loaded or classifier is None:
        log.error("模型未加载，无法处理预测请求")
        return jsonify({"error": "Model not loaded", "letter": "未知", "confidence": 0}), 500

    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({"error": "Missing image data"}), 400

        img_back = base64.b64decode(data['image'])
        nparr = np.frombuffer(img_back, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Invalid image data"}), 400

        detector = get_detector()
        hands, _ = detector.findHands(img, draw=False, flipType=True)

        if not hands:
            log.debug("未检测到手部")
            return jsonify({"letter": "无手势", "confidence": 0.0})

        hand = hands[0]
        skeleton_img = skeleton_to_black(img, hand["lmList"])
        letter, confidence = classifier.predict(skeleton_img)

        log.info(f"预测结果: {letter}, 置信度: {confidence:.4f}")
        return jsonify({"letter": letter, "confidence": round(confidence, 4)})

    except Exception as e:
        log.error(f"预测错误: {e}", exc_info=True)
        return jsonify({"error": str(e), "letter": "错误", "confidence": 0}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "device": classifier.device.type if classifier else None
    })

@app.after_request
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    return response

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, threaded=True, debug=debug)