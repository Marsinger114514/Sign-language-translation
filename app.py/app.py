import cv2
import torch
import torch.nn as nn
import numpy as np
import os
import logging
import json
import time
from flask import Flask, request, jsonify, g
from torchvision import transforms, models

# 日志配置
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)


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


# --- 从坐标生成骨架图（无需原始图像）---
def skeleton_from_landmarks(landmarks, img_shape=(480, 640)):
    """
    直接从21个关键点坐标生成骨架图
    landmarks: 21个关键点坐标 [[x,y,z], ...] 坐标范围0-1
    """
    h, w = img_shape

    # 将归一化坐标转换为像素坐标
    points = np.array([[lm[0] * w, lm[1] * h] for lm in landmarks])

    # 计算边界框
    min_x, min_y = points.min(axis=0)
    max_x, max_y = points.max(axis=0)
    padding = 20
    min_x = max(0, min_x - padding)
    min_y = max(0, min_y - padding)
    max_x = min(w, max_x + padding)
    max_y = min(h, max_y + padding)
    bbox_w = max_x - min_x
    bbox_h = max_y - min_y

    # 计算缩放比例，使手势居中在224x224图像中
    scale = 224 / max(bbox_w, bbox_h)
    new_w = int(bbox_w * scale)
    new_h = int(bbox_h * scale)

    # 创建黑色背景
    black_img = np.zeros((224, 224, 3), dtype=np.uint8)
    offset_x = (224 - new_w) // 2
    offset_y = (224 - new_h) // 2

    # 手部骨骼连接关系
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # 拇指
        (0, 5), (5, 6), (6, 7), (7, 8),  # 食指
        (5, 9), (9, 10), (10, 11), (11, 12),  # 中指
        (9, 13), (13, 14), (14, 15), (15, 16),  # 无名指
        (13, 17), (17, 18), (18, 19), (19, 20),  # 小指
        (0, 17)  # 手腕连接
    ]

    # 绘制骨骼线
    for start, end in connections:
        x1, y1 = points[start]
        x2, y2 = points[end]

        # 转换坐标到224x224画布
        x1_bbox = (x1 - min_x) * scale
        y1_bbox = (y1 - min_y) * scale
        x2_bbox = (x2 - min_x) * scale
        y2_bbox = (y2 - min_y) * scale

        p1 = (int(x1_bbox + offset_x), int(y1_bbox + offset_y))
        p2 = (int(x2_bbox + offset_x), int(y2_bbox + offset_y))

        # 绘制白色线条
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
        """预测手势"""
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
    # 配置模型路径
    model_path = os.environ.get('MODEL_PATH', 'best_model.pth')
    class_map_path = os.environ.get('CLASS_MAP_PATH', 'class_map.json')

    if os.path.exists(model_path) and os.path.exists(class_map_path):
        classifier = HandGestureClassifier(
            model_path=model_path,
            class_map_path=class_map_path,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        model_loaded = True
        log.info(f"✅ 模型加载成功，使用设备: {classifier.device}")
        log.info(f"支持的手势类别: {list(classifier.class_map.values())}")
    else:
        log.error(f"❌ 模型文件不存在: {model_path} 或 {class_map_path}")
except Exception as e:
    log.error(f"❌ 模型加载失败: {e}")


# --- 基于坐标的预测端点（主要接口）---
@app.route('/api/predict/coordinates', methods=['POST', 'OPTIONS'])
def predict_from_coordinates():
    """接收小程序传来的21个手部关键点坐标，直接进行手势识别"""

    if request.method == 'OPTIONS':
        return '', 200

    if not model_loaded or classifier is None:
        log.error("模型未加载")
        return jsonify({
            "error": "Model not loaded",
            "letter": "错误",
            "confidence": 0
        }), 500

    try:
        # 获取请求数据
        data = request.json
        if not data or 'landmarks' not in data:
            return jsonify({"error": "Missing landmarks data"}), 400

        landmarks = data['landmarks']

        # 验证坐标格式
        if len(landmarks) != 21:
            log.warning(f"关键点数量错误: {len(landmarks)}, 应为21")
            return jsonify({
                "error": f"Invalid landmarks count: {len(landmarks)}",
                "letter": "错误",
                "confidence": 0
            }), 400

        # 验证每个点的格式
        for i, point in enumerate(landmarks):
            if len(point) < 2:
                return jsonify({"error": f"Invalid point format at index {i}"}), 400
            # 确保坐标在合理范围内 (0-1)
            x, y = point[0], point[1]
            if x < 0 or x > 1 or y < 0 or y > 1:
                log.warning(f"坐标超出范围: 点{i} ({x}, {y})")

        # 从坐标生成骨架图
        skeleton_img = skeleton_from_landmarks(landmarks)

        # 预测手势
        letter, confidence = classifier.predict(skeleton_img)

        log.info(f"🎯 预测结果: {letter}, 置信度: {confidence:.4f}")

        return jsonify({
            "letter": letter,
            "confidence": round(confidence, 4),
            "landmarks_count": len(landmarks)
        })

    except Exception as e:
        log.error(f"预测错误: {e}", exc_info=True)
        return jsonify({
            "error": str(e),
            "letter": "错误",
            "confidence": 0
        }), 500


# --- 健康检查端点 ---
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "device": classifier.device.type if classifier else None,
        "num_classes": classifier.num_classes if classifier else 0,
        "classes": list(classifier.class_map.values()) if classifier else []
    })


# --- 根路径 ---
@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "service": "Hand Gesture Recognition API",
        "version": "2.0",
        "endpoints": {
            "/api/predict/coordinates": "POST - 接收手部关键点坐标",
            "/health": "GET - 健康检查"
        },
        "note": "现在只需要21个关键点坐标，无需传输图片"
    })


# --- CORS 和日志中间件 ---
@app.before_request
def before_request():
    g.start_time = time.time()


@app.after_request
def after_request(response):
    # 添加CORS头
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')

    # 记录请求耗时
    if hasattr(g, 'start_time'):
        duration = time.time() - g.start_time
        log.info(f"{request.method} {request.path} - status: {response.status_code}, duration: {duration:.3f}s")
    return response


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

    log.info(f"🚀 启动手势识别服务器")
    log.info(f"📍 端口: {port}")
    log.info(f"🐛 调试模式: {debug}")
    log.info(f"🤖 模型加载: {'是' if model_loaded else '否'}")

    app.run(host='0.0.0.0', port=port, threaded=True, debug=debug)