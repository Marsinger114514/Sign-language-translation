import torch
import torch.nn as nn
import numpy as np
import os
import logging
import json
import time
from flask import Flask, request, jsonify, g

# --- 1. 日志增强配置 ---
LOG_FILE = "gesture_recognition.log"
# 配置日志格式： 时间 | 级别 | 信息
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),  # 写入文件
        logging.StreamHandler()  # 同时输出到控制台
    ]
)
log = logging.getLogger(__name__)

app = Flask(__name__)


# --- 模型定义和分类器 (保持不变) ---
class GestureMLP(nn.Module):
    def __init__(self, num_classes):
        super(GestureMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(42, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x): return self.net(x)


def preprocess_landmarks(landmarks):
    coords = np.array([[lm[0], lm[1]] for lm in landmarks])
    coords = coords - coords[0]
    max_val = np.abs(coords).max()
    if max_val > 0: coords = coords / max_val
    return coords.flatten().astype(np.float32)


class HandGestureClassifier:
    def __init__(self, model_path, class_map_path, device='cpu'):
        with open(class_map_path, 'r') as f:
            self.class_map = json.load(f)
        self.num_classes = len(self.class_map)
        self.device = torch.device(device)
        self.model = GestureMLP(self.num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, landmarks):
        features = preprocess_landmarks(landmarks)
        input_tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            conf, idx = torch.max(probs, 0)
        return self.class_map.get(str(idx.item()), "Unknown"), conf.item()


# 实例化
classifier = None
try:
    classifier = HandGestureClassifier('asl_model.pth', 'class_map.json',
                                       device='cuda' if torch.cuda.is_available() else 'cpu')
except Exception as e:
    print(f"初始化失败: {e}")


# --- 2. 核心预测接口 (增加耗时统计和格式化日志) ---
@app.route('/api/predict/coordinates', methods=['POST'])
def predict_from_coordinates():
    start_time = time.time()  # 开始计时
    try:
        data = request.json
        landmarks = data.get('landmarks')
        if not landmarks or len(landmarks) != 21:
            return jsonify({"error": "Need 21 landmarks"}), 400

        # 执行推理
        letter, confidence = classifier.predict(landmarks)

        # 计算耗时
        duration_ms = (time.time() - start_time) * 1000

        # --- 关键：格式化输出日志，匹配你要求的格式 ---
        # 格式：字母: A | 置信度: 0.9999 | 耗时: 12.3ms
        log_msg = f"字母: {letter} | 置信度: {confidence:.4f} | 耗时: {duration_ms:.1f}ms"
        log.info(log_msg)

        return jsonify({
            "letter": letter,
            "confidence": round(confidence, 4),
            "process_time_ms": round(duration_ms, 2)
        })
    except Exception as e:
        log.error(f"预测发生异常: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500


# --- 3. 新增：获取历史日志接口 ---
@app.route('/api/logs', methods=['GET'])
def get_logs():
    """返回最近的 50 条识别记录"""
    try:
        if not os.path.exists(LOG_FILE):
            return "还没有产生任何日志。"

        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # 返回最后50行
            last_lines = lines[-50:]
            return "<pre>" + "".join(last_lines) + "</pre>"  # 使用 pre 标签保持格式
    except Exception as e:
        return str(e), 500


# CORS 处理
@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response


if __name__ == '__main__':
    log.info("🚀 识别服务器已启动，等待请求...")
    app.run(host='0.0.0.0', port=5000, threaded=True)