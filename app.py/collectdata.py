import cv2
import mediapipe as mp
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# --- 配置参数 ---
TRAIN_DIR = r'E:\Pythonproject\app.py\data\asl_alphabet_train'
TEST_DIR = r'E:\Pythonproject\app.py\data\asl_alphabet_test'
LABEL_MAP = {chr(i): i - 65 for i in range(65, 91)}  # {'A': 0, 'B': 1, ..., 'Z': 25}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化 MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)


def extract_landmarks(image_path):
    """提取图片中的手部坐标并进行归一化"""
    img = cv2.imread(image_path)
    if img is None: return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        # 只取第一只手
        landmarks = results.multi_hand_landmarks[0].landmark
        coords = []
        # 1. 提取原始坐标
        for lm in landmarks:
            coords.append([lm.x, lm.y])
        coords = np.array(coords)  # (21, 2)

        # 2. 归一化：平移不变性（以手腕点为原点）
        coords = coords - coords[0]

        # 3. 归一化：缩放不变性（按手掌最大跨度缩放）
        max_value = np.abs(coords).max()
        if max_value > 0:
            coords = coords / max_value

        return coords.flatten()  # 返回 42 维向量
    return None


# --- 数据集定义 ---
class ASLDataset(Dataset):
    def __init__(self, data_list):
        # data_list 格式: [(coords_vector, label_idx), ...]
        self.features = torch.tensor([x[0] for x in data_list], dtype=torch.float32)
        self.labels = torch.tensor([x[1] for x in data_list], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# --- 模型定义 ---
class GestureModel(nn.Module):
    def __init__(self):
        super(GestureModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(42, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 26)  # 26个字母
        )

    def forward(self, x):
        return self.net(x)


def main():
    # 1. 加载训练集 (结构: train/A/A1.jpg)
    print("正在提取训练集特征...")
    train_data = []
    for char in tqdm(os.listdir(TRAIN_DIR)):
        char_path = os.path.join(TRAIN_DIR, char)
        if os.path.isdir(char_path):
            label = LABEL_MAP.get(char.upper())
            for img_name in os.listdir(char_path):
                f = extract_landmarks(os.path.join(char_path, img_name))
                if f is not None:
                    train_data.append((f, label))

    # 2. 加载测试集 (结构: test/A_test.jpg)
    print("正在提取测试集特征...")
    test_data = []
    for img_name in tqdm(os.listdir(TEST_DIR)):
        # 假设文件名类似 A_test.jpg 或 A1_test.jpg，取第一个字母
        char = img_name[0].upper()
        if char in LABEL_MAP:
            label = LABEL_MAP[char]
            f = extract_landmarks(os.path.join(TEST_DIR, img_name))
            if f is not None:
                test_data.append((f, label))

    print(f"提取完成！有效训练样本: {len(train_data)}, 测试样本: {len(test_data)}")

    # 3. 准备数据加载器
    train_loader = DataLoader(ASLDataset(train_data), batch_size=64, shuffle=True)
    test_loader = DataLoader(ASLDataset(test_data), batch_size=32, shuffle=False)

    # 4. 训练模型
    model = GestureModel().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("开始训练...")
    for epoch in range(50):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 每10轮测试一次
        if epoch % 10 == 0:
            model.eval()
            correct = 0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    output = model(x)
                    correct += (output.argmax(1) == y).sum().item()
            print(
                f"Epoch {epoch}, Loss: {total_loss / len(train_loader):.4f}, Test Acc: {100 * correct / len(test_data):.2f}%")

    # 5. 保存模型
    torch.save(model.state_dict(), "asl_model.pth")
    print("模型已保存为 asl_model.pth")


if __name__ == '__main__':
    main()