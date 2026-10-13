import os
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from data_prepare import (MODEL_PATH, PARAM_JSON, INPUT_DIR, PREDICT_CSV,
                          IMG_SIZE, BATCH_SIZE, NUM_WORKERS)
from train import load_params                          
from model import FlowerNet

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".avif")

# 加载模型
def load_model(device):
    if not os.path.exists(MODEL_PATH):
        print(f"找不到模型文件：{MODEL_PATH}")
        print("请先运行：train.py")
        return None
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = FlowerNet().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model

# 列出待预测的图片
def list_images(input_dir):
    names = []
    for I in os.listdir(input_dir):
        if I.lower().endswith(IMAGE_EXTS):
            names.append(I)
    return sorted(names)


# 待预测数据集（只需要图片）
class InputDataset(Dataset):
    def __init__(self, root, names, transform):
        self.root = root
        self.names = names
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, self.names[index])).convert("RGB")
        return self.transform(img)


# 批量推理
@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    all_probs = []
    for x in loader:
        probs = F.softmax(model(x.to(device)), dim=1).cpu()
        all_probs += probs.tolist()
    return all_probs


# 保存结果
def save_result(names, all_probs, classes):
    os.makedirs(os.path.dirname(PREDICT_CSV), exist_ok=True)
    header = ["path", "predicted_label"] + [f"prob_{c}" for c in classes]

    with open(PREDICT_CSV, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for name, probs in zip(names, all_probs):
            pred_label = classes[int(torch.tensor(probs).argmax())]
            values = [name, pred_label] + [f"{p:.4f}" for p in probs]
            f.write(",".join(values) + "\n")


# 主流程
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params, _ , test_tf = load_params()
    classes = params["classes"]

    model = load_model(device)
    if model is None:
        return

    os.makedirs(INPUT_DIR, exist_ok=True)
    names = list_images(INPUT_DIR)
    if not names:
        print(f"待预测目录里没有图片：{INPUT_DIR}")
        print("请把 .jpg / .png 等图片直接放进该目录，再重新运行")
        return

    print(f"待预测 {len(names)} 张")

    dataset = InputDataset(INPUT_DIR, names, test_tf)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
    all_probs = predict_all(model, loader, device)
    save_result(names, all_probs, classes)

    print(f"结果已保存，请查看{PREDICT_CSV}")


if __name__ == "__main__":
    main()
