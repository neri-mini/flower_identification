import os
import json
import random
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

# 1.路径管理
SRC = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(SRC)
DATA = os.path.join(PROJECT, "data")
DOCS = os.path.join(PROJECT, "docs")
OUTPUT = os.path.join(PROJECT, "outputs")
FLOWER_DIR = os.path.join(PROJECT, "flower_data")
TRAIN_DIR = os.path.join(FLOWER_DIR, "train")
TEST_DIR = os.path.join(FLOWER_DIR, "test")

INPUT_DIR = os.path.join(PROJECT, "input_images")
TRAIN_LIST = os.path.join(DATA, "train_list.csv")
TEST_LIST = os.path.join(DATA, "test_list.csv")
PARAM_JSON = os.path.join(DATA, "prep_params.json")
MODEL_PATH = os.path.join(OUTPUT, "model.pth")
PREDICT_CSV = os.path.join(OUTPUT, "predictions.csv")

# 2. 配置
SEED = 42
NUM_CLASSES = 5
IMG_SIZE = 224

# 超参数
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4
T_MAX = 30
BATCH_SIZE = 32
NUM_WORKERS = 4

# 扫描坏图
def scan_bad(dirname):
    bad = []
    for root,_,files in os.walk(dirname):
        for picture in files:
            path = os.path.join(root,picture)
            try:
                Image.open(path).convert("RGB").load()
            except Exception:
                bad.append(path)         
    return bad

# 剔除坏图
def drop_bad(dirname):
    bad = set(scan_bad(dirname))
    full = ImageFolder(dirname)
    samples = [(path,label) for path, label in full.samples if path not in bad]
    return full.classes, samples, len(bad)

# 保存清单
def save_list(samples, path):
    with open(path,"w",encoding="utf-8") as f:
        f.write("path,label\n")
        for path,label in samples:
            rel_path = os.path.relpath(path, PROJECT).replace("\\", "/")
            f.write(rel_path + "," + str(label) + "\n")

# 计算mean和std
def compute_mean_std(samples):
    to_tensor = transforms.ToTensor()
    pixel_sum = torch.zeros(3,dtype = torch.float64)
    square_sum = torch.zeros(3,dtype = torch.float64)
    count = 0
    for path,_ in samples:
        I = Image.open(path).convert("RGB")
        x = to_tensor(I).double()
        pixel_sum += x.sum(dim = (1,2))
        square_sum += (x ** 2).sum(dim = (1,2))
        count += x.shape[1] * x.shape[2]
    mean = pixel_sum / count
    std = (square_sum / count - mean ** 2).sqrt()
    return [round(a,6) for a in mean.tolist()],[round(b,6) for b in std.tolist()]

# 主流程
def main():
    random.seed(SEED)
    if not os.path.isdir(TRAIN_DIR) or not os.path.isdir(TEST_DIR):
        print("骚年你还没下文件呢，还想训练？")
        return

    classes_list, train_samples, n_bad_train = drop_bad(TRAIN_DIR)   
    print(f"train 目录扫出坏图 {n_bad_train} 张，已剔除")
    _, test_samples, n_bad_test = drop_bad(TEST_DIR)
    print(f"test  目录扫出坏图 {n_bad_test} 张，已剔除")

    os.makedirs(DATA, exist_ok=True)
    save_list(train_samples, TRAIN_LIST)
    save_list(test_samples, TEST_LIST)

    print(f"正在统计训练集 {len(train_samples)} 张图的均值和标准差")
    mean, std = compute_mean_std(train_samples)

    with open(PARAM_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "classes": classes_list,
            "num_classes": len(classes_list),
            "img_size": IMG_SIZE,
            "mean": mean,
            "std": std,
        }, f, ensure_ascii=False, indent=2)
    print(f"data_prepare运行成功")

if __name__ == "__main__":
    main()      