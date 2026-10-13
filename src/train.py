import os
import random
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from model import FlowerNet
from data_prepare import (SEED, EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY, T_MAX,
                          DOCS, OUTPUT, MODEL_PATH, NUM_WORKERS,IMG_SIZE,
                          TRAIN_LIST, TEST_LIST,PARAM_JSON)

# 随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 读之前保存的文件清单
def read_list(list_path):
    samples = []
    with open(list_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            path, label = line.rstrip("\n").rsplit(",")
            samples.append((path, int(label)))
    return samples        

# 图像变换器
def build_transforms(mean,std):
    train_tf = transforms.Compose([
      transforms.Resize(256),
      transforms.RandomCrop(IMG_SIZE),
      transforms.RandomHorizontalFlip(),
      transforms.RandomRotation(15),
      transforms.ToTensor(),
      transforms.Normalize(mean,std),
    ])
    test_tf = transforms.Compose([
      transforms.Resize(256),
      transforms.RandomCrop(IMG_SIZE),
      transforms.ToTensor(),
      transforms.Normalize(mean,std),
    ])
    return train_tf,test_tf

# 加载之前保存的json文件
def load_params():
    with open(PARAM_JSON,"r",encoding="utf-8") as f:
        params = json.load(f)
    train_tf,test_tf = build_transforms(params["mean"], params["std"])
    return params, train_tf, test_tf

# 定义数据集类
class FlowerDataset(Dataset):
    def __init__(self,samples,transforms):
        self.samples = samples
        self.transforms = transforms

    def __len__(self):
      return len(self.samples)

    def __getitem__(self,index):
        path,label = self.samples[index]
        I = Image.open(path).convert("RGB")
        return self.transforms(I),label

# 加载函数，建立模型对象和两个数据加载器
def build_model_loader(device):
    params, train_tf, test_tf = load_params()
    model = FlowerNet().to(device)

    train_dataset = FlowerDataset(read_list(TRAIN_LIST),train_tf)
    test_dataset = FlowerDataset(read_list(TEST_LIST),test_tf)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False,
                             num_workers=NUM_WORKERS,pin_memory=True)
    return model,train_loader,test_loader

# 损失函数
@torch.no_grad()
def evaluate(model,loader,criterion,device):
    model.eval()
    loss_sum,correct,total = 0.0 ,0 ,0
    for x,y in loader:
        x,y = x.to(device),y.to(device)
        logits = model(x)
        loss_sum += criterion(logits,y).item() * len(y)  
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
    return loss_sum / total, correct / total

# 训练函数
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(SEED)

    model,train_loader,test_loader = build_model_loader(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=T_MAX)

    hist = {"train_loss": [], "train_acc":[],"test_loss":[],"test_acc":[]}
    best_acc,best_state,best_epoch = 0.0,None,0

    for epoch in range(1,EPOCHS + 1):
        model.train()
        loss_sum,correct,total = 0.0 ,0 ,0
        for x,y in train_loader:
            x,y = x.to(device),y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits,y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
        scheduler.step()

        train_loss = loss_sum / total
        train_acc = correct / total
        test_loss,test_acc = evaluate(model, test_loader, criterion, device)
        hist["train_loss"].append(train_loss)
        hist["train_acc"].append(train_acc)
        hist["test_loss"].append(test_loss)
        hist["test_acc"].append(test_acc)

        if test_acc > best_acc:
            best_acc,best_epoch = test_acc ,epoch
            best_state = {k: v.detach().clone() for k,v in model.state_dict().items()}

        print(f"当前 {epoch}/{EPOCHS}轮   训练loss {train_loss:.4f} 准确率 {train_acc * 100:.2f}% ")
        print(f"测试loss {test_loss:.4f} 准确率 {test_acc * 100:.2f}%")

    model.load_state_dict(best_state)
    print(f"最佳轮次 {best_epoch}  测试集准确率 {best_acc * 100:.2f}%")
    return model,hist

# 绘图
def plot(hist):
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False 
    ep = range(1,len(hist["train_loss"]) + 1)

    plt.figure(figsize=(7,4))
    plt.plot(ep,hist["train_loss"],label="Train_loss")
    plt.plot(ep,hist["test_loss"],label="Test_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(DOCS, "loss_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7,4))
    plt.plot(ep,hist["train_acc"],label="Train_acc")
    plt.plot(ep,hist["test_acc"],label="Test_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(DOCS, "accuracy_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()

# 保存模型
def save_model(model):
    torch.save({"state_dict": model.state_dict()},MODEL_PATH)

# 主流程
def main():
    os.makedirs(DOCS, exist_ok=True)
    os.makedirs(OUTPUT, exist_ok=True)

    model,hist = train()
    plot(hist)
    save_model(model)
    print(f"模型已保存，请到{MODEL_PATH}查看")

if __name__ == "__main__":
    main()        