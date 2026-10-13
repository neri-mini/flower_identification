# 花卉图像分类（PyTorch + CNN）

用 PyTorch 卷积神经网络对 5 种花卉做图像分类

- **任务**：5 分类（daisy / dandelion / rose / sunflower / tulip）
- **数据**：训练目录 3470 张（剔除 13 张坏图后 3457 张）+ 测试目录 860 张，原始尺寸不统一

## 运行结果

| 数据集 | 样本数 | 准确率 |
|---|---|---|
| 训练集 | 3457 | 见训练日志 |
| **测试集** | **860** | **76.51%**（658 / 860）|

5 分类的随机猜测基线是 **20%**，模型达到基线的 3.8 倍。

## 目录结构

```
flower_identification/
├── src/
│   ├── data_prepare.py     清洗坏图 + 生成两份数据清单
│   ├── model.py            网络结构
│   ├── train.py            训练 + 每轮测试集评估 + 画曲线 + 保存最佳模型
│   └── predict.py          对 input_images/ 里的图片逐张预测
├── data/
│   ├── train_list.csv      训练集清单：路径 + 类别
│   ├── test_list.csv       测试集清单：路径 + 类别
│   └── prep_params.json    预处理参数：类别名、图片尺寸、归一化均值方差
├── docs/
│   ├── 学习笔记.pdf         
│   ├── loss_curve.png      损失曲线：Train / Test
│   └── accuracy_curve.png  准确率曲线：Train / Test
├── input_images/           ← 放你要预测的图片（已带若干张示例）（示例图来自 Pexels/Unsplash，仅用于演示）
├── outputs/
│   ├── model.pth           训练好的模型文件（要自己运行才有）
│   └── predictions.csv     对 input_images/ 的预测结果
├── flower_data/            ← 数据集（自己下）
├── requirements.txt
└── README.md
```

## 食用教程（

1. 去地址(https://pan.baidu.com/s/1GKc89PeWNq1TxeNuIIk-4A?pwd=6a48)

   下载文件或自己找图片分类之后放好

2. 依次运行data_prepare.py，train.py

3. 自己找图放入input_images/（仓库自带几张示例）（只放图片哦）

4. 运行predict.py

5. 看predictions.csv



