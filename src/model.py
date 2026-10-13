import torch.nn as nn
from data_prepare import NUM_CLASSES

class FlowerNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
        nn.Conv2d(3,96,11,4,2),
        nn.ReLU(inplace=True),
        nn.Conv2d(96,256,5,1,2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2,2),
        nn.Conv2d(256,384,3,1,1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2,2),
        nn.Conv2d(384,384,3,1,1),
        nn.ReLU(inplace=True),
        nn.Conv2d(384,256,3,1,1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2,2),
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
        nn.Dropout(0.5),  
        nn.Linear(256 * 6 * 6, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(4096, 5),    
        )
    def forward(self,x):
        x = self.features(x)
        x = self.flatten(x)
        return self.classifier(x)


