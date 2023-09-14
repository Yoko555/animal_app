# 必要なモジュールのインポート
# !pip install torchvision
import torchvision
from torchvision import transforms
import pytorch_lightning as pl
import torch.nn as nn
#学習時に使ったのと同じ学習済みモデルをインポート
from torchvision.models import resnet18 
import torch.nn.functional as F

# 学習済みモデルに合わせた前処理を追加
transform = transforms.Compose([
#    transforms.Resize(256),
#    transforms.CenterCrop(224),
    transforms.ToTensor(),
#    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#　ネットワークの定義
class Net(pl.LightningModule):

    def __init__(self):
        super().__init__()

        #学習時に使ったのと同じ学習済みモデルを定義
        #self.feature = resnet18(pretrained=True) 
        #self.fc = nn.Linear(1000, 2)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=2, padding=1)
        self.bn = nn.BatchNorm2d(3)
        self.feature = resnet18(pretrained=True) #入力は 3 x 224 x 224, 出力は1000(ベクトル)
        self.fc1 = nn.Linear(1000,150 )
        self.fc3 = nn.Linear(150, 2)


    def forward(self, x):
        #学習時に使ったのと同じ順伝播
        #h = self.feature(x)
        #h = self.fc(h)
        #return h

        h = self.conv1(x) # 3 x 224 x 224 -> 6 x 224 x 224
        h = self.bn(h) # 6 x 224 x 224 -> 6 x 224 x 224
        h = F.relu(h) # 3 x 224 224
        h = self.feature(h) #入力は 3 x 224 x 224, 出力は1000(ベクトル)
        h = self.fc1(h)
        h = self.fc3(h)
        return h
