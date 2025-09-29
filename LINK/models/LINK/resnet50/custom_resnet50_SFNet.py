import torch
import torch.nn as nn
from torchvision.models import resnet50

# 创建一个自定义 ResNet50 模型类，继承自官方的 resnet50
class CustomResNet50(nn.Module):
    def __init__(self, feature_type=None, skin_patch_size=3, pretrained=False, num_classes=1000):
        super(CustomResNet50, self).__init__()
        self.feature_type = feature_type
        
        # 加载预训练的ResNet50模型
        self.resnet = resnet50(pretrained=pretrained)
        
        # 取出ResNet50的特征提取部分，并去掉分类层
        self.features = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4
        )
        
        # 池化层
        self.pool = self.resnet.avgpool
        
        # 处理局部皮肤特征的全连接层
        self.skin_feature = nn.Linear(skin_patch_size**2*3, 3)

        # 输出层
        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.resnet.fc.in_features + 3, num_classes)  # 拼接后输入到分类层
        )

    def forward(self, x, skin_feature):
        # 提取全局特征
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)

        # 如果特征类型是"patch"或者"mapping"，我们处理局部皮肤特征
        if self.feature_type in ["patch", "mapping"]:
            skin_feature = self.skin_feature(torch.flatten(skin_feature, 1))

        # 将局部皮肤特征与全局图像特征拼接
        x = torch.cat((skin_feature, x), dim=1)

        # 通过输出层进行分类
        logits = self.output(x)
        return logits
