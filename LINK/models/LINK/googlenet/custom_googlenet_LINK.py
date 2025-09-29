import torch
import torch.nn as nn
from torchvision.models import googlenet
import torch.nn.functional as F

class SPP(nn.Module):
    def __init__(self, M=None):
        super(SPP, self).__init__()
        self.pooling_4x4 = nn.AdaptiveAvgPool2d((4, 4))
        self.pooling_2x2 = nn.AdaptiveAvgPool2d((2, 2))
        self.pooling_1x1 = nn.AdaptiveAvgPool2d((1, 1))

        self.M = M
        print(self.M)

    def forward(self, x):
        x_4x4 = self.pooling_4x4(x)
        x_2x2 = self.pooling_2x2(x_4x4)
        x_1x1 = self.pooling_1x1(x_4x4)

        x_4x4_flatten = torch.flatten(x_4x4, start_dim=2, end_dim=3)  # B X C X feature_num

        x_2x2_flatten = torch.flatten(x_2x2, start_dim=2, end_dim=3)

        x_1x1_flatten = torch.flatten(x_1x1, start_dim=2, end_dim=3)

        if self.M == '[1,2,4]':
            x_feature = torch.cat((x_1x1_flatten, x_2x2_flatten, x_4x4_flatten), dim=2)
        elif self.M == '[1,2]':
            x_feature = torch.cat((x_1x1_flatten, x_2x2_flatten), dim=2)
        else:
            raise NotImplementedError('ERROR M')

        x_strength = x_feature.permute((2, 0, 1))
        x_strength = torch.mean(x_strength, dim=2)


        return x_feature, x_strength
    
    
# 创建一个自定义 GoogleNet 模型类，继承自官方的 googlenet
class CustomGoogLeNet(nn.Module):
    def __init__(self, pretrained=False, num_classes=1000):
        super(CustomGoogLeNet, self).__init__()
        # 加载预训练的GoogleNet模型
        self.googlenet = googlenet(pretrained=pretrained)
        
        # 自定义分类层
        num_features = self.googlenet.fc.in_features
        self.googlenet.fc = nn.Linear(num_features, num_classes)

    # 重写 _forward_impl 方法，修改输出结构
    def _forward_impl(self, x):
        # 保留原始结构中的前向操作
        x = self.googlenet.conv1(x)
        feat1 = x
        x = self.googlenet.maxpool1(x)
        x = self.googlenet.conv2(x)
        feat2 = x
        x = self.googlenet.maxpool2(x)
        x = self.googlenet.conv3(x)
        x = self.googlenet.maxpool3(x)
        x = self.googlenet.inception3(x)
        feat3 = x
        x = self.googlenet.maxpool4(x)
        x = self.googlenet.inception4(x)
        feat4 = x
        x = self.googlenet.avgpool(x)

        # 展开并通过自定义分类层输出
        x = torch.flatten(x, 1)
        
        avg = x
        
        out = self.googlenet.fc(x)
        
        feats = {}
        feats["pooled_feat"] = avg
        feats["feats"] = [
            F.relu(stem),
            F.relu(feat1),
            F.relu(feat2),
            F.relu(feat3),
            F.relu(feat4),
        ]
        feats["preact_feats"] = [stem, feat1, feat2, feat3, feat4]
        
        return out

    # 使用 forward 来调用 _forward_impl
    def forward(self, x):
        return self._forward_impl(x)
