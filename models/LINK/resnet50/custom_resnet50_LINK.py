import torch
import torch.nn as nn
from torchvision.models import resnet50
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


# 创建一个自定义 ResNet50 模型类，继承自官方的 resnet50
class CustomResNet50(nn.Module):
    def __init__(self, pretrained=False, num_classes=1000, M=None):
        super(CustomResNet50, self).__init__()
        self.M = M
        # 加载预训练的ResNet50模型
        self.resnet = resnet50(pretrained=pretrained)
        self.num_classes = num_classes
        # 自定义分类层
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
        self.fc = self.resnet.fc
        self.spp = SPP(M=self.M)
        
    # 重写 _forward_impl 方法，修改输出结构
    def _forward_impl(self, x):
        # 保留原始结构中的前向操作
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        stem = x
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        feat1 = self.resnet.layer1(x)
        feat2 = self.resnet.layer2(feat1)
        feat3 = self.resnet.layer3(feat2)
        feat4 = self.resnet.layer4(feat3)

        out_relu = F.relu(feat4)

        x = self.resnet.avgpool(out_relu)
        x = torch.flatten(x, 1)

        avg = x
        # 返回最终的分类结果
        out = self.resnet.fc(x)

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

        x_spp, x_strength = self.spp(out_relu)

        # Reshape the SPP features to pass through the fully connected layer
        x_spp = x_spp.permute((2, 0, 1))  # Change the dimensions
        m, b, c = x_spp.shape[0], x_spp.shape[1], x_spp.shape[2]
        x_spp = torch.reshape(x_spp, (m * b, c))

        # Get patch-level scores from fully connected layer
        patch_score = self.fc(x_spp)
        patch_score = torch.reshape(patch_score, (m, b, self.fc.out_features))
        patch_score = patch_score.permute((1, 2, 0))  # Reshape back to original order

        return out, feats, patch_score

    # 使用 forward 来调用 _forward_impl
    def forward(self, x):
        return self._forward_impl(x)
