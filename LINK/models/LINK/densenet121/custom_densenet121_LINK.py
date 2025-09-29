import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121  # 这里你可以换成任何需要的DenseNet型号

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
    
class CustomDenseNet121(nn.Module):
    def __init__(self, pretrained=False, num_classes=1000, M = None):
        super(CustomDenseNet121, self).__init__()
        self.M=M
        # 加载预训练的DenseNet模型
        self.densenet = densenet121(pretrained=pretrained)
        self.num_classes = num_classes
        # 如果需要修改分类器，可以自定义分类层
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_features, num_classes)
        self.spp = SPP(M=self.M)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 获取输入的特征
        stem = self.densenet.features[0:4](x)  # 初始stem部分
        feat1 = self.densenet.features[4:6](stem)  # DenseBlock1 输出
        feat2 = self.densenet.features[6:8](feat1)  # DenseBlock2 输出
        feat3 = self.densenet.features[8:10](feat2)  # DenseBlock3 输出
        feat4 = self.densenet.features[10:](feat3)  # DenseBlock4 输出
        
        out_relu = F.relu(feat4, inplace=True)
        out = F.adaptive_avg_pool2d(out_relu, (1, 1))  # 自适应平均池化
        out = torch.flatten(out, 1)  # 展平
        logits = self.densenet.classifier(out)

        # 保存特征信息
        feats = {
            "pooled_feat": out,
            "feats": [
                F.relu(stem),
                F.relu(feat1),
                F.relu(feat2),
                F.relu(feat3),
                F.relu(feat4),
            ],
            "preact_feats": [stem, feat1, feat2, feat3, feat4],
        }
                
        x_spp, x_strength = self.spp(out_relu)
        
        x_spp = x_spp.permute((2, 0, 1))
        m, b, c = x_spp.shape[0], x_spp.shape[1], x_spp.shape[2]
        x_spp = torch.reshape(x_spp, (m * b, c))
        # patch_score = self.fc(x_spp)
        patch_score = self.densenet.classifier(x_spp) 
        # patch_score = torch.reshape(patch_score, (m, b, 1000))
        patch_score = torch.reshape(patch_score, (m, b, self.num_classes))
        patch_score = patch_score.permute((1, 2, 0))
        
        return logits, feats, patch_score

# # 使用自定义模型
# model = CustomDenseNet(pretrained=True, num_classes=1000)
