import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
from pathlib import Path

# ------------------- 1. 数据加载器 -------------------

class CelebADataset:
    """仅用于加载属性名称，不需要完整数据集"""

    def __init__(self, project_root):
        # 使用 pathlib 构建属性文件的绝对路径
        script_dir = Path(project_root)
        attr_file = script_dir /'algorithms' /'list_attr_celeba.txt'

        # 检查文件是否存在
        if not attr_file.exists():
            raise FileNotFoundError(f"文件 {attr_file} 不存在。请检查文件路径。")

        try:
            # 读取属性文件
            with attr_file.open('r') as f:
                lines = f.readlines()

            # 获取属性名称
            self.attr_names = lines[1].strip().split()
        except Exception as e:
            print(f"读取文件时出现错误: {e}")


# ------------------- 2. 模型组件 -------------------
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return torch.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return torch.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x


# ------------------- 3. 主模型 -------------------
class EnhancedAttributeClassifier(nn.Module):
    def __init__(self, num_attributes):
        super().__init__()
        # 加载预训练模型
        base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # 保留特征提取层
        self.features = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,  # 64 channels
            base_model.layer2,  # 128 channels
            # 替换后续层为增强结构
            self._make_enhanced_block(128, 256),  # 替换layer3
            self._make_enhanced_block(256, 512),  # 替换layer4
        )

        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_attributes),
        )

    def _make_enhanced_block(self, in_channels, out_channels):
        return nn.Sequential(
            # 深度可分离卷积替代标准卷积
            DepthwiseSeparableConv(in_channels, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            # 添加CBAM注意力机制
            CBAM(out_channels),
            # 额外的卷积层增强特征提取
            DepthwiseSeparableConv(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ------------------- 4. 预测函数 -------------------
def predict_attributes(model, image, attr_names, transform, device):
    """
    预测图像属性

    Args:
        model: 训练好的属性分类模型
        image: PIL Image 对象或图像路径
        attr_names: 属性名称列表
        transform: 图像预处理转换
        device: 运行设备

    Returns:
        list: 排序后的属性预测结果 [(属性名, 概率), ...]
    """
    model.eval()

    # 如果传入的是图像路径，加载图像
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')

    # 预处理图像
    input_tensor = transform(image).unsqueeze(0).to(device)

    # 预测
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.sigmoid(output).squeeze()

    # 获取预测结果
    predicted_attrs = []
    for i, prob in enumerate(probs):
        predicted_attrs.append((attr_names[i], prob.item()))

    # 按概率排序
    predicted_attrs.sort(key=lambda x: x[1], reverse=True)

    return predicted_attrs


# ------------------- 5. 初始化函数（替代main） -------------------
def initialize_face_attribute_model(project_root, model_path_override=None):
    """
    初始化人脸属性识别模型

    Args:
        project_root: 项目根目录路径
        model_path_override: 可选的模型文件路径，若提供则使用该路径

    Returns:
        tuple: (模型, 属性名称列表, 预处理转换, 设备)
    """
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载属性名称
    dataset = CelebADataset(project_root)
    attr_names = dataset.attr_names

    # 创建模型
    model = EnhancedAttributeClassifier(num_attributes=len(attr_names))

    # 加载预训练权重
    if model_path_override:
        model_path = Path(model_path_override)
    else:
        model_path = Path(project_root) / 'best_attribute_model.pth'

    if not model_path.exists():
        raise FileNotFoundError(f"模型文件 {model_path} 不存在。请检查文件路径。")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # 定义预处理转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return model, attr_names, transform, device