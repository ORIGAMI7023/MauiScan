"""
PPT 四角点检测模型定义
使用 MobileNetV3 作为骨干网络
"""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class PPTCornerDetector(nn.Module):
    """PPT 四角点检测模型"""

    def __init__(self, pretrained: bool = True):
        """
        初始化模型

        Args:
            pretrained: 是否使用预训练权重
        """
        super().__init__()

        # 使用 MobileNetV3-Small 作为骨干网络
        if pretrained:
            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
            self.backbone = mobilenet_v3_small(weights=weights)
        else:
            self.backbone = mobilenet_v3_small(weights=None)

        # 移除分类头
        self.backbone.classifier = nn.Identity()

        # 获取骨干网络输出特征维度
        # MobileNetV3-Small 输出 576 维特征
        backbone_out_features = 576

        # 角点坐标回归头
        # 输出 8 个值：4 个角点 × (x, y)
        self.coord_head = nn.Sequential(
            nn.Linear(backbone_out_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 8),  # 4 corners × 2 coordinates
            nn.Sigmoid()  # 归一化到 [0, 1] 范围（允许略微超出）
        )

        # 置信度预测头
        # 输出 1 个值：整体置信度
        self.confidence_head = nn.Sequential(
            nn.Linear(backbone_out_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 输出 [0, 1] 范围的置信度
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入图片 [B, 3, H, W]，归一化到 [0, 1]

        Returns:
            coordinates: 角点坐标 [B, 8]，归一化到 [0, 1]
                         顺序: [x1, y1, x2, y2, x3, y3, x4, y4]
                         (左上, 右上, 右下, 左下)
            confidence: 置信度 [B, 1]
        """
        # 提取特征
        features = self.backbone(x)  # [B, 576]

        # 预测角点坐标
        coordinates = self.coord_head(features)  # [B, 8]

        # 预测置信度
        confidence = self.confidence_head(features)  # [B, 1]

        return coordinates, confidence


class CornerDetectionLoss(nn.Module):
    """角点检测损失函数"""

    def __init__(self, coord_weight: float = 1.0, order_weight: float = 0.5):
        """
        初始化损失函数

        Args:
            coord_weight: 坐标损失权重
            order_weight: 顺序约束损失权重
        """
        super().__init__()
        self.coord_weight = coord_weight
        self.order_weight = order_weight
        self.smooth_l1 = nn.SmoothL1Loss()

    def forward(
        self,
        pred_coords: torch.Tensor,
        pred_conf: torch.Tensor,
        target_coords: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        计算损失

        Args:
            pred_coords: 预测坐标 [B, 8]
            pred_conf: 预测置信度 [B, 1]
            target_coords: 真实坐标 [B, 8]

        Returns:
            losses: 包含各项损失的字典
        """
        # 1. 坐标回归损失 (Smooth L1)
        coord_loss = self.smooth_l1(pred_coords, target_coords)

        # 2. 顺序约束损失
        # 确保角点顺序正确：左上 → 右上 → 右下 → 左下
        pred_coords_reshaped = pred_coords.view(-1, 4, 2)  # [B, 4, 2]

        # 左上 < 右上 (X 坐标)
        order_loss_1 = torch.relu(pred_coords_reshaped[:, 0, 0] - pred_coords_reshaped[:, 1, 0] + 0.1)

        # 左下 < 右下 (X 坐标)
        order_loss_2 = torch.relu(pred_coords_reshaped[:, 3, 0] - pred_coords_reshaped[:, 2, 0] + 0.1)

        # 左上 < 左下 (Y 坐标)
        order_loss_3 = torch.relu(pred_coords_reshaped[:, 0, 1] - pred_coords_reshaped[:, 3, 1] + 0.1)

        # 右上 < 右下 (Y 坐标)
        order_loss_4 = torch.relu(pred_coords_reshaped[:, 1, 1] - pred_coords_reshaped[:, 2, 1] + 0.1)

        order_loss = (order_loss_1 + order_loss_2 + order_loss_3 + order_loss_4).mean()

        # 3. 总损失
        total_loss = self.coord_weight * coord_loss + self.order_weight * order_loss

        return {
            'total_loss': total_loss,
            'coord_loss': coord_loss,
            'order_loss': order_loss,
        }


def test_model():
    """测试模型"""
    print("🧪 测试模型...")

    # 创建模型
    model = PPTCornerDetector(pretrained=False)
    model.eval()

    # 创建虚拟输入
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 512, 512)

    # 前向传播
    with torch.no_grad():
        coords, conf = model(dummy_input)

    print(f"✅ 模型测试成功!")
    print(f"  - 输入形状: {dummy_input.shape}")
    print(f"  - 坐标输出形状: {coords.shape}")
    print(f"  - 置信度输出形状: {conf.shape}")
    print(f"  - 坐标范围: [{coords.min():.3f}, {coords.max():.3f}]")
    print(f"  - 置信度范围: [{conf.min():.3f}, {conf.max():.3f}]")

    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  - 总参数量: {total_params:,} ({total_params / 1e6:.2f}M)")

    # 测试损失函数
    print("\n🧪 测试损失函数...")
    criterion = CornerDetectionLoss()

    target_coords = torch.rand(batch_size, 8)  # 随机目标坐标
    losses = criterion(coords, conf, target_coords)

    print(f"✅ 损失函数测试成功!")
    for name, value in losses.items():
        print(f"  - {name}: {value.item():.4f}")


if __name__ == '__main__':
    test_model()
