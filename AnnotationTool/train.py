"""
幕布边框检测模型训练脚本

使用 MobileNetV3-Small 作为骨干网络，训练角点回归模型
"""

import os
import json
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from torchvision import models, transforms
from tqdm import tqdm


class ScreenCornerDataset(Dataset):
    """幕布角点数据集（支持内存缓存）"""

    def __init__(self, images_dir, labels_dir, transform=None, input_size=224, cache_in_memory=False):
        """
        Args:
            images_dir: 图片目录
            labels_dir: 标签目录
            transform: 图片变换
            input_size: 模型输入尺寸
            cache_in_memory: 是否将图片缓存到内存中
        """
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transform = transform
        self.input_size = input_size
        self.cache_in_memory = cache_in_memory

        # 扫描所有图片
        self.image_files = sorted(list(self.images_dir.glob('*.jpg')))
        print(f"加载 {len(self.image_files)} 张图片")

        # 如果启用缓存，预加载所有图片
        if self.cache_in_memory:
            print("正在缓存图片到内存...")
            self.cached_images = []
            self.cached_targets = []
            self.original_sizes = []

            for img_path in self.image_files:
                # 读取图片
                image = cv2.imread(str(img_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                h, w = image.shape[:2]
                self.original_sizes.append((w, h))

                # 读取标签
                label_path = self.labels_dir / f"{img_path.stem}.txt"
                with open(label_path, 'r') as f:
                    label_str = f.read().strip()

                # 解析角点坐标（保存原始像素坐标）
                coords = label_str.split(',')
                corners = []
                for i in range(0, len(coords), 2):
                    corners.append(float(coords[i]))
                    corners.append(float(coords[i + 1]))

                self.cached_images.append(image)
                self.cached_targets.append(corners)

            # 估算内存占用
            total_pixels = sum(img.shape[0] * img.shape[1] for img in self.cached_images)
            estimated_mb = total_pixels * 3 / 1024 / 1024  # 3 channels (RGB)
            print(f"图片缓存完成！占用内存约: {estimated_mb:.1f} MB")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if self.cache_in_memory:
            # 从内存中获取图片
            image = self.cached_images[idx]  # 不需要 copy，因为 transform 会创建新的 tensor
            w, h = self.original_sizes[idx]
            target_coords = self.cached_targets[idx]
        else:
            # 从磁盘读取图片
            img_path = self.image_files[idx]
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]

            # 读取标签
            label_path = self.labels_dir / f"{img_path.stem}.txt"
            with open(label_path, 'r') as f:
                label_str = f.read().strip()

            # 解析角点坐标
            coords = label_str.split(',')
            target_coords = []
            for i in range(0, len(coords), 2):
                target_coords.append(float(coords[i]))
                target_coords.append(float(coords[i + 1]))

        # 归一化角点坐标
        corners = []
        for i in range(0, len(target_coords), 2):
            x = target_coords[i] / w
            y = target_coords[i + 1] / h
            corners.extend([x, y])

        # 应用变换
        if self.transform:
            image = self.transform(image)

        # 转换为 tensor
        target = torch.tensor(corners, dtype=torch.float32)

        return image, target


class MobileNetV3CornerNet(nn.Module):
    """MobileNetV3 角点回归网络"""

    def __init__(self, num_corners=4, pretrained=True):
        """
        Args:
            num_corners: 角点数量
            pretrained: 是否使用预训练权重
        """
        super(MobileNetV3CornerNet, self).__init__()

        # 加载 MobileNetV3-Small
        mobilenet = models.mobilenet_v3_small(pretrained=pretrained)

        # 移除分类器
        self.features = mobilenet.features

        # 自定义回归头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.regressor = nn.Sequential(
            nn.Linear(576, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, num_corners * 2)  # 4 个角点，每个 2 个坐标
        )

    def forward(self, x):
        # 特征提取
        x = self.features(x)

        # 全局平均池化
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # 回归
        corners = self.regressor(x)

        # 限制在 [0, 1] 范围内
        corners = torch.sigmoid(corners)

        return corners


class WingLoss(nn.Module):
    """Wing Loss - 对异常值更鲁棒的损失函数"""

    def __init__(self, omega=10, epsilon=2):
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon

    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        c = self.omega * (1.0 - np.log(1.0 + self.omega / self.epsilon))

        loss = torch.where(
            diff < self.omega,
            self.omega * torch.log(1.0 + diff / self.epsilon),
            diff - c
        )
        return loss.mean()


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """训练一个 epoch"""
    model.train()
    running_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", ncols=100)
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)

        # 前向传播
        optimizer.zero_grad()
        outputs = model(images)

        # 计算损失
        loss = criterion(outputs, targets)

        # 反向传播
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    avg_loss = running_loss / len(dataloader)
    return avg_loss


def validate(model, dataloader, criterion, device):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    corner_errors = []

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Validation", ncols=100):
            images = images.to(device)
            targets = targets.to(device)

            # 前向传播
            outputs = model(images)

            # 计算损失
            loss = criterion(outputs, targets)
            running_loss += loss.item()

            # 计算角点误差（像素）
            outputs_np = outputs.cpu().numpy()
            targets_np = targets.cpu().numpy()

            for i in range(outputs_np.shape[0]):
                for j in range(0, 8, 2):
                    error = np.sqrt((outputs_np[i, j] - targets_np[i, j]) ** 2 +
                                   (outputs_np[i, j + 1] - targets_np[i, j + 1]) ** 2)
                    corner_errors.append(error)

    avg_loss = running_loss / len(dataloader)
    avg_error = np.mean(corner_errors) * 224  # 转换为像素（假设输入 224x224）

    return avg_loss, avg_error


def main():
    # 配置
    data_root = r'D:\Programing\C#\MauiScan\AnnotationTool\training_data'
    batch_size = 32  # 优化：增大 batch size 以充分利用 GPU
    num_epochs = 50
    learning_rate = 1e-4
    input_size = 224
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"使用设备: {device}")

    # 数据变换
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((input_size, input_size)),
        transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集
    print("加载数据集...")
    train_dataset = ScreenCornerDataset(
        os.path.join(data_root, 'images', 'train'),
        os.path.join(data_root, 'labels', 'train'),
        transform=transform_train,
        input_size=input_size,
        cache_in_memory=False  # 数据太大，不缓存
    )

    val_dataset = ScreenCornerDataset(
        os.path.join(data_root, 'images', 'val'),
        os.path.join(data_root, 'labels', 'val'),
        transform=transform_val,
        input_size=input_size,
        cache_in_memory=False  # 验证集不缓存（数据量小，从磁盘读取也很快）
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # 多线程数据加载
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True  # 保持 worker 进程
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        persistent_workers=True
    )

    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")

    # 创建模型
    print("创建模型...")
    model = MobileNetV3CornerNet(num_corners=4, pretrained=True)
    model = model.to(device)

    # 损失函数和优化器
    criterion = WingLoss()  # 使用 Wing Loss
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # 训练循环
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    print(f"\n开始训练 (共 {num_epochs} epochs)...\n")

    for epoch in range(1, num_epochs + 1):
        # 训练
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        train_losses.append(train_loss)

        # 验证
        val_loss, val_error = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        # 更新学习率
        scheduler.step()

        print(f"\nEpoch {epoch}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Error: {val_error:.2f} px")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = r'D:\Programing\C#\MauiScan\AnnotationTool\best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_error': val_error
            }, model_path)
            print(f"  [*] 保存最佳模型 (val_loss: {val_loss:.4f})")

        print()

    print("训练完成!")

    # 保存最终模型
    final_model_path = r'D:\Programing\C#\MauiScan\AnnotationTool\final_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses
    }, final_model_path)
    print(f"最终模型已保存: {final_model_path}")

    # 导出为 ONNX
    print("\n导出 ONNX 模型...")
    onnx_path = r'D:\Programing\C#\MauiScan\AnnotationTool\screen_detector.onnx'
    dummy_input = torch.randn(1, 3, input_size, input_size).to(device)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    print(f"ONNX 模型已保存: {onnx_path}")


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    main()
