"""
改进版训练脚本 - 添加数据增强和学习率调度
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image, ImageEnhance
import torchvision.transforms as transforms
from tqdm import tqdm
import random

# 导入模型
sys.path.append(str(Path(__file__).parent / 'models'))
sys.path.append(str(Path(__file__).parent / 'dataset'))
from corner_detector import PPTCornerDetector, CornerDetectionLoss
from prepare_data import AnnotationDataset


class AugmentedPPTDataset(Dataset):
    """带数据增强的 PyTorch Dataset"""

    def __init__(self, annotation_dataset, input_size=512, augment=False):
        self.dataset = annotation_dataset
        self.input_size = input_size
        self.augment = augment

        # 基础转换
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.dataset)

    def augment_image_and_corners(self, image, corners):
        """数据增强 - 图片和角点坐标一起变换"""
        width, height = image.size

        # 1. 随机旋转 (-5 到 +5 度)
        if random.random() > 0.5:
            angle = random.uniform(-5, 5)
            image = image.rotate(angle, expand=False, fillcolor=(255, 255, 255))

            # 旋转角点坐标
            center_x, center_y = width / 2, height / 2
            rad = np.radians(angle)
            cos_val, sin_val = np.cos(rad), np.sin(rad)

            for i in range(len(corners)):
                x, y = corners[i]
                # 平移到原点
                x -= center_x
                y -= center_y
                # 旋转
                new_x = x * cos_val - y * sin_val
                new_y = x * sin_val + y * cos_val
                # 平移回来
                corners[i] = [new_x + center_x, new_y + center_y]

        # 2. 随机亮度调整 (0.8 - 1.2)
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(factor)

        # 3. 随机对比度调整 (0.8 - 1.2)
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(factor)

        # 4. 随机颜色抖动 (0.9 - 1.1)
        if random.random() > 0.5:
            factor = random.uniform(0.9, 1.1)
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(factor)

        # 5. 随机锐度 (0.8 - 1.2)
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(factor)

        return image, corners

    def __getitem__(self, idx):
        # 加载图片和标注
        image_array, corners, (width, height) = self.dataset[idx]

        # PIL Image
        image = Image.fromarray(image_array)
        corners = corners.copy()

        # 数据增强
        if self.augment:
            image, corners = self.augment_image_and_corners(image, corners)

        # 归一化坐标 (0-1 范围)
        corners_norm = corners.copy()
        corners_norm[:, 0] /= width   # X
        corners_norm[:, 1] /= height  # Y

        # 裁剪到 [0, 1] 范围（防止旋转后超出边界）
        corners_norm = np.clip(corners_norm, 0, 1)

        # 缩放到目标尺寸
        image = image.resize((self.input_size, self.input_size), Image.BILINEAR)

        # 转换为 Tensor
        image_tensor = self.to_tensor(image)
        corners_tensor = torch.from_numpy(corners_norm).flatten().float()  # [8]

        return image_tensor, corners_tensor


def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    total_coord_loss = 0.0

    pbar = tqdm(dataloader, desc='Training')
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)

        # 前向传播
        pred_coords, pred_conf = model(images)

        # 计算损失
        losses = criterion(pred_coords, pred_conf, targets)
        loss = losses['total_loss']

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # 统计
        total_loss += loss.item()
        total_coord_loss += losses['coord_loss'].item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'coord': f'{losses["coord_loss"].item():.4f}'
        })

    return total_loss / len(dataloader), total_coord_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0.0
    pixel_errors = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            # 前向传播
            pred_coords, pred_conf = model(images)

            # 计算损失
            losses = criterion(pred_coords, pred_conf, targets)
            total_loss += losses['total_loss'].item()

            # 计算像素误差（假设图片尺寸 512）
            pred_coords_np = pred_coords.cpu().numpy()
            targets_np = targets.cpu().numpy()

            # 反归一化到像素坐标
            pred_pixels = pred_coords_np * 512
            target_pixels = targets_np * 512

            # 计算每个角点的误差
            pred_pixels_reshaped = pred_pixels.reshape(-1, 4, 2)
            target_pixels_reshaped = target_pixels.reshape(-1, 4, 2)

            errors = np.linalg.norm(pred_pixels_reshaped - target_pixels_reshaped, axis=2)
            pixel_errors.extend(errors.flatten())

    avg_loss = total_loss / len(dataloader)
    avg_pixel_error = np.mean(pixel_errors)

    return avg_loss, avg_pixel_error


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='改进版训练脚本')
    parser.add_argument('--data-root', type=str, default='../AnnotationTool/data',
                        help='数据根目录')
    parser.add_argument('--epochs', type=int, default=30,
                        help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='初始学习率')
    parser.add_argument('--input-size', type=int, default=512,
                        help='输入图片尺寸')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='设备')
    parser.add_argument('--save-path', type=str, default='checkpoints/model_enhanced.pth',
                        help='模型保存路径')
    parser.add_argument('--augment', action='store_true', default=True,
                        help='使用数据增强')
    parser.add_argument('--scheduler', action='store_true', default=True,
                        help='使用学习率调度')

    args = parser.parse_args()

    print("="*60)
    print("  PPT Corner Detector - Enhanced Training")
    print("="*60)
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Input size: {args.input_size}")
    print(f"Data augmentation: {args.augment}")
    print(f"LR scheduler: {args.scheduler}")
    print("="*60)

    # 加载数据
    print("\n[1/5] Loading dataset...")
    annotation_dataset = AnnotationDataset(args.data_root)

    if len(annotation_dataset) == 0:
        print("[ERROR] No data found!")
        return

    print(f"Total samples: {len(annotation_dataset)}")

    # 划分训练集和验证集
    train_dataset_raw, val_dataset_raw = annotation_dataset.split_train_val(val_ratio=0.15)

    # 创建 PyTorch Dataset（训练集使用增强，验证集不使用）
    train_dataset = AugmentedPPTDataset(train_dataset_raw, input_size=args.input_size, augment=args.augment)
    val_dataset = AugmentedPPTDataset(val_dataset_raw, input_size=args.input_size, augment=False)

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # 创建模型
    print("\n[2/5] Creating model...")
    device = torch.device(args.device)
    model = PPTCornerDetector(pretrained=True).to(device)

    # 损失函数和优化器
    print("\n[3/5] Setting up training...")
    criterion = CornerDetectionLoss(coord_weight=1.0, order_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 学习率调度器 - 余弦退火
    scheduler = None
    if args.scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
        print(f"Using CosineAnnealingLR scheduler (eta_min={args.lr * 0.01})")

    # 训练
    print(f"\n[4/5] Training for {args.epochs} epochs...")
    print("-"*60)

    best_val_loss = float('inf')
    best_pixel_error = float('inf')

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        # 显示当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {current_lr:.6f}")

        # 训练
        train_loss, train_coord_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # 验证
        val_loss, val_pixel_error = validate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} (Coord: {train_coord_loss:.4f})")
        print(f"Val Loss: {val_loss:.4f}, Pixel Error: {val_pixel_error:.2f}px")

        # 更新学习率
        if scheduler:
            scheduler.step()

        # 保存最佳模型
        if val_pixel_error < best_pixel_error:
            best_val_loss = val_loss
            best_pixel_error = val_pixel_error
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_pixel_error': val_pixel_error,
            }, args.save_path)
            print(f"[SAVE] Best model saved (pixel_error: {val_pixel_error:.2f}px)")

    print("\n"+"="*60)
    print(f"[5/5] Training completed!")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Best pixel error: {best_pixel_error:.2f}px")
    print(f"Model saved to: {args.save_path}")
    print("="*60)

    print(f"\nNext step:")
    print(f"  python models/export_onnx.py {args.save_path} --output ppt_corner_detector.onnx")


if __name__ == '__main__':
    main()
