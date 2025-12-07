"""
简化版训练脚本 - 用于测试 50+ 张图片的小规模训练
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

# 导入模型
sys.path.append(str(Path(__file__).parent / 'models'))
sys.path.append(str(Path(__file__).parent / 'dataset'))
from corner_detector import PPTCornerDetector, CornerDetectionLoss
from prepare_data import AnnotationDataset


class PPTDataset(Dataset):
    """PyTorch Dataset 封装"""

    def __init__(self, annotation_dataset, input_size=512, augment=False):
        self.dataset = annotation_dataset
        self.input_size = input_size
        self.augment = augment

        # 图片预处理
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),  # 转换为 [0, 1] 并变为 CHW 格式
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 加载图片和标注
        image_array, corners, (width, height) = self.dataset[idx]

        # PIL Image
        image = Image.fromarray(image_array)

        # 归一化坐标 (0-1 范围)
        corners_norm = corners.copy()
        corners_norm[:, 0] /= width   # X
        corners_norm[:, 1] /= height  # Y

        # 转换为 Tensor
        image_tensor = self.transform(image)
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

    parser = argparse.ArgumentParser(description='简化训练脚本')
    parser.add_argument('--data-root', type=str, default='../AnnotationTool/data',
                        help='数据根目录')
    parser.add_argument('--epochs', type=int, default=10,
                        help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--input-size', type=int, default=512,
                        help='输入图片尺寸')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='设备')
    parser.add_argument('--save-path', type=str, default='checkpoints/model_test.pth',
                        help='模型保存路径')

    args = parser.parse_args()

    print("="*60)
    print("  PPT Corner Detector - Simple Training")
    print("="*60)
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Input size: {args.input_size}")
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

    # 创建 PyTorch Dataset
    train_dataset = PPTDataset(train_dataset_raw, input_size=args.input_size, augment=False)
    val_dataset = PPTDataset(val_dataset_raw, input_size=args.input_size, augment=False)

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 创建模型
    print("\n[2/5] Creating model...")
    device = torch.device(args.device)
    model = PPTCornerDetector(pretrained=True).to(device)

    # 损失函数和优化器
    print("\n[3/5] Setting up training...")
    criterion = CornerDetectionLoss(coord_weight=1.0, order_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 训练
    print(f"\n[4/5] Training for {args.epochs} epochs...")
    print("-"*60)

    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        # 训练
        train_loss, train_coord_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # 验证
        val_loss, val_pixel_error = validate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} (Coord: {train_coord_loss:.4f})")
        print(f"Val Loss: {val_loss:.4f}, Pixel Error: {val_pixel_error:.2f}px")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_pixel_error': val_pixel_error,
            }, args.save_path)
            print(f"[SAVE] Best model saved (val_loss: {val_loss:.4f})")

    print("\n"+"="*60)
    print(f"[5/5] Training completed!")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {args.save_path}")
    print("="*60)

    print(f"\nNext step:")
    print(f"  python models/export_onnx.py {args.save_path} --output ppt_corner_detector.onnx")


if __name__ == '__main__':
    main()
