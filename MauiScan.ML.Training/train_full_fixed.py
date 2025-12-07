"""
完整数据集训练 - Fixed版本配置
- 关闭Order Loss
- GT坐标clamp到[0.02, 0.98]
- 使用Sigmoid（不用Tanh，因为Fixed版本表现更好）
- 合理的学习率调度
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
import json

# 导入模型和数据集
sys.path.append(str(Path(__file__).parent / 'models'))
sys.path.append(str(Path(__file__).parent / 'dataset'))
from corner_detector import PPTCornerDetector, CornerDetectionLoss
from prepare_data import AnnotationDataset


class PPTDatasetFixed(Dataset):
    """Fixed版本数据集 - GT坐标clamp + 内存预加载"""

    def __init__(self, annotation_dataset, input_size=512, clamp_eps=0.02):
        self.input_size = input_size
        self.clamp_eps = clamp_eps

        # 图片预处理
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
        ])

        # ⭐ 预加载所有数据到内存
        print(f"  预加载 {len(annotation_dataset)} 张图片到内存...")
        self.cached_data = []

        for idx in tqdm(range(len(annotation_dataset)), desc='  Loading'):
            # 加载图片和标注
            image_array, corners, (width, height) = annotation_dataset[idx]
            image = Image.fromarray(image_array)

            # 归一化坐标
            corners_norm = corners.copy()
            corners_norm[:, 0] /= width
            corners_norm[:, 1] /= height
            corners_norm = np.clip(corners_norm, self.clamp_eps, 1.0 - self.clamp_eps)

            # 转换为 Tensor 并缓存
            image_tensor = transform(image)
            corners_tensor = torch.from_numpy(corners_norm).flatten().float()

            self.cached_data.append((image_tensor, corners_tensor))

        print(f"  完成！所有数据已加载到内存")

    def __len__(self):
        return len(self.cached_data)

    def __getitem__(self, idx):
        # 直接从内存返回
        return self.cached_data[idx]


def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_coord_loss = 0.0

    pbar = tqdm(dataloader, desc='Training')
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)

        # 混合精度训练
        if scaler is not None:
            with torch.cuda.amp.autocast():
                pred_coords, pred_conf = model(images)
                losses = criterion(pred_coords, pred_conf, targets)
                loss = losses['total_loss']

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # 标准训练
            pred_coords, pred_conf = model(images)
            losses = criterion(pred_coords, pred_conf, targets)
            loss = losses['total_loss']

            optimizer.zero_grad()
            loss.backward()
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

    parser = argparse.ArgumentParser(description='完整数据集训练 - Fixed配置')
    parser.add_argument('--data-root', type=str, default='../AnnotationTool/data',
                        help='数据根目录')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='批次大小（GPU环境建议32-64）')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='初始学习率')
    parser.add_argument('--input-size', type=int, default=512,
                        help='输入图片尺寸')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='设备')
    parser.add_argument('--save-path', type=str, default='checkpoints/model_full_fixed.pth',
                        help='模型保存路径')
    parser.add_argument('--clamp-eps', type=float, default=0.02,
                        help='GT坐标clamp范围')
    parser.add_argument('--amp', action='store_true', default=True,
                        help='使用混合精度训练（自动启用）')

    args = parser.parse_args()

    print("="*60)
    print("  完整数据集训练 - Fixed配置")
    print("="*60)
    print(f"设备: {args.device}")
    print(f"训练轮数: {args.epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"输入尺寸: {args.input_size}")
    print(f"GT Clamp: [{args.clamp_eps:.2f}, {1-args.clamp_eps:.2f}]")
    print("="*60)

    # 加载数据
    print("\n[1/5] 加载数据集...")
    annotation_dataset = AnnotationDataset(args.data_root)

    if len(annotation_dataset) == 0:
        print("[ERROR] 没有找到数据！")
        return

    print(f"总样本数: {len(annotation_dataset)}")

    # 划分训练集和验证集
    train_dataset_raw, val_dataset_raw = annotation_dataset.split_train_val(val_ratio=0.15)

    # 创建 PyTorch Dataset
    train_dataset = PPTDatasetFixed(train_dataset_raw, input_size=args.input_size, clamp_eps=args.clamp_eps)
    val_dataset = PPTDatasetFixed(val_dataset_raw, input_size=args.input_size, clamp_eps=args.clamp_eps)

    # 创建 DataLoader
    # ⭐ 数据已预加载到内存，不需要多进程
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")

    # 创建模型
    print("\n[2/5] 创建模型...")
    device = torch.device(args.device)
    model = PPTCornerDetector(pretrained=True).to(device)

    # ⭐ 损失函数：关闭Order Loss
    print("\n[3/5] 设置训练...")
    criterion = CornerDetectionLoss(coord_weight=1.0, order_weight=0.0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # ⭐ 学习率调度器：余弦退火
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )

    print(f"Order Loss: DISABLED (weight=0)")
    print(f"LR Scheduler: CosineAnnealingLR (eta_min={args.lr * 0.01:.6f})")

    # ⭐ 混合精度训练 Scaler
    scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type == 'cuda') else None
    if scaler:
        print(f"混合精度训练: ENABLED (FP16 + FP32)")
    else:
        print(f"混合精度训练: DISABLED")

    # 训练
    print(f"\n[4/5] 开始训练 {args.epochs} 轮...")
    print("-"*60)

    best_val_loss = float('inf')
    best_pixel_error = float('inf')

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        # 显示当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f"学习率: {current_lr:.6f}")

        # 训练
        train_loss, train_coord_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler)

        # 验证
        val_loss, val_pixel_error = validate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} (Coord: {train_coord_loss:.4f})")
        print(f"Val Loss: {val_loss:.4f}, Pixel Error: {val_pixel_error:.2f}px")

        # 更新学习率
        scheduler.step()

        # 保存最佳模型（按像素误差）
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
            print(f"[保存] 最佳模型 (像素误差: {val_pixel_error:.2f}px)")

    print("\n"+"="*60)
    print(f"[5/5] 训练完成!")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"最佳像素误差: {best_pixel_error:.2f}px")
    print(f"模型保存到: {args.save_path}")
    print("="*60)

    # 导出为ONNX
    print(f"\n下一步：")
    print(f"  python models/export_onnx.py {args.save_path} --output ppt_corner_detector_full.onnx")


if __name__ == '__main__':
    main()
