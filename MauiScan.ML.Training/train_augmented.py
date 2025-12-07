"""
数据增强版训练脚本
- 随机亮度/对比度/饱和度
- 随机模糊
- 随机噪声
- 保持角点坐标不变（不做几何变换）
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torchvision.transforms as transforms
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent / 'models'))
sys.path.append(str(Path(__file__).parent / 'dataset'))
from corner_detector import PPTCornerDetector, CornerDetectionLoss
from prepare_data import AnnotationDataset


class AugmentedPPTDataset(Dataset):
    """数据增强 + 内存预加载"""

    def __init__(self, annotation_dataset, input_size=512, clamp_eps=0.02, augment=True):
        self.input_size = input_size
        self.clamp_eps = clamp_eps
        self.augment = augment

        print(f"  预加载 {len(annotation_dataset)} 张图片到内存...")
        self.cached_data = []

        for idx in tqdm(range(len(annotation_dataset)), desc='  Loading'):
            image_array, corners, (width, height) = annotation_dataset[idx]
            image = Image.fromarray(image_array)

            # 归一化坐标
            corners_norm = corners.copy()
            corners_norm[:, 0] /= width
            corners_norm[:, 1] /= height
            corners_norm = np.clip(corners_norm, self.clamp_eps, 1.0 - self.clamp_eps)

            # 缓存原始图片（PIL格式）+ 坐标
            self.cached_data.append((image, corners_norm))

        print(f"  完成！所有数据已加载到内存")

    def __len__(self):
        return len(self.cached_data)

    def __getitem__(self, idx):
        image, corners_norm = self.cached_data[idx]

        # ⭐ 数据增强（仅对图片，角点坐标不变）
        if self.augment:
            image = self._augment_image(image)

        # Resize + ToTensor
        image = image.resize((self.input_size, self.input_size), Image.BILINEAR)
        image_tensor = transforms.ToTensor()(image)

        corners_tensor = torch.from_numpy(corners_norm).flatten().float()

        return image_tensor, corners_tensor

    def _augment_image(self, image):
        """
        图像增强（不改变几何结构）
        """
        # 1. 随机亮度 (0.7 - 1.3)
        if np.random.rand() < 0.5:
            factor = np.random.uniform(0.7, 1.3)
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(factor)

        # 2. 随机对比度 (0.7 - 1.3)
        if np.random.rand() < 0.5:
            factor = np.random.uniform(0.7, 1.3)
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(factor)

        # 3. 随机饱和度 (0.7 - 1.3)
        if np.random.rand() < 0.5:
            factor = np.random.uniform(0.7, 1.3)
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(factor)

        # 4. 随机锐度 (0.5 - 1.5)
        if np.random.rand() < 0.3:
            factor = np.random.uniform(0.5, 1.5)
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(factor)

        # 5. 随机模糊
        if np.random.rand() < 0.2:
            radius = np.random.uniform(0.5, 2.0)
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))

        return image


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
            pred_coords, pred_conf = model(images)
            losses = criterion(pred_coords, pred_conf, targets)
            loss = losses['total_loss']

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        total_coord_loss += losses['coord_loss'].item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'coord': f'{losses["coord_loss"].item():.4f}'
        })

    return total_loss / len(dataloader), total_coord_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """验证（不增强）"""
    model.eval()
    total_loss = 0.0
    pixel_errors = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            pred_coords, pred_conf = model(images)
            losses = criterion(pred_coords, pred_conf, targets)
            total_loss += losses['total_loss'].item()

            # 像素误差
            pred_coords_np = pred_coords.cpu().numpy()
            targets_np = targets.cpu().numpy()

            pred_pixels = pred_coords_np * 512
            target_pixels = targets_np * 512

            pred_pixels_reshaped = pred_pixels.reshape(-1, 4, 2)
            target_pixels_reshaped = target_pixels.reshape(-1, 4, 2)

            errors = np.linalg.norm(pred_pixels_reshaped - target_pixels_reshaped, axis=2)
            pixel_errors.extend(errors.flatten())

    avg_loss = total_loss / len(dataloader)
    avg_pixel_error = np.mean(pixel_errors)

    return avg_loss, avg_pixel_error


def main():
    import argparse

    parser = argparse.ArgumentParser(description='数据增强训练')
    parser.add_argument('--data-root', type=str, default='../AnnotationTool/data',
                        help='数据根目录')
    parser.add_argument('--epochs', type=int, default=300,
                        help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='初始学习率')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='设备')
    parser.add_argument('--save-path', type=str, default='checkpoints/model_augmented.pth',
                        help='模型保存路径')
    parser.add_argument('--resume', type=str, default=None,
                        help='从checkpoint继续训练')

    args = parser.parse_args()

    print("="*60)
    print("  数据增强训练")
    print("="*60)
    print(f"设备: {args.device}")
    print(f"训练轮数: {args.epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
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

    # 创建 PyTorch Dataset（训练集增强，验证集不增强）
    train_dataset = AugmentedPPTDataset(train_dataset_raw, input_size=512, augment=True)
    val_dataset = AugmentedPPTDataset(val_dataset_raw, input_size=512, augment=False)

    # 创建 DataLoader
    # ⭐ 启用多进程加速数据增强
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)

    print(f"训练集: {len(train_dataset)} 样本（增强）")
    print(f"验证集: {len(val_dataset)} 样本（不增强）")

    # 创建模型
    print("\n[2/5] 创建模型...")
    device = torch.device(args.device)
    model = PPTCornerDetector(pretrained=True).to(device)

    # 损失函数
    print("\n[3/5] 设置训练...")
    criterion = CornerDetectionLoss(coord_weight=1.0, order_weight=0.0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )

    # 混合精度
    scaler = torch.cuda.amp.GradScaler() if (device.type == 'cuda') else None
    if scaler:
        print(f"混合精度训练: ENABLED")

    # 恢复训练
    start_epoch = 0
    best_pixel_error = float('inf')

    if args.resume:
        print(f"\n[Resume] 加载 checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_pixel_error = checkpoint.get('val_pixel_error', float('inf'))
        print(f"  从 Epoch {start_epoch} 继续训练")
        print(f"  当前最佳误差: {best_pixel_error:.2f}px")

    # 训练
    print(f"\n[4/5] 开始训练...")
    print("-"*60)

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

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

        # 保存最佳模型
        if val_pixel_error < best_pixel_error:
            best_pixel_error = val_pixel_error
            import os
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_pixel_error': val_pixel_error,
            }, args.save_path)
            print(f"[保存] 最佳模型 ({val_pixel_error:.2f}px)")

    print("\n"+"="*60)
    print(f"[5/5] 训练完成!")
    print(f"最佳像素误差: {best_pixel_error:.2f}px")
    print(f"模型保存到: {args.save_path}")
    print("="*60)


if __name__ == '__main__':
    main()
