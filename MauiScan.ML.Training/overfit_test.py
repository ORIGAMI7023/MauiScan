"""
过拟合测试 - 判断是"学不会"还是"没学对"
在少量图片（3-5张）上疯狂训练，看能否完全拟合
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from tqdm import tqdm
import json

# 导入模型
sys.path.append(str(Path(__file__).parent / 'models'))
sys.path.append(str(Path(__file__).parent / 'dataset'))
from corner_detector import PPTCornerDetector, CornerDetectionLoss


class SmallDataset(Dataset):
    """小数据集 - 直接从指定目录加载"""

    def __init__(self, data_dir, input_size=512):
        self.data_dir = Path(data_dir)
        self.input_size = input_size
        self.samples = []

        # 查找所有有标注的图片
        for img_path in self.data_dir.glob("*.png"):
            json_path = img_path.with_suffix('.json')
            if json_path.exists():
                self.samples.append((img_path, json_path))

        for img_path in self.data_dir.glob("*.jpg"):
            json_path = img_path.with_suffix('.json')
            if json_path.exists():
                self.samples.append((img_path, json_path))

        print(f"找到 {len(self.samples)} 个样本")

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, json_path = self.samples[idx]

        # 加载图片
        image = Image.open(img_path).convert('RGB')
        original_width, original_height = image.size

        # 加载标注
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        corners = data['Corners']
        corners_array = np.array([[c['X'], c['Y']] for c in corners], dtype=np.float32)

        # 归一化坐标
        corners_norm = corners_array.copy()
        corners_norm[:, 0] /= original_width
        corners_norm[:, 1] /= original_height

        # 缩放图片
        image = image.resize((self.input_size, self.input_size), Image.BILINEAR)

        # 转换为 Tensor
        image_tensor = self.to_tensor(image)
        corners_tensor = torch.from_numpy(corners_norm).flatten().float()

        return image_tensor, corners_tensor, str(img_path.name)


def visualize_prediction(image_tensor, pred_coords, target_coords, output_path, epoch, pixel_error):
    """可视化预测结果"""
    # 转换为 PIL Image
    image = transforms.ToPILImage()(image_tensor)
    draw = ImageDraw.Draw(image)

    size = 512

    # 反归一化到像素坐标
    pred_points = []
    target_points = []
    for i in range(0, 8, 2):
        pred_points.append((pred_coords[i] * size, pred_coords[i + 1] * size))
        target_points.append((target_coords[i] * size, target_coords[i + 1] * size))

    # 画真实标注（绿色）
    for i in range(4):
        draw.line([target_points[i], target_points[(i + 1) % 4]], fill='lime', width=4)

    # 画预测（红色）
    for i in range(4):
        draw.line([pred_points[i], pred_points[(i + 1) % 4]], fill='red', width=4)

    # 画角点
    for pt in target_points:
        x, y = pt
        draw.ellipse([x - 10, y - 10, x + 10, y + 10], fill='lime', outline='white', width=2)

    for pt in pred_points:
        x, y = pt
        draw.ellipse([x - 10, y - 10, x + 10, y + 10], fill='red', outline='white', width=2)

    # 添加文字说明
    draw.text((10, 10), f"Epoch {epoch}", fill='white')
    draw.text((10, 30), f"Pixel Error: {pixel_error:.2f}px", fill='yellow')
    draw.text((10, 50), "绿色=真实  红色=预测", fill='white')

    image.save(output_path, quality=95)


def train_and_visualize(model, dataloader, criterion, optimizer, device, epoch, output_dir):
    """训练一个epoch并可视化每个样本"""
    model.train()
    total_loss = 0.0
    pixel_errors = []

    for images, targets, filenames in dataloader:
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        # 计算像素误差
        pred_coords_np = pred_coords.detach().cpu().numpy()
        targets_np = targets.cpu().numpy()

        pred_pixels = pred_coords_np * 512
        target_pixels = targets_np * 512

        for i in range(len(images)):
            pred_pts = pred_pixels[i].reshape(4, 2)
            target_pts = target_pixels[i].reshape(4, 2)
            errors = np.linalg.norm(pred_pts - target_pts, axis=1)
            avg_error = np.mean(errors)
            pixel_errors.append(avg_error)

            # 可视化（每10个epoch或最后一个epoch）
            if epoch % 10 == 0 or epoch == 1:
                output_path = output_dir / f"epoch_{epoch:04d}_{filenames[i]}"
                visualize_prediction(
                    images[i].cpu(),
                    pred_coords_np[i],
                    targets_np[i],
                    str(output_path),
                    epoch,
                    avg_error
                )

    avg_loss = total_loss / len(dataloader)
    avg_pixel_error = np.mean(pixel_errors)

    return avg_loss, avg_pixel_error


def main():
    import argparse

    parser = argparse.ArgumentParser(description='过拟合测试')
    parser.add_argument('--data-dir', type=str, default='../AnnotationTool/data/test',
                        help='测试数据目录')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='训练轮数（默认1000）')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--device', type=str, default='cpu',
                        help='设备')
    parser.add_argument('--output-dir', type=str, default='overfit_results',
                        help='可视化结果输出目录')

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("="*60)
    print("  过拟合测试 - Overfit Test")
    print("="*60)
    print(f"数据目录: {args.data_dir}")
    print(f"训练轮数: {args.epochs}")
    print(f"学习率: {args.lr}")
    print(f"输出目录: {args.output_dir}")
    print("="*60)

    # 加载数据
    print("\n[1/4] 加载数据...")
    dataset = SmallDataset(args.data_dir, input_size=512)

    if len(dataset) == 0:
        print("[ERROR] 没有找到数据！")
        return

    print(f"样本数量: {len(dataset)}")

    # 创建 DataLoader（不打乱，每次都是同样的顺序）
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    # 创建模型
    print("\n[2/4] 创建模型...")
    device = torch.device(args.device)
    model = PPTCornerDetector(pretrained=True).to(device)

    # 损失函数和优化器
    print("\n[3/4] 设置训练...")
    criterion = CornerDetectionLoss(coord_weight=1.0, order_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 开始训练
    print(f"\n[4/4] 开始过拟合测试（{args.epochs} epochs）...")
    print("-"*60)

    best_error = float('inf')
    patience = 0
    max_patience = 100  # 如果100轮没改进就停止

    for epoch in range(1, args.epochs + 1):
        loss, pixel_error = train_and_visualize(
            model, dataloader, criterion, optimizer, device, epoch, output_dir
        )

        # 打印进度
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{args.epochs} | Loss: {loss:.6f} | Pixel Error: {pixel_error:.3f}px")

        # 检查是否改进
        if pixel_error < best_error:
            best_error = pixel_error
            patience = 0

            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'pixel_error': pixel_error,
            }, output_dir / 'best_overfit_model.pth')
        else:
            patience += 1

        # 早停（如果误差已经很小）
        if pixel_error < 2.0:
            print(f"\n✅ 成功！在第 {epoch} 轮达到 {pixel_error:.2f}px 误差")
            print("👉 结论: 模型架构和损失函数没问题，问题在于数据量/分布")
            break

        # 早停（如果长时间不改进）
        if patience >= max_patience:
            print(f"\n⚠️  警告：{max_patience} 轮没有改进，停止训练")
            print(f"最佳误差: {best_error:.2f}px")
            if best_error > 20:
                print("👉 结论: 连少量样本都学不会，高度怀疑有bug:")
                print("   - 坐标归一化/反归一化问题")
                print("   - 角点顺序不一致")
                print("   - 损失函数权重问题")
                print("   - 模型输出维度错误")
            break

    print("\n"+"="*60)
    print("过拟合测试完成!")
    print(f"最佳像素误差: {best_error:.2f}px")
    print(f"可视化结果保存在: {output_dir}")
    print("="*60)

    # 最终可视化
    print("\n生成最终可视化...")
    model.eval()
    with torch.no_grad():
        for images, targets, filenames in dataloader:
            images = images.to(device)
            pred_coords, _ = model(images)

            pred_coords_np = pred_coords.cpu().numpy()
            targets_np = targets.cpu().numpy()

            pred_pixels = pred_coords_np * 512
            target_pixels = targets_np * 512

            for i in range(len(images)):
                pred_pts = pred_pixels[i].reshape(4, 2)
                target_pts = target_pixels[i].reshape(4, 2)
                errors = np.linalg.norm(pred_pts - target_pts, axis=1)
                avg_error = np.mean(errors)

                output_path = output_dir / f"FINAL_{filenames[i]}"
                visualize_prediction(
                    images[i].cpu(),
                    pred_coords_np[i],
                    targets_np[i],
                    str(output_path),
                    epoch,
                    avg_error
                )

                print(f"  {filenames[i]}: {avg_error:.2f}px")


if __name__ == '__main__':
    main()
