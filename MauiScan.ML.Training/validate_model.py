"""
验证模型并可视化预测结果
"""

import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image, ImageDraw
import json

sys.path.append(str(Path(__file__).parent / 'models'))
sys.path.append(str(Path(__file__).parent / 'dataset'))
from corner_detector import PPTCornerDetector
from prepare_data import AnnotationDataset


def visualize_prediction(image_path, pred_corners, gt_corners, output_path):
    """
    可视化预测结果
    蓝色：预测
    绿色：真实标注
    """
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)

    width, height = image.size

    # 反归一化到像素坐标
    pred_pixels = pred_corners.reshape(4, 2)
    pred_pixels[:, 0] *= width
    pred_pixels[:, 1] *= height

    gt_pixels = gt_corners.reshape(4, 2)
    gt_pixels[:, 0] *= width
    gt_pixels[:, 1] *= height

    # 绘制预测（蓝色）
    for i, (x, y) in enumerate(pred_pixels):
        # 大圆标注点
        draw.ellipse([x-15, y-15, x+15, y+15], outline='blue', width=4)
        draw.text((x+20, y-10), f'P{i+1}', fill='blue')

    # 绘制四边形
    pred_polygon = [(x, y) for x, y in pred_pixels]
    draw.polygon(pred_polygon, outline='blue', width=3)

    # 绘制真实标注（绿色）
    for i, (x, y) in enumerate(gt_pixels):
        # 小圆标注点
        draw.ellipse([x-8, y-8, x+8, y+8], outline='green', width=3)
        draw.text((x+20, y+10), f'GT{i+1}', fill='green')

    # 绘制四边形
    gt_polygon = [(x, y) for x, y in gt_pixels]
    draw.polygon(gt_polygon, outline='green', width=2)

    # 计算误差
    errors = np.linalg.norm(pred_pixels - gt_pixels, axis=1)
    avg_error = np.mean(errors)

    # 添加误差文本
    draw.text((10, 10), f'Avg Error: {avg_error:.2f}px', fill='red', font=None)
    for i, err in enumerate(errors):
        draw.text((10, 40 + i*30), f'Corner {i+1}: {err:.2f}px', fill='red')

    image.save(output_path)
    return avg_error


def main():
    import argparse

    parser = argparse.ArgumentParser(description='验证模型')
    parser.add_argument('--model', type=str, default='checkpoints/model_full_fixed.pth',
                        help='模型路径')
    parser.add_argument('--data-root', type=str, default='../AnnotationTool/data',
                        help='数据根目录')
    parser.add_argument('--output-dir', type=str, default='validation',
                        help='输出目录')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='验证样本数')

    args = parser.parse_args()

    print("="*60)
    print("  模型验证与可视化")
    print("="*60)

    # 加载模型
    print(f"\n[1/4] 加载模型: {args.model}")
    device = torch.device('cpu')
    model = PPTCornerDetector(pretrained=False).to(device)

    checkpoint = torch.load(args.model, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
    print(f"  Pixel Error: {checkpoint['val_pixel_error']:.2f}px")

    # 加载数据
    print(f"\n[2/4] 加载数据集")
    annotation_dataset = AnnotationDataset(args.data_root)
    _, val_dataset_raw = annotation_dataset.split_train_val(val_ratio=0.15)

    print(f"  验证集: {len(val_dataset_raw)} 样本")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # 验证
    print(f"\n[3/4] 验证 {min(args.num_samples, len(val_dataset_raw))} 个样本")

    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    errors = []

    for i in range(min(args.num_samples, len(val_dataset_raw))):
        image_array, corners, (width, height) = val_dataset_raw[i]

        # ⭐ 直接从子数据集获取图片路径（修复bug）
        image_path = val_dataset_raw.samples[i]['image_path']

        # 预处理
        image = Image.fromarray(image_array)
        image_tensor = transform(image).unsqueeze(0).to(device)

        # 推理
        with torch.no_grad():
            pred_coords, _ = model(image_tensor)

        pred_coords_np = pred_coords.cpu().numpy()[0]

        # GT归一化
        gt_coords = corners.copy()
        gt_coords[:, 0] /= width
        gt_coords[:, 1] /= height
        gt_coords_flat = gt_coords.flatten()

        # 保存可视化
        output_path = output_dir / f"val_{i+1}.png"
        avg_error = visualize_prediction(image_path, pred_coords_np, gt_coords_flat, output_path)
        errors.append(avg_error)

        print(f"  [{i+1}/{args.num_samples}] {Path(image_path).name}: {avg_error:.2f}px -> {output_path}")

    # 统计
    print(f"\n[4/4] 统计结果")
    print(f"  平均误差: {np.mean(errors):.2f}px")
    print(f"  最小误差: {np.min(errors):.2f}px")
    print(f"  最大误差: {np.max(errors):.2f}px")
    print(f"  标准差: {np.std(errors):.2f}px")

    print(f"\n可视化结果保存到: {output_dir.absolute()}")
    print("="*60)


if __name__ == '__main__':
    main()
