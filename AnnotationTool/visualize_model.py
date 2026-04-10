"""
模型可视化验证脚本

功能：
- 加载训练好的模型
- 在验证集上进行推理
- 可视化预测结果 vs 真实标注
- 计算详细的误差指标
"""
import os
import json
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
from pathlib import Path
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


# ==================== 模型定义 ====================

class MobileNetV3CornerNet(nn.Module):
    """MobileNetV3 角点回归网络"""

    def __init__(self, num_corners=4, pretrained=False):
        super(MobileNetV3CornerNet, self).__init__()

        # 加载 MobileNetV3-Small
        from torchvision import models
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
            nn.Linear(128, num_corners * 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        corners = self.regressor(x)
        corners = torch.sigmoid(corners)
        return corners


# ==================== 数据集定义 ====================

class SimpleDataset:
    """简单数据集（用于推理）"""

    def __init__(self, images_dir, labels_dir, input_size=224):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.input_size = input_size

        self.image_files = sorted(list(self.images_dir.glob('*.jpg')))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]

        # 读取原始图片（用于可视化）
        image_orig = cv2.imread(str(img_path))
        image_orig = cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB)
        h, w = image_orig.shape[:2]

        # 读取标签
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        with open(label_path, 'r') as f:
            label_str = f.read().strip()

        coords = label_str.split(',')
        corners_gt = []
        for i in range(0, len(coords), 2):
            corners_gt.append([float(coords[i]), float(coords[i + 1])])

        return {
            'image': image_orig,
            'corners_gt': np.array(corners_gt),
            'path': str(img_path)
        }


# ==================== 验证函数 ====================

def draw_corners(image, corners_gt, corners_pred, errors=None):
    """在图片上绘制角点"""
    h, w = image.shape[:2]

    # 创建绘图
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)

    # 绘制真实边框（绿色）
    corners_gt_np = np.array(corners_gt)
    for i in range(4):
        p1 = corners_gt_np[i]
        p2 = corners_gt_np[(i + 1) % 4]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'g-', linewidth=2, label='Ground Truth' if i == 0 else '')

    # 绘制真实角点（绿色圆圈）
    for i, pt in enumerate(corners_gt_np):
        ax.add_patch(plt.Circle((pt[0], pt[1]), 10, color='green', fill=False, linewidth=2))
        ax.text(pt[0] + 15, pt[1], f'GT-{i}', color='green', fontsize=12, fontweight='bold')

    # 绘制预测边框（红色）
    corners_pred_np = np.array(corners_pred)
    for i in range(4):
        p1 = corners_pred_np[i]
        p2 = corners_pred_np[(i + 1) % 4]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r--', linewidth=2, label='Prediction' if i == 0 else '')

    # 绘制预测角点（红色圆圈）
    for i, pt in enumerate(corners_pred_np):
        ax.add_patch(plt.Circle((pt[0], pt[1]), 10, color='red', fill=False, linewidth=2))
        if errors is not None:
            ax.text(pt[0] + 15, pt[1] + 15, f'Pred-{i}\n{errors[i]:.1f}px',
                   color='red', fontsize=10, fontweight='bold')
        else:
            ax.text(pt[0] + 15, pt[1] + 15, f'Pred-{i}',
                   color='red', fontsize=12, fontweight='bold')

    ax.legend(loc='upper right', fontsize=12)
    ax.axis('off')
    ax.set_title('Screen Border Detection', fontsize=14, fontweight='bold')

    # 转换为 OpenCV 格式
    canvas = FigureCanvas(fig)
    canvas.draw()
    img_vis = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
    img_vis = img_vis[:, :, :3]  # 移除 alpha 通道
    plt.close(fig)

    return cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)


def visualize_model(model_path, val_images_dir, val_labels_dir, output_dir, num_samples=20):
    """可视化模型预测结果"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载模型
    print(f"加载模型: {model_path}")
    model = MobileNetV3CornerNet(num_corners=4, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    model = model.to(device)
    model.eval()

    # 数据变换
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集
    dataset = SimpleDataset(val_images_dir, val_labels_dir)
    print(f"验证集样本数: {len(dataset)}")

    # 随机选择样本
    if num_samples > len(dataset):
        num_samples = len(dataset)
    indices = random.sample(range(len(dataset)), num_samples)

    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 统计信息
    all_errors = []
    corner_errors = [[], [], [], []]

    print("\n开始可视化...")
    for idx in tqdm(indices):
        sample = dataset[idx]
        image_orig = sample['image']
        corners_gt = sample['corners_gt']
        h, w = image_orig.shape[:2]

        # 准备输入
        image_tensor = transform(image_orig).unsqueeze(0).to(device)

        # 推理
        with torch.no_grad():
            corners_pred_norm = model(image_tensor)[0].cpu().numpy()

        # 反归一化
        corners_pred = []
        for i in range(0, len(corners_pred_norm), 2):
            x = corners_pred_norm[i] * w
            y = corners_pred_norm[i + 1] * h
            corners_pred.append([x, y])
        corners_pred = np.array(corners_pred)

        # 计算每个角点的误差
        errors = []
        for i in range(4):
            error = np.linalg.norm(corners_gt[i] - corners_pred[i])
            errors.append(error)
            corner_errors[i].append(error)

        avg_error = np.mean(errors)
        all_errors.append(avg_error)

        # 可视化
        vis_image = draw_corners(image_orig, corners_gt, corners_pred, errors)

        # 保存
        output_path = output_dir / f"vis_{Path(sample['path']).stem}.jpg"
        cv2.imwrite(str(output_path), vis_image)

    # 打印统计信息
    print("\n" + "="*60)
    print("统计信息:")
    print("="*60)
    print(f"平均角点误差: {np.mean(all_errors):.2f} px")
    print(f"中位数误差: {np.median(all_errors):.2f} px")
    print(f"最大误差: {np.max(all_errors):.2f} px")
    print(f"最小误差: {np.min(all_errors):.2f} px")
    print(f"标准差: {np.std(all_errors):.2f} px")

    print("\n各角点误差:")
    corner_names = ['左上', '右上', '右下', '左下']
    for i in range(4):
        print(f"  {corner_names[i]}: {np.mean(corner_errors[i]):.2f} px (±{np.std(corner_errors[i]):.2f})")

    print("\n误差分布:")
    bins = [0, 5, 10, 20, 50, 100, float('inf')]
    labels = ['0-5px', '5-10px', '10-20px', '20-50px', '50-100px', '>100px']
    for i in range(len(bins) - 1):
        count = sum(1 for e in all_errors if bins[i] <= e < bins[i+1])
        percentage = count / len(all_errors) * 100
        print(f"  {labels[i]}: {count} ({percentage:.1f}%)")

    print(f"\n可视化结果已保存到: {output_dir}")
    print("="*60)


def evaluate_on_dataset(model_path, images_dir, labels_dir):
    """在完整数据集上评估模型"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载模型
    print(f"加载模型: {model_path}")
    model = MobileNetV3CornerNet(num_corners=4, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    model = model.to(device)
    model.eval()

    # 数据变换
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集
    dataset = SimpleDataset(images_dir, labels_dir)
    print(f"数据集样本数: {len(dataset)}")

    # 统计信息
    all_errors = []
    corner_errors = [[], [], [], []]

    print("\n开始评估...")
    for idx in tqdm(range(len(dataset))):
        sample = dataset[idx]
        image_orig = sample['image']
        corners_gt = sample['corners_gt']
        h, w = image_orig.shape[:2]

        # 准备输入
        image_tensor = transform(image_orig).unsqueeze(0).to(device)

        # 推理
        with torch.no_grad():
            corners_pred_norm = model(image_tensor)[0].cpu().numpy()

        # 反归一化
        corners_pred = []
        for i in range(0, len(corners_pred_norm), 2):
            x = corners_pred_norm[i] * w
            y = corners_pred_norm[i + 1] * h
            corners_pred.append([x, y])
        corners_pred = np.array(corners_pred)

        # 计算误差
        for i in range(4):
            error = np.linalg.norm(corners_gt[i] - corners_pred[i])
            corner_errors[i].append(error)

        avg_error = np.mean(corner_errors[i] for i in range(4))
        all_errors.append(avg_error)

    # 打印统计信息
    print("\n" + "="*60)
    print("完整数据集评估结果:")
    print("="*60)
    print(f"样本数量: {len(all_errors)}")
    print(f"平均角点误差: {np.mean(all_errors):.2f} px")
    print(f"中位数误差: {np.median(all_errors):.2f} px")
    print(f"最大误差: {np.max(all_errors):.2f} px")
    print(f"最小误差: {np.min(all_errors):.2f} px")
    print(f"标准差: {np.std(all_errors):.2f} px")

    print("\n各角点误差:")
    corner_names = ['左上', '右上', '右下', '左下']
    for i in range(4):
        print(f"  {corner_names[i]}: {np.mean(corner_errors[i]):.2f} px (±{np.std(corner_errors[i]):.2f})")

    # 成功率统计
    success_5px = sum(1 for e in all_errors if e < 5) / len(all_errors) * 100
    success_10px = sum(1 for e in all_errors if e < 10) / len(all_errors) * 100
    success_20px = sum(1 for e in all_errors if e < 20) / len(all_errors) * 100

    print(f"\n成功率:")
    print(f"  < 5px: {success_5px:.1f}%")
    print(f"  < 10px: {success_10px:.1f}%")
    print(f"  < 20px: {success_20px:.1f}%")

    print("="*60)


if __name__ == '__main__':
    import sys

    # 配置
    model_path = r'D:\Programing\C#\MauiScan\AnnotationTool\best_model.pth'
    val_images_dir = r'D:\Programing\C#\MauiScan\AnnotationTool\training_data\images\val'
    val_labels_dir = r'D:\Programing\C#\MauiScan\AnnotationTool\training_data\labels\val'
    output_dir = r'D:\Programing\C#\MauiScan\AnnotationTool\visualization'

    # 随机种子
    random.seed(42)
    np.random.seed(42)

    if len(sys.argv) > 1 and sys.argv[1] == '--evaluate':
        # 完整评估模式
        evaluate_on_dataset(model_path, val_images_dir, val_labels_dir)
    else:
        # 可视化模式（默认）
        visualize_model(model_path, val_images_dir, val_labels_dir, output_dir, num_samples=20)
