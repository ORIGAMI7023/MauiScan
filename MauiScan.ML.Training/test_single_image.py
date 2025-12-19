"""
测试单张图片的推理效果
"""

import cv2
import numpy as np
import onnxruntime as ort
import json
from pathlib import Path
import sys

def preprocess_opencv(image_path, target_size=512):
    """使用 OpenCV 预处理（和训练时一样）"""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"无法加载图片: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = (image.shape[1], image.shape[0])  # (width, height)

    # OpenCV resize
    image_resized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    # 转换为 CHW float tensor
    img_array = image_resized.astype(np.float32) / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)

    return img_array, original_size

def main():
    # 文件路径
    image_path = Path(r"D:\Programing\C#\MauiScan\test.jpg")
    json_path = Path(r"D:\Programing\C#\MauiScan\test.json")
    model_path = Path(__file__).parent / "ppt_corner_detector_opencv.onnx"

    if not image_path.exists():
        print(f"错误：找不到图片 {image_path}")
        return

    if not json_path.exists():
        print(f"错误：找不到标注 {json_path}")
        return

    if not model_path.exists():
        print(f"错误：找不到模型 {model_path}")
        return

    print(f"图片: {image_path.name}")
    print(f"标注: {json_path.name}")
    print(f"模型: {model_path.name}")
    print("=" * 60)

    # 加载真实标注
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    gt_corners = data['Corners']
    gt_array = np.array([[c['X'], c['Y']] for c in gt_corners], dtype=np.float32)

    # 预处理图片
    img_tensor, (width, height) = preprocess_opencv(image_path)
    print(f"\n原图尺寸: {width}x{height}")

    # 打印像素值（用于和 Android 对比）
    print(f"\n像素值统计:")
    print(f"  min: {img_tensor.min():.3f}, max: {img_tensor.max():.3f}, mean: {img_tensor.mean():.3f}")
    print(f"  First 10 R: {', '.join([f'{v:.3f}' for v in img_tensor[0, 0, 0, :10]])}")
    print(f"  First 10 G: {', '.join([f'{v:.3f}' for v in img_tensor[0, 1, 0, :10]])}")
    print(f"  First 10 B: {', '.join([f'{v:.3f}' for v in img_tensor[0, 2, 0, :10]])}")

    # 归一化真实标注
    gt_normalized = gt_array.copy()
    gt_normalized[:, 0] /= width
    gt_normalized[:, 1] /= height
    gt_normalized_flat = gt_normalized.flatten()

    print(f"真实坐标 (归一化):")
    print(f"  TL: [{gt_normalized[0, 0]:.3f}, {gt_normalized[0, 1]:.3f}]")
    print(f"  TR: [{gt_normalized[1, 0]:.3f}, {gt_normalized[1, 1]:.3f}]")
    print(f"  BR: [{gt_normalized[2, 0]:.3f}, {gt_normalized[2, 1]:.3f}]")
    print(f"  BL: [{gt_normalized[3, 0]:.3f}, {gt_normalized[3, 1]:.3f}]")

    # 加载模型并推理
    session = ort.InferenceSession(str(model_path))
    outputs = session.run(None, {'input': img_tensor})
    pred_coords = outputs[0][0]  # [8]

    print(f"\n预测坐标 (归一化):")
    print(f"  TL: [{pred_coords[0]:.3f}, {pred_coords[1]:.3f}]")
    print(f"  TR: [{pred_coords[2]:.3f}, {pred_coords[3]:.3f}]")
    print(f"  BR: [{pred_coords[4]:.3f}, {pred_coords[5]:.3f}]")
    print(f"  BL: [{pred_coords[6]:.3f}, {pred_coords[7]:.3f}]")

    # 计算 MSE Loss（和训练时一样）
    mse_loss = np.mean((pred_coords - gt_normalized_flat) ** 2)

    print(f"\n{'=' * 60}")
    print(f"MSE Loss: {mse_loss:.6f}")
    print(f"{'=' * 60}")

    # 计算像素误差
    pred_pixels = pred_coords.reshape(4, 2) * np.array([width, height])
    errors = np.linalg.norm(pred_pixels - gt_array, axis=1)

    print(f"\n像素误差:")
    print(f"  TL: {errors[0]:.1f}px")
    print(f"  TR: {errors[1]:.1f}px")
    print(f"  BR: {errors[2]:.1f}px")
    print(f"  BL: {errors[3]:.1f}px")
    print(f"  平均: {errors.mean():.1f}px")

    # 映射到 512x512 的误差（用于和训练验证误差对比）
    pred_512 = pred_coords.reshape(4, 2) * 512
    gt_512 = gt_normalized.reshape(4, 2) * 512
    errors_512 = np.linalg.norm(pred_512 - gt_512, axis=1)
    print(f"\n512x512 上的误差 (对比训练验证误差):")
    print(f"  平均: {errors_512.mean():.1f}px")

    # 可视化结果
    print(f"\n生成可视化图...")
    vis_image = cv2.imread(str(image_path))

    # 画真实标注（绿色）
    gt_points = gt_array.astype(np.int32)
    for i in range(4):
        cv2.circle(vis_image, tuple(gt_points[i]), 15, (0, 255, 0), -1)
        cv2.line(vis_image, tuple(gt_points[i]), tuple(gt_points[(i+1)%4]), (0, 255, 0), 3)

    # 画预测结果（红色）
    pred_points = pred_pixels.astype(np.int32)
    for i in range(4):
        cv2.circle(vis_image, tuple(pred_points[i]), 15, (0, 0, 255), -1)
        cv2.line(vis_image, tuple(pred_points[i]), tuple(pred_points[(i+1)%4]), (0, 0, 255), 3)

    # 保存结果
    output_path = Path(__file__).parent / "test_result_visualization.jpg"
    cv2.imwrite(str(output_path), vis_image)
    print(f"可视化结果保存到: {output_path}")
    print(f"\n绿色 = 真实标注")
    print(f"红色 = 模型预测")

if __name__ == "__main__":
    main()
