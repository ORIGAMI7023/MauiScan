#!/usr/bin/env python3
"""
CV精修批量可视化诊断工具
自动加载ONNX模型 → ML推理 → CV精修 → 生成可视化结果
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path
from typing import Tuple, List, Optional
import argparse
import onnxruntime as ort


class MLCornerDetector:
    """ONNX ML模型推理"""

    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(model_path)
        self.input_size = (512, 512)

    def detect(self, image: np.ndarray) -> dict:
        """
        检测角点
        返回: {TopLeftX, TopLeftY, TopRightX, TopRightY, ...}
        """
        h, w = image.shape[:2]

        # 1. Resize到512x512
        resized = cv2.resize(image, self.input_size, interpolation=cv2.INTER_LINEAR)

        # 2. 转换为RGB并归一化 [0, 1]
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0

        # 3. 转换为CHW格式 (1, 3, 512, 512)
        chw = np.transpose(normalized, (2, 0, 1))
        input_tensor = np.expand_dims(chw, axis=0)

        # 4. 推理
        outputs = self.session.run(None, {'input': input_tensor})
        coordinates = outputs[0][0]  # [8] - 归一化坐标

        # 5. 反归一化到原图尺寸
        corners = {
            'TopLeftX': float(coordinates[0] * w),
            'TopLeftY': float(coordinates[1] * h),
            'TopRightX': float(coordinates[2] * w),
            'TopRightY': float(coordinates[3] * h),
            'BottomRightX': float(coordinates[4] * w),
            'BottomRightY': float(coordinates[5] * h),
            'BottomLeftX': float(coordinates[6] * w),
            'BottomLeftY': float(coordinates[7] * h),
        }

        return corners


def refine_corner_visualize(
    image: np.ndarray,
    ml_x: float,
    ml_y: float,
    corner_name: str,
    output_dir: Path
) -> Tuple[Optional[Tuple[float, float]], int, dict]:
    """
    复刻C++的scanner_refine_corner逻辑，并生成可视化图片
    """
    debug_info = {}

    # 1. 转灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    h, w = gray.shape
    debug_info['image_size'] = (w, h)

    # 2. 动态计算patch大小
    image_min_dim = min(w, h)
    patch_size = min(256, image_min_dim // 12)
    patch_size = max(64, patch_size)
    half_patch = patch_size // 2

    debug_info['patch_size'] = patch_size

    center_x = int(ml_x)
    center_y = int(ml_y)

    # 3. 裁剪patch（防止越界）
    x1 = max(0, center_x - half_patch)
    y1 = max(0, center_y - half_patch)
    x2 = min(w, center_x + half_patch)
    y2 = min(h, center_y + half_patch)

    debug_info['patch_roi'] = (x1, y1, x2, y2)

    if x2 - x1 < 20 or y2 - y1 < 20:
        print(f"  [{corner_name}] Patch too small: {x2-x1}x{y2-y1}")
        return None, 0, debug_info

    patch = gray[y1:y2, x1:x2]
    patch_color = image[y1:y2, x1:x2] if len(image.shape) == 3 else cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)

    # 4. Canny边缘检测
    edges = cv2.Canny(patch, 30, 100, apertureSize=3)

    # 5. Hough直线检测
    min_line_length = max(10, patch_size // 8)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 15, minLineLength=min_line_length, maxLineGap=10)

    debug_info['hough_params'] = {
        'threshold': 15,
        'min_line_length': min_line_length,
        'max_gap': 10
    }

    if lines is None or len(lines) < 2:
        print(f"  [{corner_name}] Too few lines: {0 if lines is None else len(lines)}")
        debug_info['num_lines'] = 0 if lines is None else len(lines)
        vis = create_visualization(patch_color, edges, None, None, None, None, corner_name, debug_info)
        cv2.imwrite(str(output_dir / f"{corner_name}_fail.jpg"), vis)
        return None, 0, debug_info

    # 6. 直线分类（水平 vs 垂直）
    horizontal_lines = []
    vertical_lines = []

    for line in lines:
        x1_l, y1_l, x2_l, y2_l = line[0]
        dx = abs(x2_l - x1_l)
        dy = abs(y2_l - y1_l)

        if dx > dy:
            horizontal_lines.append(line[0])
        else:
            vertical_lines.append(line[0])

    debug_info['num_h_lines'] = len(horizontal_lines)
    debug_info['num_v_lines'] = len(vertical_lines)

    print(f"  [{corner_name}] Lines: H={len(horizontal_lines)}, V={len(vertical_lines)}")

    if len(horizontal_lines) == 0 or len(vertical_lines) == 0:
        print(f"  [{corner_name}] Missing line groups")
        vis = create_visualization(patch_color, edges, lines, horizontal_lines, vertical_lines, None, corner_name, debug_info)
        cv2.imwrite(str(output_dir / f"{corner_name}_missing_group.jpg"), vis)
        return None, 0, debug_info

    # 7. 拟合直线（最小二乘法）
    def fit_line(line_group):
        points = []
        for x1_l, y1_l, x2_l, y2_l in line_group:
            points.append([x1_l, y1_l])
            points.append([x2_l, y2_l])
        points = np.array(points, dtype=np.float32)

        avg_x = np.mean(points[:, 0])
        avg_y = np.mean(points[:, 1])

        numerator = np.sum((points[:, 0] - avg_x) * (points[:, 1] - avg_y))
        denominator = np.sum((points[:, 0] - avg_x) ** 2)

        if abs(denominator) < 1e-6:
            return None, None

        k = numerator / denominator
        b = avg_y - k * avg_x
        return k, b

    k1, b1 = fit_line(horizontal_lines)
    k2, b2 = fit_line(vertical_lines)

    if k1 is not None and k2 is not None:
        debug_info['fitted_lines'] = {
            'h_line': {'k': float(k1), 'b': float(b1)},
            'v_line': {'k': float(k2), 'b': float(b2)}
        }
    else:
        debug_info['fitted_lines'] = {
            'h_line': {'k': None, 'b': None},
            'v_line': {'k': None, 'b': None}
        }

    if k1 is None or k2 is None:
        print(f"  [{corner_name}] Failed to fit lines")
        vis = create_visualization(patch_color, edges, lines, horizontal_lines, vertical_lines, None, corner_name, debug_info)
        cv2.imwrite(str(output_dir / f"{corner_name}_fit_fail.jpg"), vis)
        return None, 0, debug_info

    # 8. 计算交点
    if abs(k1 - k2) < 1e-6:
        print(f"  [{corner_name}] Lines are parallel: k1={k1:.3f}, k2={k2:.3f}")
        vis = create_visualization(patch_color, edges, lines, horizontal_lines, vertical_lines, None, corner_name, debug_info)
        cv2.imwrite(str(output_dir / f"{corner_name}_parallel.jpg"), vis)
        return None, 0, debug_info

    x_intersect = (b2 - b1) / (k1 - k2)
    y_intersect = k1 * x_intersect + b1

    debug_info['intersection_in_patch'] = (float(x_intersect), float(y_intersect))

    # 9. 转换回原图坐标
    refined_x = x1 + x_intersect
    refined_y = y1 + y_intersect

    # 10. 验证合理性
    distance = np.sqrt((refined_x - ml_x) ** 2 + (refined_y - ml_y) ** 2)
    max_allowed_distance = patch_size * 0.67

    debug_info['distance_from_ml'] = float(distance)
    debug_info['max_allowed_distance'] = float(max_allowed_distance)

    print(f"  [{corner_name}] Refined: ({refined_x:.1f}, {refined_y:.1f}), distance={distance:.1f}px")

    if distance > max_allowed_distance:
        print(f"  [{corner_name}] Distance too large: {distance:.1f} > {max_allowed_distance:.1f}")
        vis = create_visualization(patch_color, edges, lines, horizontal_lines, vertical_lines, (x_intersect, y_intersect), corner_name, debug_info, distance_too_large=True)
        cv2.imwrite(str(output_dir / f"{corner_name}_too_far.jpg"), vis)
        return None, 0, debug_info

    # 11. 确定置信度
    if len(horizontal_lines) >= 2 and len(vertical_lines) >= 2:
        confidence = 2  # 高置信度
    elif len(horizontal_lines) >= 1 and len(vertical_lines) >= 1:
        confidence = 1  # 低置信度
    else:
        confidence = 0

    debug_info['confidence'] = confidence

    # 可视化：成功
    vis = create_visualization(patch_color, edges, lines, horizontal_lines, vertical_lines, (x_intersect, y_intersect), corner_name, debug_info)
    cv2.imwrite(str(output_dir / f"{corner_name}_success_conf{confidence}.jpg"), vis)

    return (refined_x, refined_y), confidence, debug_info


def create_visualization(
    patch_color: np.ndarray,
    edges: np.ndarray,
    all_lines: Optional[np.ndarray],
    h_lines: Optional[List],
    v_lines: Optional[List],
    intersection: Optional[Tuple[float, float]],
    corner_name: str,
    debug_info: dict,
    distance_too_large: bool = False
) -> np.ndarray:
    """创建4宫格可视化图片"""
    # 1. 原始patch
    img1 = patch_color.copy()
    cv2.circle(img1, (img1.shape[1]//2, img1.shape[0]//2), 5, (0, 0, 255), -1)

    # 2. Canny边缘
    img2 = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # 3. Hough检测的线
    img3 = patch_color.copy()
    if h_lines is not None:
        for x1, y1, x2, y2 in h_lines:
            cv2.line(img3, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if v_lines is not None:
        for x1, y1, x2, y2 in v_lines:
            cv2.line(img3, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # 4. 拟合线+交点
    img4 = patch_color.copy()
    h, w = img4.shape[:2]

    if 'fitted_lines' in debug_info:
        h_line = debug_info['fitted_lines']['h_line']
        v_line = debug_info['fitted_lines']['v_line']

        if h_line['k'] is not None:
            k1, b1 = h_line['k'], h_line['b']
            x1_fit, x2_fit = 0, w-1
            y1_fit, y2_fit = int(k1 * x1_fit + b1), int(k1 * x2_fit + b1)
            cv2.line(img4, (x1_fit, y1_fit), (x2_fit, y2_fit), (0, 255, 255), 2)

        if v_line['k'] is not None:
            k2, b2 = v_line['k'], v_line['b']
            x1_fit, x2_fit = 0, w-1
            y1_fit, y2_fit = int(k2 * x1_fit + b2), int(k2 * x2_fit + b2)
            cv2.line(img4, (x1_fit, y1_fit), (x2_fit, y2_fit), (255, 255, 0), 2)

    if intersection is not None:
        x_int, y_int = intersection
        color = (0, 0, 255) if distance_too_large else (0, 255, 0)
        cv2.circle(img4, (int(x_int), int(y_int)), 5, color, -1)
        cv2.circle(img4, (w//2, h//2), 3, (255, 0, 255), -1)

    # 拼接4宫格
    top_row = np.hstack([img1, img2])
    bottom_row = np.hstack([img3, img4])
    result = np.vstack([top_row, bottom_row])

    # 添加文字
    info_text = [
        f"{corner_name}",
        f"Patch: {debug_info.get('patch_size', 0)}px",
        f"H: {debug_info.get('num_h_lines', 0)}  V: {debug_info.get('num_v_lines', 0)}",
        f"Conf: {debug_info.get('confidence', 0)}",
        f"Dist: {debug_info.get('distance_from_ml', 0):.1f}px"
    ]

    y_offset = 30
    for i, text in enumerate(info_text):
        cv2.putText(result, text, (10, y_offset + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(result, text, (10, y_offset + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    return result


def process_image(
    image_path: Path,
    ml_detector: MLCornerDetector,
    output_dir: Path
) -> dict:
    """处理单张图片"""
    print(f"\n{'='*60}")
    print(f"处理: {image_path.name}")
    print(f"{'='*60}")

    # 读取图片
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"  ❌ 无法读取图片")
        return None

    h, w = image.shape[:2]
    print(f"图片尺寸: {w}x{h}")

    # ML推理
    print("运行ML推理...")
    ml_corners = ml_detector.detect(image)
    print(f"ML角点: TL({ml_corners['TopLeftX']:.1f},{ml_corners['TopLeftY']:.1f}) "
          f"TR({ml_corners['TopRightX']:.1f},{ml_corners['TopRightY']:.1f}) "
          f"BR({ml_corners['BottomRightX']:.1f},{ml_corners['BottomRightY']:.1f}) "
          f"BL({ml_corners['BottomLeftX']:.1f},{ml_corners['BottomLeftY']:.1f})")

    # 创建输出目录
    img_output_dir = output_dir / image_path.stem
    img_output_dir.mkdir(parents=True, exist_ok=True)

    # CV精修4个角点
    corners = ['TL', 'TR', 'BR', 'BL']
    corner_keys = ['TopLeft', 'TopRight', 'BottomRight', 'BottomLeft']

    results = {
        'image': str(image_path),
        'size': (w, h),
        'ml_corners': ml_corners,
        'refined_corners': {},
        'debug_info': {}
    }

    print("\nCV精修:")
    for corner, key in zip(corners, corner_keys):
        ml_x = ml_corners[f'{key}X']
        ml_y = ml_corners[f'{key}Y']

        refined, confidence, debug_info = refine_corner_visualize(
            image, ml_x, ml_y, corner, img_output_dir
        )

        results['refined_corners'][corner] = {
            'ml': (ml_x, ml_y),
            'refined': refined,
            'confidence': confidence
        }
        results['debug_info'][corner] = debug_info

    # 绘制最终对比图
    vis_final = image.copy()

    for corner, key in zip(corners, corner_keys):
        ml_x = int(ml_corners[f'{key}X'])
        ml_y = int(ml_corners[f'{key}Y'])

        # ML点（红色）
        cv2.circle(vis_final, (ml_x, ml_y), 10, (0, 0, 255), 3)
        cv2.putText(vis_final, f"{corner}_ML", (ml_x+15, ml_y-15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 精修点（绿色）
        refined = results['refined_corners'][corner]['refined']
        if refined is not None:
            rx, ry = int(refined[0]), int(refined[1])
            cv2.circle(vis_final, (rx, ry), 10, (0, 255, 0), 3)
            cv2.line(vis_final, (ml_x, ml_y), (rx, ry), (255, 255, 0), 2)

            distance = np.sqrt((rx - ml_x)**2 + (ry - ml_y)**2)
            cv2.putText(vis_final, f"{corner}_CV({distance:.0f}px)", (rx+15, ry+15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imwrite(str(img_output_dir / "final_comparison.jpg"), vis_final)

    # 保存JSON（转换numpy类型）
    def convert_to_json_serializable(obj):
        """递归转换numpy类型为Python原生类型"""
        if isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_to_json_serializable(item) for item in obj)
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    results_serializable = convert_to_json_serializable(results)

    with open(img_output_dir / "results.json", 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 完成，结果: {img_output_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(description='CV精修批量可视化诊断工具')
    parser.add_argument('--images', type=str, required=True, help='测试图片文件夹路径')
    parser.add_argument('--model', type=str, required=True, help='ONNX模型路径')
    parser.add_argument('--output', type=str, default='./cv_debug_output', help='输出目录')

    args = parser.parse_args()

    images_dir = Path(args.images)
    model_path = Path(args.model)
    output_dir = Path(args.output)

    # 检查路径
    if not images_dir.exists():
        print(f"❌ 图片目录不存在: {images_dir}")
        return

    if not model_path.exists():
        print(f"❌ 模型文件不存在: {model_path}")
        return

    # 加载ML模型
    print(f"加载ONNX模型: {model_path}")
    ml_detector = MLCornerDetector(str(model_path))
    print("✅ 模型加载成功\n")

    # 扫描图片
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpeg"))
    print(f"找到 {len(image_files)} 张图片\n")

    if len(image_files) == 0:
        print(f"❌ 目录中没有图片文件")
        return

    # 批量处理
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for image_path in image_files:
        result = process_image(image_path, ml_detector, output_dir)
        if result:
            all_results.append(result)

    # 生成汇总报告
    print(f"\n{'='*60}")
    print("汇总报告")
    print(f"{'='*60}")

    for result in all_results:
        image_name = Path(result['image']).name
        print(f"\n{image_name}:")
        for corner in ['TL', 'TR', 'BR', 'BL']:
            refined = result['refined_corners'][corner]['refined']
            conf = result['refined_corners'][corner]['confidence']
            status = f"✅ Conf={conf}" if refined else "❌ Failed"
            print(f"  {corner}: {status}")

    print(f"\n{'='*60}")
    print(f"✅ 所有结果保存在: {output_dir.absolute()}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
