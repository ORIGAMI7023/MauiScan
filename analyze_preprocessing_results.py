#!/usr/bin/env python3
"""
分析预处理方法测试结果
读取所有240组JSON结果，生成详细的统计报告
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import csv
from dataclasses import dataclass
import numpy as np


@dataclass
class AnalysisResult:
    """单个方法的分析结果"""
    method_name: str
    total_images: int
    success_count: int
    success_rate: float
    avg_score: float
    avg_contours: float
    avg_valid_contours: float
    avg_vertices: float
    # 新增：包含中心点分析
    contains_center_count: int
    contains_center_rate: float
    # GT对比
    avg_iou: float  # 平均IoU (Intersection over Union)


def calculate_polygon_area(points: List[Tuple[int, int]]) -> float:
    """计算多边形面积（Shoelace公式）"""
    if len(points) < 3:
        return 0.0

    n = len(points)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]

    return abs(area) / 2.0


def calculate_iou(quad1: List[Tuple[int, int]], quad2: List[Tuple[int, int]]) -> float:
    """
    计算两个四边形的IoU
    简化版：使用面积比例估算
    """
    if not quad1 or not quad2:
        return 0.0

    try:
        area1 = calculate_polygon_area(quad1)
        area2 = calculate_polygon_area(quad2)

        if area1 == 0 or area2 == 0:
            return 0.0

        # 简化：如果两个四边形面积相近，认为有较高重叠
        # 真实IoU需要计算交集，这里用面积比作为近似
        ratio = min(area1, area2) / max(area1, area2)

        return ratio
    except:
        return 0.0


def point_in_polygon(point: Tuple[int, int], polygon: List[Tuple[int, int]]) -> bool:
    """判断点是否在多边形内（射线法）"""
    if len(polygon) < 3:
        return False

    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def analyze_single_result(result_json: Dict, image_size: Tuple[int, int]) -> Dict:
    """分析单个结果"""
    gt_quad = result_json['gt_quad']
    detected_quad = result_json['detected_quad']

    # 图片中心点
    center_x = image_size[0] // 2
    center_y = image_size[1] // 2

    analysis = {
        'success': result_json['detection_success'],
        'score': result_json['best_score'],
        'num_contours': result_json['num_contours'],
        'num_valid_contours': result_json['num_valid_contours'],
        'vertices': result_json['best_contour_vertices'],
        'contains_center': False,
        'iou': 0.0
    }

    # 检查是否包含中心点
    if detected_quad and len(detected_quad) >= 3:
        analysis['contains_center'] = point_in_polygon((center_x, center_y), detected_quad)

    # 计算IoU
    if detected_quad and gt_quad:
        analysis['iou'] = calculate_iou(detected_quad, gt_quad)

    return analysis


def load_all_results(results_dir: Path) -> Dict[str, List[Dict]]:
    """加载所有结果JSON文件"""
    method_results = {
        'baseline': [],
        'no_blur': [],
        'stronger_blur': [],
        'clahe': [],
        'morphology': [],
        'adaptive_canny': []
    }

    # 扫描所有子目录
    for subdir in results_dir.iterdir():
        if not subdir.is_dir():
            continue

        result_json_path = subdir / "results.json"
        if not result_json_path.exists():
            continue

        # 读取结果
        with open(result_json_path, 'r', encoding='utf-8') as f:
            results = json.load(f)

        # 按方法分类
        for result in results:
            method = result['method_name']
            if method in method_results:
                method_results[method].append(result)

    return method_results


def analyze_method_results(results: List[Dict]) -> AnalysisResult:
    """分析单个方法的所有结果"""
    if len(results) == 0:
        return None

    method_name = results[0]['method_name']

    total = len(results)
    success_count = 0
    total_score = 0.0
    total_contours = 0
    total_valid_contours = 0
    total_vertices = 0
    contains_center_count = 0
    total_iou = 0.0

    # 假设图片尺寸（从GT角点推断）
    # 这里简化处理，使用常见的4080x3060
    image_size = (4080, 3060)

    for result in results:
        # 从GT推断图片尺寸
        if result['gt_quad']:
            max_x = max(p[0] for p in result['gt_quad'])
            max_y = max(p[1] for p in result['gt_quad'])
            image_size = (max(image_size[0], max_x + 100), max(image_size[1], max_y + 100))

        analysis = analyze_single_result(result, image_size)

        if analysis['success']:
            success_count += 1

        total_score += analysis['score']
        total_contours += analysis['num_contours']
        total_valid_contours += analysis['num_valid_contours']
        total_vertices += analysis['vertices']

        if analysis['contains_center']:
            contains_center_count += 1

        total_iou += analysis['iou']

    return AnalysisResult(
        method_name=method_name,
        total_images=total,
        success_count=success_count,
        success_rate=success_count / total * 100 if total > 0 else 0,
        avg_score=total_score / total if total > 0 else 0,
        avg_contours=total_contours / total if total > 0 else 0,
        avg_valid_contours=total_valid_contours / total if total > 0 else 0,
        avg_vertices=total_vertices / total if total > 0 else 0,
        contains_center_count=contains_center_count,
        contains_center_rate=contains_center_count / total * 100 if total > 0 else 0,
        avg_iou=total_iou / total if total > 0 else 0
    )


def generate_detailed_report(results_dir: Path, output_dir: Path):
    """生成详细分析报告"""
    print("=" * 80)
    print("预处理方法结果分析")
    print("=" * 80)
    print(f"结果目录: {results_dir}")

    # 加载所有结果
    print("\n加载结果文件...")
    method_results = load_all_results(results_dir)

    # 统计每个方法
    print("\n分析各方法...")
    analyses = []

    for method, results in method_results.items():
        if len(results) == 0:
            print(f"  ⚠️  {method}: 没有结果")
            continue

        analysis = analyze_method_results(results)
        analyses.append(analysis)
        print(f"  ✅ {method}: {len(results)} 张图片")

    # 排序（按成功率）
    analyses.sort(key=lambda x: x.success_rate, reverse=True)

    # 打印汇总表
    print("\n" + "=" * 80)
    print("方法对比汇总")
    print("=" * 80)
    print(f"{'方法':<20} {'图片数':<8} {'成功':<8} {'成功率':<10} {'平均分':<10} {'包含中心':<12} {'平均IoU':<10}")
    print("-" * 80)

    for analysis in analyses:
        print(f"{analysis.method_name:<20} "
              f"{analysis.total_images:<8} "
              f"{analysis.success_count:<8} "
              f"{analysis.success_rate:>6.1f}%   "
              f"{analysis.avg_score:>6.1f}    "
              f"{analysis.contains_center_rate:>6.1f}%      "
              f"{analysis.avg_iou:>6.3f}")

    # 保存详细CSV
    csv_path = output_dir / "detailed_analysis.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Method', 'Total Images', 'Success Count', 'Success Rate (%)',
            'Avg Score', 'Avg Contours', 'Avg Valid Contours', 'Avg Vertices',
            'Contains Center Count', 'Contains Center Rate (%)', 'Avg IoU'
        ])

        for analysis in analyses:
            writer.writerow([
                analysis.method_name,
                analysis.total_images,
                analysis.success_count,
                f"{analysis.success_rate:.1f}",
                f"{analysis.avg_score:.1f}",
                f"{analysis.avg_contours:.1f}",
                f"{analysis.avg_valid_contours:.1f}",
                f"{analysis.avg_vertices:.1f}",
                analysis.contains_center_count,
                f"{analysis.contains_center_rate:.1f}",
                f"{analysis.avg_iou:.3f}"
            ])

    print(f"\n✅ 详细报告已保存: {csv_path}")

    # 分析问题1：检测到幕布而非PPT的情况
    print("\n" + "=" * 80)
    print("问题分析：检测对象偏差")
    print("=" * 80)

    for method, results in method_results.items():
        if len(results) == 0:
            continue

        # 统计IoU分布
        ious = []
        for result in results:
            if result['detected_quad'] and result['gt_quad']:
                iou = calculate_iou(result['detected_quad'], result['gt_quad'])
                ious.append(iou)

        if len(ious) > 0:
            low_iou_count = sum(1 for iou in ious if iou < 0.5)
            print(f"\n{method}:")
            print(f"  低IoU (<0.5): {low_iou_count}/{len(ious)} ({low_iou_count/len(ious)*100:.1f}%)")
            print(f"  平均IoU: {np.mean(ious):.3f}")

    # 分析问题2：是否包含中心点
    print("\n" + "=" * 80)
    print("问题分析：检测框是否包含图片中心")
    print("=" * 80)

    for analysis in analyses:
        not_contain_rate = 100 - analysis.contains_center_rate
        print(f"{analysis.method_name:<20} 不包含中心: {not_contain_rate:>6.1f}%")

    print("\n" + "=" * 80)


def main():
    results_dir = Path(r"D:\Programing\C#\MauiScan\preprocessing_results")
    output_dir = results_dir

    if not results_dir.exists():
        print(f"❌ 结果目录不存在: {results_dir}")
        return

    generate_detailed_report(results_dir, output_dir)

    print("\n" + "=" * 80)
    print("分析完成")
    print("=" * 80)


if __name__ == '__main__':
    main()
