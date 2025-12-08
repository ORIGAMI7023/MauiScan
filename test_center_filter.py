#!/usr/bin/env python3
"""
测试"要求包含中心点"过滤器的效果

对比两种模式：
1. 原始模式（不要求包含中心点）
2. 过滤模式（要求包含中心点）
"""

import sys
from pathlib import Path

# 导入原始测试脚本的函数
sys.path.insert(0, str(Path(__file__).parent))

from test_preprocessing_methods import (
    BaselineMethod, NoBlurMethod, StrongerBlurMethod,
    CLAHEMethod, MorphologyMethod, AdaptiveCannyMethod,
    detect_document_bounds, load_ground_truth,
    PreprocessingResult
)

import cv2
import json
from dataclasses import asdict


def test_single_image_comparison(image_path: Path, json_path: Path, output_dir: Path):
    """对比单张图片的两种检测模式"""

    # 读取图片和GT
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    gt_quad = load_ground_truth(json_path)

    # 测试baseline方法
    method = BaselineMethod()

    # 1. 不要求包含中心点
    preprocessed, edges = method.apply(image)
    result_no_filter = detect_document_bounds(image, preprocessed, edges, method.name, require_center=False)

    # 2. 要求包含中心点
    result_with_filter = detect_document_bounds(image, preprocessed, edges, method.name, require_center=True)

    return {
        'image_name': image_path.name,
        'no_filter': {
            'success': result_no_filter['detection_success'],
            'score': result_no_filter['best_score'],
            'contains_center': result_no_filter['contains_center'],
            'vertices': result_no_filter['best_contour_vertices']
        },
        'with_filter': {
            'success': result_with_filter['detection_success'],
            'score': result_with_filter['best_score'],
            'contains_center': result_with_filter['contains_center'],
            'vertices': result_with_filter['best_contour_vertices']
        }
    }


def main():
    data_dir = Path(r"D:\Programing\C#\MauiScan\AnnotationTool\data")
    output_dir = Path(r"D:\Programing\C#\MauiScan\center_filter_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 扫描所有图片
    image_files = []
    for subdir in ['ios', 'matepad', 'meizu']:
        subdir_path = data_dir / subdir
        if not subdir_path.exists():
            continue

        for img_path in subdir_path.glob("*.jpg"):
            json_path = img_path.with_suffix('.json')
            if json_path.exists():
                image_files.append((img_path, json_path))

        for img_path in subdir_path.glob("*.png"):
            json_path = img_path.with_suffix('.json')
            if json_path.exists():
                image_files.append((img_path, json_path))

    print("=" * 80)
    print("测试中心点过滤器效果")
    print("=" * 80)
    print(f"测试图片数: {len(image_files)}")

    results = []
    improved = 0
    degraded = 0
    unchanged = 0

    for img_path, json_path in image_files:
        result = test_single_image_comparison(img_path, json_path, output_dir)
        if result is None:
            continue

        results.append(result)

        # 统计效果变化
        no_filter_success = result['no_filter']['success']
        with_filter_success = result['with_filter']['success']

        if not no_filter_success and with_filter_success:
            improved += 1
            print(f"✅ {result['image_name']}: 过滤后改善")
        elif no_filter_success and not with_filter_success:
            degraded += 1
            print(f"❌ {result['image_name']}: 过滤后变差")
        else:
            unchanged += 1

    # 保存结果
    with open(output_dir / "comparison.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 统计汇总
    print("\n" + "=" * 80)
    print("汇总结果")
    print("=" * 80)

    no_filter_success_count = sum(1 for r in results if r['no_filter']['success'])
    with_filter_success_count = sum(1 for r in results if r['with_filter']['success'])

    no_filter_contains_center = sum(1 for r in results if r['no_filter']['contains_center'])
    with_filter_contains_center = sum(1 for r in results if r['with_filter']['contains_center'])

    print(f"总图片数: {len(results)}")
    print()
    print(f"不过滤模式:")
    print(f"  成功率: {no_filter_success_count}/{len(results)} ({no_filter_success_count/len(results)*100:.1f}%)")
    print(f"  包含中心点: {no_filter_contains_center}/{len(results)} ({no_filter_contains_center/len(results)*100:.1f}%)")
    print()
    print(f"过滤模式 (要求包含中心点):")
    print(f"  成功率: {with_filter_success_count}/{len(results)} ({with_filter_success_count/len(results)*100:.1f}%)")
    print(f"  包含中心点: {with_filter_contains_center}/{len(results)} ({with_filter_contains_center/len(results)*100:.1f}%)")
    print()
    print(f"效果变化:")
    print(f"  改善: {improved}")
    print(f"  变差: {degraded}")
    print(f"  不变: {unchanged}")

    print(f"\n结果已保存: {output_dir / 'comparison.json'}")


if __name__ == '__main__':
    main()
