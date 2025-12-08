#!/usr/bin/env python3
"""
组合测试脚本：加密采样 + 预处理组合

测试策略：
1. 对参数敏感的算法：加密采样（包括参数=0，即不使用）
2. 对参数不敏感的算法：只保留最优参数
3. 预处理方法 + 边缘检测方法的两阶段组合
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import csv
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time


@dataclass
class PreprocessingResult:
    """单张图片的预处理结果"""
    image_name: str
    method_name: str
    num_contours: int
    num_valid_contours: int
    best_score: float
    best_contour_area_ratio: float
    best_contour_vertices: int
    detection_success: bool
    detected_quad: Optional[List[Tuple[int, int]]]
    gt_quad: List[Tuple[int, int]]


# ==================== 预处理层方法 ====================

class PreprocessingMethod:
    """预处理方法基类"""
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def apply(self, image: np.ndarray) -> np.ndarray:
        """返回预处理后的灰度图"""
        raise NotImplementedError


class BaselinePreprocess(PreprocessingMethod):
    """Baseline预处理：高斯模糊 + 对比度增强"""
    def __init__(self):
        super().__init__("baseline", "Baseline (Gaussian5x5 + Contrast1.15)")

    def apply(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        enhanced = cv2.convertScaleAbs(blurred, alpha=1.15, beta=0)
        return enhanced


class SharpeningPreprocess(PreprocessingMethod):
    """锐化预处理"""
    def __init__(self):
        super().__init__("sharpening", "Sharpening (强度1.0)")

    def apply(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        kernel = np.array([[-1, -1, -1],
                          [-1, 9, -1],
                          [-1, -1, -1]]) / 9.0
        kernel[1, 1] = 2.0

        sharpened = cv2.filter2D(gray, -1, kernel)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

        blurred = cv2.GaussianBlur(sharpened, (5, 5), 0)
        enhanced = cv2.convertScaleAbs(blurred, alpha=1.15, beta=0)
        return enhanced


class BilateralPreprocess(PreprocessingMethod):
    """双边滤波预处理 - 加密采样"""
    def __init__(self, d: int = 0):
        """
        d: 0(不使用), 7, 9, 11
        """
        self.d = d
        if d == 0:
            super().__init__("bilateral_none", "无双边滤波")
        else:
            super().__init__(f"bilateral_d{d}", f"双边滤波 (d={d})")

    def apply(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        if self.d > 0:
            sigma = self.d * 5  # d=7→σ=35, d=9→σ=45, d=11→σ=55
            filtered = cv2.bilateralFilter(gray, self.d, sigma, sigma)
            enhanced = cv2.convertScaleAbs(filtered, alpha=1.15, beta=0)
        else:
            # d=0: 不使用双边滤波
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            enhanced = cv2.convertScaleAbs(blurred, alpha=1.15, beta=0)

        return enhanced


class CLAHEPreprocess(PreprocessingMethod):
    """CLAHE预处理（保留low）"""
    def __init__(self):
        super().__init__("clahe_low", "CLAHE (clip=1.0)")

    def apply(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        return enhanced


# ==================== 边缘检测层方法 ====================

class EdgeDetectionMethod:
    """边缘检测方法基类"""
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def apply(self, preprocessed: np.ndarray) -> np.ndarray:
        """返回边缘图"""
        raise NotImplementedError


class MorphGradientEdge(EdgeDetectionMethod):
    """形态学梯度 - 加密采样"""
    def __init__(self, kernel_size: int = 0):
        """
        kernel_size: 0(不使用), 2, 3, 4, 5
        """
        self.kernel_size = kernel_size
        if kernel_size == 0:
            super().__init__("morph_grad_none", "无形态学梯度")
        else:
            super().__init__(f"morph_grad_k{kernel_size}",
                           f"形态学梯度 (kernel={kernel_size}x{kernel_size})")

    def apply(self, preprocessed: np.ndarray) -> np.ndarray:
        if self.kernel_size == 0:
            # 不使用形态学梯度，直接Canny
            edges = cv2.Canny(preprocessed, 30, 100)
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                              (self.kernel_size, self.kernel_size))
            gradient = cv2.morphologyEx(preprocessed, cv2.MORPH_GRADIENT, kernel)
            _, edges = cv2.threshold(gradient, 30, 255, cv2.THRESH_BINARY)

        return edges


class MultiScaleCannyEdge(EdgeDetectionMethod):
    """多尺度Canny - 加密采样"""
    def __init__(self, num_scales: int = 0, canny_low: int = 30, canny_high: int = 100):
        """
        num_scales: 0(单尺度), 2, 3, 4, 5
        canny_low/high: Canny阈值
        """
        self.num_scales = num_scales
        self.canny_low = canny_low
        self.canny_high = canny_high

        if num_scales == 0:
            super().__init__(f"canny_{canny_low}_{canny_high}",
                           f"单尺度Canny ({canny_low}/{canny_high})")
        else:
            super().__init__(f"multi_canny_s{num_scales}_c{canny_low}_{canny_high}",
                           f"多尺度Canny ({num_scales}尺度, {canny_low}/{canny_high})")

    def apply(self, preprocessed: np.ndarray) -> np.ndarray:
        if self.num_scales == 0:
            # 单尺度
            edges = cv2.Canny(preprocessed, self.canny_low, self.canny_high)
        else:
            # 多尺度
            edges_list = []
            for i in range(self.num_scales):
                kernel_size = 3 + i * 2
                blurred = cv2.GaussianBlur(preprocessed, (kernel_size, kernel_size), 0)
                edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
                edges_list.append(edges)
            edges = np.maximum.reduce(edges_list)

        return edges


# ==================== 轮廓检测和评分 ====================

def point_in_contour(point: tuple, contour: np.ndarray) -> bool:
    """检查点是否在轮廓内"""
    return cv2.pointPolygonTest(contour, point, False) >= 0


def calculate_contour_score(contour: np.ndarray, edges: np.ndarray, image_area: float) -> float:
    """复刻C++的轮廓评分算法"""
    score = 0.0

    area = cv2.contourArea(contour)
    area_ratio = area / image_area
    score += min(area_ratio * 100.0, 30.0)

    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.03 * peri, True)

    if len(approx) == 4:
        score += 40.0
    elif len(approx) == 5 or len(approx) == 6:
        score += 25.0
    elif 7 <= len(approx) <= 10:
        score += 12.0

    if cv2.isContourConvex(approx):
        score += 15.0

    compactness = (peri * peri) / area if area > 0 else 0
    normalized_compactness = min(compactness / 50.0, 1.0)
    score += (1.0 - normalized_compactness) * 15.0

    return score


def detect_document_bounds(
    image: np.ndarray,
    preprocessed: np.ndarray,
    edges: np.ndarray,
    method_name: str,
    require_center: bool = False
) -> Dict:
    """文档边界检测"""
    h, w = image.shape[:2]
    image_area = w * h
    center_point = (w // 2, h // 2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = {
        'num_contours': len(contours),
        'num_valid_contours': 0,
        'best_score': 0.0,
        'best_contour': None,
        'best_contour_area_ratio': 0.0,
        'best_contour_vertices': 0,
        'detected_quad': None,
        'detection_success': False,
        'contains_center': False
    }

    if len(contours) == 0:
        return result

    min_area = image_area * 0.03
    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        contains_center = point_in_contour(center_point, contour)

        if require_center and not contains_center:
            continue

        result['num_valid_contours'] += 1

        score = calculate_contour_score(contour, edges, image_area)
        candidates.append({
            'contour': contour,
            'score': score,
            'area': area,
            'contains_center': contains_center
        })

    if len(candidates) == 0:
        return result

    candidates.sort(key=lambda x: x['score'], reverse=True)
    best_candidate = candidates[0]

    result['best_score'] = best_candidate['score']
    result['best_contour'] = best_candidate['contour']
    result['best_contour_area_ratio'] = best_candidate['area'] / image_area
    result['contains_center'] = best_candidate['contains_center']

    peri = cv2.arcLength(best_candidate['contour'], True)
    approx = cv2.approxPolyDP(best_candidate['contour'], 0.03 * peri, True)

    result['best_contour_vertices'] = len(approx)

    if best_candidate['score'] >= 38.0:
        if len(approx) == 4:
            result['detection_success'] = True
            result['detected_quad'] = [(int(p[0][0]), int(p[0][1])) for p in approx]
        elif 4 <= len(approx) <= 6:
            approx2 = cv2.approxPolyDP(best_candidate['contour'], 0.05 * peri, True)
            if len(approx2) == 4:
                result['detection_success'] = True
                result['detected_quad'] = [(int(p[0][0]), int(p[0][1])) for p in approx2]

    return result


def load_ground_truth(json_path: Path) -> List[Tuple[int, int]]:
    """加载Ground Truth角点"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    corners = data['Corners']
    return [(c['X'], c['Y']) for c in corners]


def process_single_combination(args):
    """处理单个图片的单个组合方法"""
    img_path, json_path, preprocess_method, edge_method = args

    # 读取图片
    image = cv2.imread(str(img_path))
    if image is None:
        return None

    gt_quad = load_ground_truth(json_path)

    # 组合方法名
    method_name = f"{preprocess_method.name}+{edge_method.name}"

    # 1. 预处理
    preprocessed = preprocess_method.apply(image)

    # 2. 边缘检测
    edges = edge_method.apply(preprocessed)

    # 3. 文档检测
    detection_result = detect_document_bounds(image, preprocessed, edges, method_name)

    # 4. 保存结果
    result = PreprocessingResult(
        image_name=img_path.name,
        method_name=method_name,
        num_contours=detection_result['num_contours'],
        num_valid_contours=detection_result['num_valid_contours'],
        best_score=detection_result['best_score'],
        best_contour_area_ratio=detection_result['best_contour_area_ratio'],
        best_contour_vertices=detection_result['best_contour_vertices'],
        detection_success=detection_result['detection_success'],
        detected_quad=detection_result['detected_quad'],
        gt_quad=gt_quad
    )

    return result


def generate_summary_report(all_results: List[PreprocessingResult], output_path: Path):
    """生成汇总CSV报告"""
    method_stats = {}

    for result in all_results:
        method = result.method_name
        if method not in method_stats:
            method_stats[method] = {
                'total': 0,
                'success': 0,
                'total_score': 0.0,
                'total_contours': 0,
                'total_valid_contours': 0
            }

        stats = method_stats[method]
        stats['total'] += 1
        if result.detection_success:
            stats['success'] += 1
        stats['total_score'] += result.best_score
        stats['total_contours'] += result.num_contours
        stats['total_valid_contours'] += result.num_valid_contours

    # 写入CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Method', 'Total Images', 'Success Count', 'Success Rate (%)',
            'Avg Score', 'Avg Contours', 'Avg Valid Contours'
        ])

        for method, stats in sorted(method_stats.items(),
                                   key=lambda x: x[1]['success'] / x[1]['total'] if x[1]['total'] > 0 else 0,
                                   reverse=True):
            success_rate = (stats['success'] / stats['total'] * 100) if stats['total'] > 0 else 0
            avg_score = stats['total_score'] / stats['total'] if stats['total'] > 0 else 0
            avg_contours = stats['total_contours'] / stats['total'] if stats['total'] > 0 else 0
            avg_valid = stats['total_valid_contours'] / stats['total'] if stats['total'] > 0 else 0

            writer.writerow([
                method,
                stats['total'],
                stats['success'],
                f"{success_rate:.1f}",
                f"{avg_score:.1f}",
                f"{avg_contours:.1f}",
                f"{avg_valid:.1f}"
            ])

    # 打印Top 10
    print("\n" + "="*80)
    print("TOP 10 最佳组合")
    print("="*80)
    print(f"{'Method':<50} {'Images':<10} {'Success':<10} {'Rate':<10}")
    print("-" * 80)

    sorted_methods = sorted(method_stats.items(),
                          key=lambda x: x[1]['success'] / x[1]['total'] if x[1]['total'] > 0 else 0,
                          reverse=True)

    for i, (method, stats) in enumerate(sorted_methods[:10], 1):
        success_rate = (stats['success'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"{i:2}. {method:<47} {stats['total']:<10} {stats['success']:<10} {success_rate:>6.1f}%")


def main():
    data_dir = Path(r"D:\Programing\C#\MauiScan\AnnotationTool\data")
    output_dir = Path(r"D:\Programing\C#\MauiScan\combination_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("组合测试：加密采样 + 预处理层×边缘检测层")
    print("="*80)

    # 定义预处理层方法
    preprocess_methods = [
        BaselinePreprocess(),
        SharpeningPreprocess(),
        BilateralPreprocess(0),   # 不使用
        BilateralPreprocess(7),
        BilateralPreprocess(9),
        BilateralPreprocess(11),
        CLAHEPreprocess(),
    ]

    # 定义边缘检测层方法
    edge_methods = [
        # 形态学梯度 - 加密采样
        MorphGradientEdge(0),  # 不使用
        MorphGradientEdge(2),
        MorphGradientEdge(3),
        MorphGradientEdge(4),
        MorphGradientEdge(5),

        # 多尺度Canny - 加密采样
        MultiScaleCannyEdge(0, 30, 100),  # 单尺度，原始阈值
        MultiScaleCannyEdge(0, 20, 80),   # 单尺度，低阈值
        MultiScaleCannyEdge(0, 40, 120),  # 单尺度，高阈值
        MultiScaleCannyEdge(3, 30, 100),  # 3尺度，原始阈值
        MultiScaleCannyEdge(4, 30, 100),  # 4尺度，原始阈值
        MultiScaleCannyEdge(5, 30, 100),  # 5尺度，原始阈值
        MultiScaleCannyEdge(4, 20, 80),   # 4尺度，低阈值
        MultiScaleCannyEdge(4, 40, 120),  # 4尺度，高阈值
    ]

    total_combinations = len(preprocess_methods) * len(edge_methods)
    print(f"\n预处理方法: {len(preprocess_methods)}个")
    for m in preprocess_methods:
        print(f"  - {m.name}")

    print(f"\n边缘检测方法: {len(edge_methods)}个")
    for m in edge_methods:
        print(f"  - {m.name}")

    print(f"\n总组合数: {total_combinations}个")

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

    print(f"\n找到图片: {len(image_files)}张")

    total_tests = len(image_files) * total_combinations
    print(f"总测试数: {total_tests:,}个")

    # 获取CPU核心数
    cpu_count = multiprocessing.cpu_count()
    print(f"\nCPU核心数: {cpu_count}")
    print("\n开始测试...")

    # 准备所有任务
    tasks = []
    for img_path, json_path in image_files:
        for preprocess in preprocess_methods:
            for edge in edge_methods:
                tasks.append((img_path, json_path, preprocess, edge))

    # 多进程处理
    all_results = []
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=cpu_count) as executor:
        futures = {executor.submit(process_single_combination, task): task for task in tasks}

        completed = 0
        for future in as_completed(futures):
            completed += 1
            if completed % 500 == 0:
                elapsed = time.time() - start_time
                progress = completed / len(tasks) * 100
                print(f"进度: {completed}/{len(tasks)} ({progress:.1f}%) - 耗时: {elapsed:.1f}秒")

            try:
                result = future.result()
                if result:
                    all_results.append(result)
            except Exception as e:
                print(f"错误: {e}")

    elapsed_time = time.time() - start_time
    print(f"\n处理完成！总耗时: {elapsed_time:.1f}秒")
    print(f"平均速度: {len(tasks)/elapsed_time:.1f}个/秒")

    # 生成汇总报告
    summary_path = output_dir / "combination_summary.csv"
    generate_summary_report(all_results, summary_path)

    print(f"\n✅ 结果已保存: {output_dir.absolute()}")
    print(f"✅ 汇总报告: {summary_path}")
    print("="*80)


if __name__ == '__main__':
    main()
