#!/usr/bin/env python3
"""
图像预处理方法对CV轮廓检测效果的测试脚本

测试不同预处理方法对 Canny边缘检测 + findContours 的影响
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
    num_valid_contours: int  # 面积>3%的轮廓
    best_score: float
    best_contour_area_ratio: float
    best_contour_vertices: int
    detection_success: bool  # 是否成功检测到四边形
    detected_quad: Optional[List[Tuple[int, int]]]  # 检测到的四边形顶点
    gt_quad: List[Tuple[int, int]]  # Ground Truth四边形


class PreprocessingMethod:
    """预处理方法基类"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def apply(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        应用预处理方法
        返回: (preprocessed_image, edges)
        """
        raise NotImplementedError


class BaselineMethod(PreprocessingMethod):
    """Baseline: 当前算法"""

    def __init__(self, level: str = "mid"):
        """
        level: "low", "mid", "high"
        low: Canny 20/80
        mid: Canny 30/100
        high: Canny 40/120
        """
        self.level = level
        if level == "low":
            self.canny_low = 20
            self.canny_high = 80
        elif level == "high":
            self.canny_low = 40
            self.canny_high = 120
        else:
            self.canny_low = 30
            self.canny_high = 100

        super().__init__(f"baseline_{level}",
                        f"Baseline{level} (Canny{self.canny_low}/{self.canny_high})")

    def apply(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        enhanced = cv2.convertScaleAbs(blurred, alpha=1.15, beta=0)
        edges = cv2.Canny(enhanced, self.canny_low, self.canny_high)

        return enhanced, edges


class NoBlurMethod(PreprocessingMethod):
    """去掉高斯模糊"""

    def __init__(self, level: str = "mid"):
        """
        level: "low", "mid", "high"
        low: Canny 20/80
        mid: Canny 30/100
        high: Canny 40/120
        """
        self.level = level
        if level == "low":
            self.canny_low = 20
            self.canny_high = 80
        elif level == "high":
            self.canny_low = 40
            self.canny_high = 120
        else:
            self.canny_low = 30
            self.canny_high = 100

        super().__init__(f"no_blur_{level}",
                        f"无模糊{level} (Canny{self.canny_low}/{self.canny_high})")

    def apply(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        enhanced = cv2.convertScaleAbs(gray, alpha=1.15, beta=0)
        edges = cv2.Canny(enhanced, self.canny_low, self.canny_high)

        return enhanced, edges


class StrongerBlurMethod(PreprocessingMethod):
    """更强的高斯模糊"""

    def __init__(self, level: str = "mid"):
        """
        level: "low", "mid", "high"
        low: kernel 5x5
        mid: kernel 7x7
        high: kernel 9x9
        """
        self.level = level
        if level == "low":
            self.kernel_size = 5
        elif level == "high":
            self.kernel_size = 9
        else:
            self.kernel_size = 7

        super().__init__(f"stronger_blur_{level}",
                        f"强模糊{level} (Gaussian{self.kernel_size}x{self.kernel_size})")

    def apply(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        blurred = cv2.GaussianBlur(gray, (self.kernel_size, self.kernel_size), 0)
        enhanced = cv2.convertScaleAbs(blurred, alpha=1.15, beta=0)
        edges = cv2.Canny(enhanced, 30, 100)

        return enhanced, edges


class CLAHEMethod(PreprocessingMethod):
    """CLAHE替代线性对比度增强"""

    def __init__(self, level: str = "mid"):
        """
        level: "low", "mid", "high"
        low: clipLimit 1.0
        mid: clipLimit 2.0
        high: clipLimit 3.0
        """
        self.level = level
        if level == "low":
            self.clip_limit = 1.0
        elif level == "high":
            self.clip_limit = 3.0
        else:
            self.clip_limit = 2.0

        super().__init__(f"clahe_{level}",
                        f"CLAHE{level} (clip={self.clip_limit})")

    def apply(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)

        edges = cv2.Canny(enhanced, 30, 100)

        return enhanced, edges


class MorphologyMethod(PreprocessingMethod):
    """Canny后形态学闭运算"""

    def __init__(self, level: str = "mid"):
        """
        level: "low", "mid", "high"
        low: kernel 3x3
        mid: kernel 5x5
        high: kernel 7x7
        """
        self.level = level
        if level == "low":
            self.kernel_size = 3
        elif level == "high":
            self.kernel_size = 7
        else:
            self.kernel_size = 5

        super().__init__(f"morphology_{level}",
                        f"形态学{level} (kernel{self.kernel_size}x{self.kernel_size})")

    def apply(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        enhanced = cv2.convertScaleAbs(blurred, alpha=1.15, beta=0)
        edges = cv2.Canny(enhanced, 30, 100)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.kernel_size, self.kernel_size))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        return enhanced, edges


class AdaptiveCannyMethod(PreprocessingMethod):
    """自适应Canny阈值"""

    def __init__(self, level: str = "mid"):
        """
        level: "low", "mid", "high"
        low: 中值系数 0.5/1.5
        mid: 中值系数 0.66/1.33
        high: 中值系数 0.8/1.2
        """
        self.level = level
        if level == "low":
            self.lower_factor = 0.5
            self.upper_factor = 1.5
        elif level == "high":
            self.lower_factor = 0.8
            self.upper_factor = 1.2
        else:
            self.lower_factor = 0.66
            self.upper_factor = 1.33

        super().__init__(f"adaptive_canny_{level}",
                        f"自适应Canny{level} (系数{self.lower_factor}/{self.upper_factor})")

    def apply(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        enhanced = cv2.convertScaleAbs(blurred, alpha=1.15, beta=0)

        # 自适应Canny阈值：基于图像中值
        median = np.median(enhanced)
        lower = int(max(0, self.lower_factor * median))
        upper = int(min(255, self.upper_factor * median))

        edges = cv2.Canny(enhanced, lower, upper)

        return enhanced, edges


class SharpeningMethod(PreprocessingMethod):
    """边缘锐化"""

    def __init__(self, level: str = "mid"):
        """
        level: "low", "mid", "high"
        low: 强度 0.5
        mid: 强度 1.0
        high: 强度 2.0
        """
        self.level = level
        if level == "low":
            self.strength = 0.5
        elif level == "high":
            self.strength = 2.0
        else:
            self.strength = 1.0

        super().__init__(f"sharpening_{level}", f"边缘锐化{level} (强度{self.strength})")

    def apply(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 锐化核
        kernel = np.array([[-1, -1, -1],
                          [-1, 9, -1],
                          [-1, -1, -1]]) * self.strength / 9.0
        kernel[1, 1] = 1 + self.strength

        # 应用锐化
        sharpened = cv2.filter2D(gray, -1, kernel)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

        # 高斯模糊
        blurred = cv2.GaussianBlur(sharpened, (5, 5), 0)
        enhanced = cv2.convertScaleAbs(blurred, alpha=1.15, beta=0)

        edges = cv2.Canny(enhanced, 30, 100)

        return enhanced, edges


class BilateralMethod(PreprocessingMethod):
    """双边滤波（保边去噪）"""

    def __init__(self, level: str = "mid"):
        """
        level: "low", "mid", "high"
        low: d=5, sigma=30
        mid: d=9, sigma=50
        high: d=13, sigma=75
        """
        self.level = level
        if level == "low":
            self.d = 5
            self.sigma = 30
        elif level == "high":
            self.d = 13
            self.sigma = 75
        else:
            self.d = 9
            self.sigma = 50

        super().__init__(f"bilateral_{level}",
                        f"双边滤波{level} (d={self.d},σ={self.sigma})")

    def apply(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 双边滤波
        filtered = cv2.bilateralFilter(gray, self.d, self.sigma, self.sigma)
        enhanced = cv2.convertScaleAbs(filtered, alpha=1.15, beta=0)

        edges = cv2.Canny(enhanced, 30, 100)

        return enhanced, edges


class MorphGradientMethod(PreprocessingMethod):
    """形态学梯度"""

    def __init__(self, level: str = "mid"):
        """
        level: "low", "mid", "high"
        low: kernel 3x3
        mid: kernel 5x5
        high: kernel 7x7
        """
        self.level = level
        if level == "low":
            self.kernel_size = 3
        elif level == "high":
            self.kernel_size = 7
        else:
            self.kernel_size = 5

        super().__init__(f"morph_gradient_{level}",
                        f"形态学梯度{level} (kernel{self.kernel_size}x{self.kernel_size})")

    def apply(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        enhanced = cv2.convertScaleAbs(blurred, alpha=1.15, beta=0)

        # 形态学梯度（膨胀 - 腐蚀）
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.kernel_size, self.kernel_size))
        gradient = cv2.morphologyEx(enhanced, cv2.MORPH_GRADIENT, kernel)

        # 二值化后作为边缘
        _, edges = cv2.threshold(gradient, 30, 255, cv2.THRESH_BINARY)

        return enhanced, edges


class MultiScaleCannyMethod(PreprocessingMethod):
    """多尺度Canny"""

    def __init__(self, level: str = "mid"):
        """
        level: "low", "mid", "high"
        low: 2个尺度
        mid: 3个尺度
        high: 4个尺度
        """
        self.level = level
        if level == "low":
            self.num_scales = 2
        elif level == "high":
            self.num_scales = 4
        else:
            self.num_scales = 3

        super().__init__(f"multi_scale_canny_{level}",
                        f"多尺度Canny{level} ({self.num_scales}尺度)")

    def apply(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 对比度增强
        enhanced = cv2.convertScaleAbs(gray, alpha=1.15, beta=0)

        # 多尺度Canny
        edges_list = []
        for i in range(self.num_scales):
            kernel_size = 3 + i * 2  # 3, 5, 7, 9
            blurred = cv2.GaussianBlur(enhanced, (kernel_size, kernel_size), 0)
            edges = cv2.Canny(blurred, 30, 100)
            edges_list.append(edges)

        # 融合所有尺度的边缘
        edges_combined = np.maximum.reduce(edges_list)

        return enhanced, edges_combined


def calculate_contour_score(contour: np.ndarray, edges: np.ndarray, image_area: float) -> float:
    """
    复刻C++的轮廓评分算法
    """
    score = 0.0

    # 1. 面积分数 (0-30分)
    area = cv2.contourArea(contour)
    area_ratio = area / image_area
    score += min(area_ratio * 100.0, 30.0)

    # 2. 四边形拟合度 (0-40分)
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.03 * peri, True)

    if len(approx) == 4:
        score += 40.0
    elif len(approx) == 5 or len(approx) == 6:
        score += 25.0
    elif 7 <= len(approx) <= 10:
        score += 12.0

    # 3. 凸性分数 (0-15分)
    if cv2.isContourConvex(approx):
        score += 15.0

    # 4. 边缘清晰度 (0-15分)
    compactness = (peri * peri) / area if area > 0 else 0
    normalized_compactness = min(compactness / 50.0, 1.0)
    score += (1.0 - normalized_compactness) * 15.0

    return score


def point_in_contour(point: tuple, contour: np.ndarray) -> bool:
    """检查点是否在轮廓内"""
    return cv2.pointPolygonTest(contour, point, False) >= 0


def detect_document_bounds(
    image: np.ndarray,
    preprocessed: np.ndarray,
    edges: np.ndarray,
    method_name: str,
    require_center: bool = True
) -> Dict:
    """
    复刻C++的文档边界检测算法

    Args:
        require_center: 是否要求检测框包含图片中心点
    """
    h, w = image.shape[:2]
    image_area = w * h
    center_point = (w // 2, h // 2)

    # 查找轮廓
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

    # 筛选候选轮廓（面积>3%）
    min_area = image_area * 0.03
    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        # 检查是否包含中心点
        contains_center = point_in_contour(center_point, contour)

        # 如果要求包含中心点，则过滤掉不包含的
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

    # 选择最高分的轮廓
    candidates.sort(key=lambda x: x['score'], reverse=True)
    best_candidate = candidates[0]

    result['best_score'] = best_candidate['score']
    result['best_contour'] = best_candidate['contour']
    result['best_contour_area_ratio'] = best_candidate['area'] / image_area
    result['contains_center'] = best_candidate['contains_center']

    # 尝试近似为四边形
    peri = cv2.arcLength(best_candidate['contour'], True)
    approx = cv2.approxPolyDP(best_candidate['contour'], 0.03 * peri, True)

    result['best_contour_vertices'] = len(approx)

    # 检查是否成功检测（评分>38且为四边形）
    if best_candidate['score'] >= 38.0:
        if len(approx) == 4:
            result['detection_success'] = True
            result['detected_quad'] = [(int(p[0][0]), int(p[0][1])) for p in approx]
        elif 4 <= len(approx) <= 6:
            # 尝试更宽松的近似
            approx2 = cv2.approxPolyDP(best_candidate['contour'], 0.05 * peri, True)
            if len(approx2) == 4:
                result['detection_success'] = True
                result['detected_quad'] = [(int(p[0][0]), int(p[0][1])) for p in approx2]

    return result


def create_visualization(
    original: np.ndarray,
    preprocessed: np.ndarray,
    edges: np.ndarray,
    detection_result: Dict,
    gt_quad: List[Tuple[int, int]],
    method_name: str,
    method_desc: str
) -> np.ndarray:
    """
    创建4宫格可视化图
    """
    h, w = original.shape[:2]

    # 根据图片尺寸调整字体大小和线宽
    scale_factor = max(h, w) / 1000.0  # 基于图片大小的缩放因子
    font_scale = max(0.67, scale_factor * 0.5)  # 缩小到1/3
    thickness = max(2, int(scale_factor * 1.33))  # 缩小到1/3
    line_width = max(3, int(scale_factor * 1.67))  # 缩小到1/3

    # 1. 原图 + GT四边形
    img1 = original.copy()
    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)

    # 绘制GT四边形（绿色）
    gt_pts = np.array(gt_quad, dtype=np.int32)
    cv2.polylines(img1, [gt_pts], True, (0, 255, 0), line_width)
    cv2.putText(img1, "Ground Truth", (40, int(80*scale_factor)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

    # 2. Canny边缘检测结果
    img2 = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cv2.putText(img2, "Canny Edges", (40, int(80*scale_factor)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    # 3. 检测到的轮廓
    img3 = original.copy()
    if len(img3.shape) == 2:
        img3 = cv2.cvtColor(img3, cv2.COLOR_GRAY2BGR)

    if detection_result['best_contour'] is not None:
        # 绘制最佳轮廓（红色）
        cv2.drawContours(img3, [detection_result['best_contour']], -1, (0, 0, 255), line_width)

        # 如果检测到四边形，绘制（黄色）
        if detection_result['detected_quad']:
            quad_pts = np.array(detection_result['detected_quad'], dtype=np.int32)
            cv2.polylines(img3, [quad_pts], True, (0, 255, 255), line_width)

    status_text = "SUCCESS" if detection_result['detection_success'] else "FAILED"
    status_color = (0, 255, 0) if detection_result['detection_success'] else (0, 0, 255)
    cv2.putText(img3, status_text, (40, int(80*scale_factor)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, status_color, thickness)

    # 4. 统计信息
    img4 = np.zeros((h, w, 3), dtype=np.uint8)

    info_lines = [
        f"Method: {method_name}",
        f"{method_desc[:40]}",  # 截断过长的描述
        "",
        f"Contours: {detection_result['num_contours']}",
        f"Valid: {detection_result['num_valid_contours']}",
        f"Score: {detection_result['best_score']:.1f}",
        f"Area: {detection_result['best_contour_area_ratio']*100:.1f}%",
        f"Vertices: {detection_result['best_contour_vertices']}",
        "",
        f"{status_text}",
    ]

    line_spacing = int(80 * scale_factor)
    y_offset = int(100 * scale_factor)

    for i, line in enumerate(info_lines):
        color = (255, 255, 255)
        if "SUCCESS" in line:
            color = (0, 255, 0)
        elif "FAILED" in line:
            color = (0, 0, 255)

        cv2.putText(img4, line, (40, y_offset + i*line_spacing),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale*0.9, color, thickness)

    # 拼接4宫格
    top_row = np.hstack([img1, img2])
    bottom_row = np.hstack([img3, img4])
    result = np.vstack([top_row, bottom_row])

    return result


def load_ground_truth(json_path: Path) -> List[Tuple[int, int]]:
    """加载Ground Truth角点"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    corners = data['Corners']
    # 返回: TL, TR, BR, BL
    return [(c['X'], c['Y']) for c in corners]


def process_single_image(
    image_path: Path,
    gt_quad: List[Tuple[int, int]],
    methods: List[PreprocessingMethod],
    output_dir: Path,
    quiet: bool = False
) -> List[PreprocessingResult]:
    """处理单张图片"""
    if not quiet:
        print(f"\n{'='*60}")
        print(f"处理: {image_path.name}")
        print(f"{'='*60}")

    # 读取图片
    image = cv2.imread(str(image_path))
    if image is None:
        if not quiet:
            print(f"  ❌ 无法读取图片")
        return []

    h, w = image.shape[:2]
    if not quiet:
        print(f"图片尺寸: {w}x{h}")
        print(f"GT角点: {gt_quad}")

    # 创建输出目录
    img_output_dir = output_dir / image_path.stem
    img_output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for method in methods:
        if not quiet:
            print(f"\n  测试方法: {method.name}")

        # 应用预处理
        preprocessed, edges = method.apply(image)

        # 检测文档边界
        detection_result = detect_document_bounds(image, preprocessed, edges, method.name)

        if not quiet:
            print(f"    轮廓数: {detection_result['num_contours']}, 有效: {detection_result['num_valid_contours']}")
            print(f"    最佳评分: {detection_result['best_score']:.1f}")
            print(f"    检测结果: {'✅ 成功' if detection_result['detection_success'] else '❌ 失败'}")

        # 创建可视化
        vis = create_visualization(
            image, preprocessed, edges,
            detection_result, gt_quad,
            method.name, method.description
        )

        vis_path = img_output_dir / f"{method.name}.jpg"
        cv2.imwrite(str(vis_path), vis)

        # 保存结果
        result = PreprocessingResult(
            image_name=image_path.name,
            method_name=method.name,
            num_contours=detection_result['num_contours'],
            num_valid_contours=detection_result['num_valid_contours'],
            best_score=detection_result['best_score'],
            best_contour_area_ratio=detection_result['best_contour_area_ratio'],
            best_contour_vertices=detection_result['best_contour_vertices'],
            detection_success=detection_result['detection_success'],
            detected_quad=detection_result['detected_quad'],
            gt_quad=gt_quad
        )
        results.append(result)

    # 保存JSON
    json_data = [asdict(r) for r in results]
    with open(img_output_dir / "results.json", 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    if not quiet:
        print(f"\n✅ 完成，结果: {img_output_dir}")

    return results


def process_image_wrapper(args):
    """多进程包装函数"""
    img_path, json_path, output_dir = args

    # 重新初始化预处理方法（每个进程需要自己的实例）
    methods = []

    # 现有6个方法 × 3参数 = 18个
    for level in ["low", "mid", "high"]:
        methods.append(BaselineMethod(level))
        methods.append(NoBlurMethod(level))
        methods.append(StrongerBlurMethod(level))
        methods.append(CLAHEMethod(level))
        methods.append(MorphologyMethod(level))
        methods.append(AdaptiveCannyMethod(level))

    # 新增4个方法 × 3参数 = 12个
    for level in ["low", "mid", "high"]:
        methods.append(SharpeningMethod(level))
        methods.append(BilateralMethod(level))
        methods.append(MorphGradientMethod(level))
        methods.append(MultiScaleCannyMethod(level))

    # 加载GT
    gt_quad = load_ground_truth(json_path)

    # 处理图片（静默模式）
    results = process_single_image(img_path, gt_quad, methods, output_dir, quiet=True)

    return (img_path.name, results)


def generate_summary_report(all_results: List[PreprocessingResult], output_path: Path):
    """生成汇总CSV报告"""
    # 按方法统计
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

        for method, stats in sorted(method_stats.items()):
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

    print(f"\n{'='*60}")
    print("汇总报告")
    print(f"{'='*60}")
    print(f"{'Method':<20} {'Images':<10} {'Success':<10} {'Rate':<10} {'Avg Score':<12}")
    print('-' * 60)
    for method, stats in sorted(method_stats.items()):
        success_rate = (stats['success'] / stats['total'] * 100) if stats['total'] > 0 else 0
        avg_score = stats['total_score'] / stats['total'] if stats['total'] > 0 else 0
        print(f"{method:<20} {stats['total']:<10} {stats['success']:<10} {success_rate:>6.1f}%   {avg_score:>8.1f}")


def main():
    # 数据路径
    data_dir = Path(r"D:\Programing\C#\MauiScan\AnnotationTool\data")
    output_dir = Path(r"D:\Programing\C#\MauiScan\preprocessing_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化30种预处理方法组合
    methods = []

    # 现有6个方法 × 3参数 = 18个
    for level in ["low", "mid", "high"]:
        methods.append(BaselineMethod(level))
        methods.append(NoBlurMethod(level))
        methods.append(StrongerBlurMethod(level))
        methods.append(CLAHEMethod(level))
        methods.append(MorphologyMethod(level))
        methods.append(AdaptiveCannyMethod(level))

    # 新增4个方法 × 3参数 = 12个
    for level in ["low", "mid", "high"]:
        methods.append(SharpeningMethod(level))
        methods.append(BilateralMethod(level))
        methods.append(MorphGradientMethod(level))
        methods.append(MultiScaleCannyMethod(level))

    print("="*80)
    print("图像预处理方法测试 - 30种方法组合")
    print("="*80)
    print(f"数据目录: {data_dir}")
    print(f"输出目录: {output_dir}")
    print(f"\n测试方法: {len(methods)}个")
    print("  现有方法(3水平): baseline, no_blur, stronger_blur, clahe, morphology, adaptive_canny")
    print("  新增方法(3水平): sharpening, bilateral, morph_gradient, multi_scale_canny")
    print("\n参数水平: low(低), mid(中), high(高)")

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

    print(f"\n找到 {len(image_files)} 张标注图片")

    if len(image_files) == 0:
        print("❌ 没有找到图片文件")
        return

    # 获取CPU核心数
    cpu_count = multiprocessing.cpu_count()
    print(f"\n检测到 {cpu_count} 个CPU核心，使用多进程并行处理...")

    # 批量处理（多进程）
    all_results = []
    start_time = time.time()

    # 准备任务参数
    tasks = [(img_path, json_path, output_dir) for img_path, json_path in image_files]

    # 使用ProcessPoolExecutor进行并行处理
    with ProcessPoolExecutor(max_workers=cpu_count) as executor:
        # 提交所有任务
        futures = {executor.submit(process_image_wrapper, task): task for task in tasks}

        # 处理完成的任务
        completed = 0
        for future in as_completed(futures):
            completed += 1
            try:
                img_name, results = future.result()
                all_results.extend(results)
                print(f"[{completed}/{len(image_files)}] ✅ {img_name}")
            except Exception as e:
                task = futures[future]
                img_path = task[0]
                print(f"[{completed}/{len(image_files)}] ❌ {img_path.name} 失败: {e}")

    elapsed_time = time.time() - start_time
    print(f"\n处理完成，总耗时: {elapsed_time:.1f}秒 (平均 {elapsed_time/len(image_files):.2f}秒/张)")

    # 生成汇总报告
    summary_path = output_dir / "summary.csv"
    generate_summary_report(all_results, summary_path)

    print(f"\n{'='*60}")
    print(f"✅ 所有结果保存在: {output_dir.absolute()}")
    print(f"✅ 汇总报告: {summary_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
