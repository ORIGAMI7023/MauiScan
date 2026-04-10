"""
数据增强脚本 - 幕布边框检测训练数据生成

功能：
- 几何变换：透视变换、旋转、缩放、翻转
- 光学增强：亮度、对比度、高斯噪声、运动模糊
- 遮挡模拟：随机遮挡、线条干扰
"""

import os
import json
import random
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Dict, Optional


class DataAugmentor:
    """数据增强器"""

    def __init__(self, augment_factor: int = 200):
        """
        Args:
            augment_factor: 每个原始样本生成的增强样本数量
        """
        self.augment_factor = augment_factor
        self.rng = np.random.default_rng()

    def augment_image(self,
                     image: np.ndarray,
                     corners: List[Dict[str, int]],
                     aug_id: int) -> Tuple[np.ndarray, List[Dict[str, int]]]:
        """
        对单张图片进行增强

        Args:
            image: 原始图片 (BGR)
            corners: 角点坐标 [{'X': x, 'Y': y}, ...]
            aug_id: 增强样本编号

        Returns:
            (增强后的图片, 增强后的角点)
        """
        h, w = image.shape[:2]

        # 1. 几何变换（必选其一）
        geom_type = random.choice(['perspective', 'rotate', 'scale', 'flip'])
        if geom_type == 'perspective':
            image, corners = self._perspective_transform(image, corners)
        elif geom_type == 'rotate':
            image, corners = self._rotate(image, corners, max_angle=15)
        elif geom_type == 'scale':
            image, corners = self._scale(image, corners, scale_range=(0.8, 1.2))
        elif geom_type == 'flip':
            image, corners = self._flip(image, corners)

        # 2. 光学增强（随机组合）
        if random.random() < 0.7:
            image = self._adjust_brightness(image, delta=random.randint(-50, 50))
        if random.random() < 0.5:
            image = self._adjust_contrast(image, factor=random.uniform(0.7, 1.3))
        if random.random() < 0.3:
            image = self._add_gaussian_noise(image, mean=0, std=random.randint(5, 25))
        if random.random() < 0.2:
            image = self._add_motion_blur(image, kernel_size=random.randint(3, 7))

        # 3. 遮挡模拟（随机应用）
        if random.random() < 0.4:
            image, corners = self._random_occlusion(image, corners)

        if random.random() < 0.3:
            image, corners = self._add_line_interference(image, corners)

        return image, corners

    # ============ 几何变换 ============

    def _perspective_transform(self,
                              image: np.ndarray,
                              corners: List[Dict[str, int]]) -> Tuple[np.ndarray, List[Dict[str, int]]]:
        """透视变换"""
        h, w = image.shape[:2]

        # 原始角点
        src_pts = np.float32([[c['X'], c['Y']] for c in corners])

        # 随机偏移量（最大 5% 图片尺寸）
        max_offset_x = w * 0.05
        max_offset_y = h * 0.05

        dst_pts = src_pts + self.rng.uniform(-max_offset_x, max_offset_x, size=(4, 2))
        dst_pts = dst_pts.astype(np.float32)

        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # 应用变换
        warped = cv2.warpPerspective(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # 更新角点坐标
        new_corners = []
        for i, pt in enumerate(dst_pts):
            new_corners.append({'X': int(pt[0]), 'Y': int(pt[1])})

        return warped, new_corners

    def _rotate(self,
               image: np.ndarray,
               corners: List[Dict[str, int]],
               max_angle: float = 15) -> Tuple[np.ndarray, List[Dict[str, int]]]:
        """旋转"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        # 随机旋转角度
        angle = random.uniform(-max_angle, max_angle)

        # 旋转矩阵
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # 应用旋转
        rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # 更新角点坐标
        new_corners = []
        for c in corners:
            pt = np.array([c['X'], c['Y'], 1])
            rotated_pt = M.dot(pt)
            new_corners.append({'X': int(rotated_pt[0]), 'Y': int(rotated_pt[1])})

        return rotated, new_corners

    def _scale(self,
              image: np.ndarray,
              corners: List[Dict[str, int]],
              scale_range: Tuple[float, float] = (0.8, 1.2)) -> Tuple[np.ndarray, List[Dict[str, int]]]:
        """缩放"""
        h, w = image.shape[:2]

        # 随机缩放因子
        scale = random.uniform(*scale_range)

        # 缩放图片
        scaled = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        # 裁剪或填充到原尺寸
        if scale > 1.0:
            # 裁剪中心区域
            start_x = (scaled.shape[1] - w) // 2
            start_y = (scaled.shape[0] - h) // 2
            scaled = scaled[start_y:start_y + h, start_x:start_x + w]
            offset_x, offset_y = -start_x, -start_y
        else:
            # 填充黑色边框
            pad_x = (w - scaled.shape[1]) // 2
            pad_y = (h - scaled.shape[0]) // 2
            scaled = cv2.copyMakeBorder(scaled, pad_y, h - scaled.shape[0] - pad_y,
                                       pad_x, w - scaled.shape[1] - pad_x,
                                       cv2.BORDER_CONSTANT, value=(0, 0, 0))
            offset_x, offset_y = pad_x, pad_y

        # 更新角点坐标
        new_corners = []
        for c in corners:
            new_x = int(c['X'] * scale + offset_x)
            new_y = int(c['Y'] * scale + offset_y)
            new_corners.append({'X': new_x, 'Y': new_y})

        return scaled, new_corners

    def _flip(self,
             image: np.ndarray,
             corners: List[Dict[str, int]]) -> Tuple[np.ndarray, List[Dict[str, int]]]:
        """翻转"""
        h, w = image.shape[:2]

        # 随机选择水平或垂直翻转
        if random.random() < 0.5:
            # 水平翻转
            flipped = cv2.flip(image, 1)
            new_corners = [{'X': w - c['X'], 'Y': c['Y']} for c in corners]
            # 调整角点顺序（左右交换）
            if len(new_corners) == 4:
                new_corners = [new_corners[1], new_corners[0], new_corners[3], new_corners[2]]
        else:
            # 垂直翻转
            flipped = cv2.flip(image, 0)
            new_corners = [{'X': c['X'], 'Y': h - c['Y']} for c in corners]
            # 调整角点顺序（上下交换）
            if len(new_corners) == 4:
                new_corners = [new_corners[3], new_corners[2], new_corners[1], new_corners[0]]

        return flipped, new_corners

    # ============ 光学增强 ============

    def _adjust_brightness(self,
                         image: np.ndarray,
                         delta: int = 30) -> np.ndarray:
        """调整亮度"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + delta, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _adjust_contrast(self,
                        image: np.ndarray,
                        factor: float = 1.2) -> np.ndarray:
        """调整对比度"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = np.clip(l.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    def _add_gaussian_noise(self,
                           image: np.ndarray,
                           mean: int = 0,
                           std: int = 15) -> np.ndarray:
        """添加高斯噪声"""
        noise = self.rng.normal(mean, std, image.shape).astype(np.int16)
        return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    def _add_motion_blur(self,
                        image: np.ndarray,
                        kernel_size: int = 5) -> np.ndarray:
        """添加运动模糊"""
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel /= kernel_size
        return cv2.filter2D(image, -1, kernel)

    # ============ 遮挡模拟 ============

    def _random_occlusion(self,
                         image: np.ndarray,
                         corners: List[Dict[str, int]]) -> Tuple[np.ndarray, List[Dict[str, int]]]:
        """随机遮挡（在边框和角落区域添加遮挡）"""
        h, w = image.shape[:2]

        if len(corners) != 4:
            return image, corners

        # 将角点转换为 numpy 数组
        pts = np.array([[c['X'], c['Y']] for c in corners], dtype=np.int32)

        # 遮挡类型：角落遮挡 或 边框遮挡
        occlusion_type = random.choice(['corner', 'edge', 'both'])

        if occlusion_type in ['corner', 'both']:
            # 随机选择 1-2 个角落进行遮挡
            num_corners = random.randint(1, 2)
            corner_indices = random.sample(range(4), num_corners)

            for idx in corner_indices:
                c = corners[idx]

                # 遮挡尺寸（角落处较大）
                occlusion_size = random.randint(80, 200)

                # 遮挡位置（角落区域，偏向边框外侧）
                offset_x = random.randint(-occlusion_size // 2, occlusion_size // 4)
                offset_y = random.randint(-occlusion_size // 2, occlusion_size // 4)

                x1 = max(0, c['X'] + offset_x - occlusion_size // 2)
                y1 = max(0, c['Y'] + offset_y - occlusion_size // 2)
                x2 = min(w, c['X'] + offset_x + occlusion_size // 2)
                y2 = min(h, c['Y'] + offset_y + occlusion_size // 2)

                # 绘制黑色或灰色方块（模拟不同遮挡物）
                gray_value = random.randint(0, 80)
                cv2.rectangle(image, (x1, y1), (x2, y2), (gray_value, gray_value, gray_value), -1)

        if occlusion_type in ['edge', 'both']:
            # 随机选择一条边进行遮挡
            edge_idx = random.randint(0, 3)
            p1 = pts[edge_idx]
            p2 = pts[(edge_idx + 1) % 4]

            # 计算边框向量和垂直向量
            edge_vector = p2 - p1
            edge_length = np.linalg.norm(edge_vector)
            edge_unit = edge_vector / (edge_length + 1e-6)

            # 垂直向量（向外）
            perp_vector = np.array([-edge_unit[1], edge_unit[0]])

            # 遮挡位置：边框的随机一段
            occlusion_ratio = random.uniform(0.2, 0.8)  # 遮挡边框的 20%-80%
            occlusion_length = int(edge_length * occlusion_ratio)

            # 遮挡起始点（在边框上的随机位置）
            start_ratio = random.uniform(0, 1 - occlusion_ratio)
            start_point = p1 + edge_vector * start_ratio

            # 遮挡宽度
            occlusion_width = random.randint(30, 80)

            # 生成遮挡区域的多边形
            occlusion_points = np.array([
                start_point + perp_vector * occlusion_width,
                start_point + edge_unit * occlusion_length + perp_vector * occlusion_width,
                start_point + edge_unit * occlusion_length - perp_vector * (occlusion_width // 2),
                start_point - perp_vector * (occlusion_width // 2)
            ], dtype=np.int32)

            # 绘制遮挡多边形
            gray_value = random.randint(0, 100)
            cv2.fillPoly(image, [occlusion_points], (gray_value, gray_value, gray_value))

        return image, corners

    def _add_line_interference(self,
                              image: np.ndarray,
                              corners: List[Dict[str, int]]) -> Tuple[np.ndarray, List[Dict[str, int]]]:
        """添加线条干扰（只在边框区域附近，模拟 PPT 内容）"""
        h, w = image.shape[:2]

        if len(corners) != 4:
            return image, corners

        # 将角点转换为 numpy 数组
        pts = np.array([[c['X'], c['Y']] for c in corners], dtype=np.int32)

        # 随机添加 3-8 条线（主要在边框附近）
        num_lines = random.randint(3, 8)

        for _ in range(num_lines):
            # 选择边框区域附近的点
            if random.random() < 0.7:
                # 70% 的概率在边框附近生成线条
                edge_idx = random.randint(0, 3)
                p1 = pts[edge_idx]
                p2 = pts[(edge_idx + 1) % 4]

                # 在边框上随机选点
                t = random.random()
                base_point = p1 + (p2 - p1) * t

                # 从边框向外延伸
                edge_vector = p2 - p1
                edge_length = np.linalg.norm(edge_vector)
                edge_unit = edge_vector / (edge_length + 1e-6)
                perp_vector = np.array([-edge_unit[1], edge_unit[0]])

                # 线段起点和终点（从边框向内或向外延伸）
                direction = 1 if random.random() < 0.5 else -1
                line_length = random.randint(50, 200)

                x1 = int(base_point[0])
                y1 = int(base_point[1])
                x2 = int(base_point[0] + perp_vector[0] * line_length * direction)
                y2 = int(base_point[1] + perp_vector[1] * line_length * direction)
            else:
                # 30% 的概率在图片内任意位置（但靠近边框）
                edge_idx = random.randint(0, 3)
                p1 = pts[edge_idx]
                p2 = pts[(edge_idx + 1) % 4]

                # 在边框附近随机选点
                t = random.random()
                base_point = p1 + (p2 - p1) * t

                # 稍微向内偏移
                edge_vector = p2 - p1
                edge_unit = edge_vector / (np.linalg.norm(edge_vector) + 1e-6)
                perp_vector = np.array([-edge_unit[1], edge_unit[0]])

                offset = random.randint(-100, 100)
                x1 = int(base_point[0] + perp_vector[0] * offset)
                y1 = int(base_point[1] + perp_vector[1] * offset)

                # 线段方向大致平行于边框
                line_length = random.randint(50, 150)
                x2 = int(x1 + edge_unit[0] * line_length * (1 if random.random() < 0.5 else -1))
                y2 = int(y1 + edge_unit[1] * line_length * (1 if random.random() < 0.5 else -1))

            # 随机颜色和粗细
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            thickness = random.randint(1, 3)

            # 确保坐标在图片范围内
            x1, y1 = max(0, min(w-1, x1)), max(0, min(h-1, y1))
            x2, y2 = max(0, min(w-1, x2)), max(0, min(h-1, y2))

            cv2.line(image, (x1, y1), (x2, y2), color, thickness)

        return image, corners


def generate_training_data(data_root: str,
                          output_root: str,
                          augment_factor: int = 200,
                          train_ratio: float = 0.9):
    """
    生成训练数据集

    Args:
        data_root: 原始数据根目录
        output_root: 输出目录
        augment_factor: 每个样本的增强倍数
        train_ratio: 训练集比例
    """
    data_root = Path(data_root)
    output_root = Path(output_root)

    # 创建输出目录
    train_img_dir = output_root / 'images' / 'train'
    train_lbl_dir = output_root / 'labels' / 'train'
    val_img_dir = output_root / 'images' / 'val'
    val_lbl_dir = output_root / 'labels' / 'val'

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 扫描所有标注文件
    print(f"扫描数据目录: {data_root}")
    json_files = list(data_root.rglob('*.json'))
    print(f"找到 {len(json_files)} 个标注文件")

    # 过滤有效标注（有 ScreenCorners）
    valid_samples = []
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 检查是否有幕布边缘标注（优先使用 ScreenCorners）
            if 'ScreenCorners' in data and data['ScreenCorners'] is not None:
                # 查找对应的图片文件
                img_file = json_file.with_suffix('.jpg')
                if not img_file.exists():
                    img_file = json_file.with_suffix('.jpeg')

                if img_file.exists():
                    valid_samples.append((str(img_file), data['ScreenCorners']))
        except Exception as e:
            print(f"警告: 跳过 {json_file} - {e}")

    print(f"有效样本数: {len(valid_samples)}")

    # 划分训练集和验证集
    random.shuffle(valid_samples)
    split_idx = int(len(valid_samples) * train_ratio)
    train_samples = valid_samples[:split_idx]
    val_samples = valid_samples[split_idx:]

    print(f"训练集: {len(train_samples)} 样本")
    print(f"验证集: {len(val_samples)} 样本")

    # 初始化增强器
    augmentor = DataAugmentor(augment_factor)

    # 生成训练集
    print(f"\n生成训练集数据...")
    generate_samples(train_samples, train_img_dir, train_lbl_dir, augmentor, 'train')

    # 生成验证集（验证集不需要增强）
    print(f"\n生成验证集数据...")
    generate_samples(val_samples, val_img_dir, val_lbl_dir, None, 'val')

    print(f"\n完成! 输出目录: {output_root}")


def generate_samples(samples: List[Tuple[str, List[Dict[str, int]]]],
                    img_dir: Path,
                    lbl_dir: Path,
                    augmentor: Optional[DataAugmentor],
                    split_name: str):
    """生成样本"""
    total_samples = len(samples) * (augmentor.augment_factor if augmentor else 1)

    for idx, (img_path, corners) in enumerate(samples):
        image = cv2.imread(img_path)
        if image is None:
            print(f"警告: 无法读取图片 {img_path}")
            continue

        base_name = Path(img_path).stem

        # 如果是训练集，进行数据增强
        if augmentor:
            for aug_id in range(augmentor.augment_factor):
                aug_image, aug_corners = augmentor.augment_image(image.copy(), corners, aug_id)

                # 生成文件名
                aug_name = f"{base_name}_aug_{aug_id:04d}"

                # 保存图片
                cv2.imwrite(str(img_dir / f"{aug_name}.jpg"), aug_image, [cv2.IMWRITE_JPEG_QUALITY, 95])

                # 保存标签（格式: x1,y1,x2,y2,x3,y3,x4,y4）
                label_str = ','.join([f"{c['X']},{c['Y']}" for c in aug_corners])
                with open(lbl_dir / f"{aug_name}.txt", 'w') as f:
                    f.write(label_str)

                if (idx + 1) % 10 == 0:
                    progress = (idx * augmentor.augment_factor + aug_id + 1) / total_samples * 100
                    print(f"  {split_name.capitalize()} 进度: {progress:.1f}%")
        else:
            # 验证集直接复制
            cv2.imwrite(str(img_dir / f"{base_name}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            label_str = ','.join([f"{c['X']},{c['Y']}" for c in corners])
            with open(lbl_dir / f"{base_name}.txt", 'w') as f:
                f.write(label_str)

            if (idx + 1) % 10 == 0:
                progress = (idx + 1) / len(samples) * 100
                print(f"  {split_name.capitalize()} 进度: {progress:.1f}%")


if __name__ == '__main__':
    import sys

    random.seed(42)
    np.random.seed(42)

    # 配置
    data_root = r'D:\Programing\C#\MauiScan\AnnotationTool\data'
    output_root = r'D:\Programing\C#\MauiScan\AnnotationTool\training_data'

    # 从命令行参数读取增强倍数，默认 200
    augment_factor = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    print(f"增强倍数: {augment_factor}")

    # 生成训练数据
    generate_training_data(
        data_root=data_root,
        output_root=output_root,
        augment_factor=augment_factor,
        train_ratio=0.9
    )
