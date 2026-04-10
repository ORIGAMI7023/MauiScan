"""
单应性矩阵修正测试

原理：
1. 用模型检测到的 4 个角点，计算单应性矩阵 H
2. 通过 SVD 分解 H，求出最优矩形（比例未知）
3. 用修正后的 H 反投影，得到几何约束下的 4 个角点
4. 确保对边平行、邻边垂直，但不约束长宽比
"""

import numpy as np
import cv2
import json
from pathlib import Path
from tqdm import tqdm


def refine_corners_homography(corners, image_size=None, iterations=10):
    """
    用单应性矩阵修正角点（不假设固定比例）

    Args:
        corners: 4个角点，顺序 [左上, 右上, 右下, 左下]，shape (4, 2)
        image_size: (w, h) 图片尺寸，用于约束点在图片范围内
        iterations: SVD 优化迭代次数

    Returns:
        refined_corners: 修正后的角点 (4, 2)
    """
    corners = np.array(corners, dtype=np.float32)

    # 标准矩形的 4 个角点（单位矩形）
    # [左上, 右上, 右下, 左下]
    dst_pts = np.array([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1]
    ], dtype=np.float32)

    # 计算初始单应性矩阵：从单位矩形 -> 检测到的四边形
    H = cv2.getPerspectiveTransform(dst_pts, corners)

    # 用 SVD 分解单应性矩阵，找到最优的仿射+透视分解
    # H = K * [R | t]  (当场景是平面矩形时)
    # 我们通过迭代优化来修正

    # 方法：直接用 DLT 从检测点和目标矩形拟合最优 H
    # 目标：找到一个矩形 (w, h) 和 H，使得 H * [矩形角点] ≈ 检测点
    # 约束：矩形的 4 个角必须是 [0,0], [w,0], [w,h], [0,h]

    # 第一步：估算矩形的实际宽高
    # 用对角线的几何性质来估算

    # 计算四条边的长度
    top_edge = np.linalg.norm(corners[1] - corners[0])
    bottom_edge = np.linalg.norm(corners[2] - corners[3])
    left_edge = np.linalg.norm(corners[3] - corners[0])
    right_edge = np.linalg.norm(corners[2] - corners[1])

    # 平均宽高
    avg_width = (top_edge + bottom_edge) / 2
    avg_height = (left_edge + right_edge) / 2

    # 第二步：定义目标矩形（使用估算的宽高比）
    aspect = avg_width / (avg_height + 1e-6)

    # 第三步：迭代优化
    # 目标点：标准矩形的角点，宽为 aspect，高为 1
    rect_pts = np.array([
        [0, 0],
        [aspect, 0],
        [aspect, 1],
        [0, 1]
    ], dtype=np.float64)

    # 用 DLT 算法求解最优单应性矩阵
    # 使得 H * rect_pts[i] ≈ corners[i]

    # 构造 DLT 方程组：Ah = 0
    A = []
    for i in range(4):
        x, y = corners[i]
        X, Y = rect_pts[i]
        A.append([-X, -Y, -1, 0, 0, 0, x*X, x*Y, x])
        A.append([0, 0, 0, -X, -Y, -1, y*X, y*Y, y])

    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)

    # 第四步：用修正后的 H 反投影得到修正后的角点
    refined_corners = []
    for i in range(4):
        pt = np.array([rect_pts[i][0], rect_pts[i][1], 1])
        projected = H @ pt
        projected = projected[:2] / projected[2]
        refined_corners.append(projected)

    refined_corners = np.array(refined_corners)

    return refined_corners, aspect


def compute_geometry_score(corners):
    """
    计算四边形的几何质量分数

    评分标准：
    1. 对边长度比（越接近 1 越好）
    2. 对边平行度（越平行越好）
    3. 面积（不能太小）
    """
    corners = np.array(corners, dtype=np.float64)

    # 四条边
    edges = []
    for i in range(4):
        edge = corners[(i+1) % 4] - corners[i]
        edges.append(edge)

    # 1. 对边长度比
    top_len = np.linalg.norm(edges[0])
    bottom_len = np.linalg.norm(edges[2])
    left_len = np.linalg.norm(edges[3])
    right_len = np.linalg.norm(edges[1])

    length_ratio_h = min(top_len, bottom_len) / (max(top_len, bottom_len) + 1e-6)
    length_ratio_v = min(left_len, right_len) / (max(left_len, right_len) + 1e-6)
    length_score = (length_ratio_h + length_ratio_v) / 2

    # 2. 对边平行度（用向量夹角的余弦值）
    def cos_angle(v1, v2):
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6
        return abs(dot / norm)

    parallel_h = cos_angle(edges[0], edges[2])
    parallel_v = cos_angle(edges[3], edges[1])
    parallel_score = (parallel_h + parallel_v) / 2

    # 3. 面积（用交叉积）
    def polygon_area(pts):
        n = len(pts)
        area = 0
        for i in range(n):
            j = (i + 1) % n
            area += pts[i][0] * pts[j][1]
            area -= pts[j][0] * pts[i][1]
        return abs(area) / 2

    area = polygon_area(corners)

    # 综合分数
    score = length_score * 0.3 + parallel_score * 0.7

    return {
        'length_ratio_h': length_ratio_h,
        'length_ratio_v': length_ratio_v,
        'parallel_h': parallel_h,
        'parallel_v': parallel_v,
        'area': area,
        'score': score
    }


def test_homography_refinement():
    """测试单应性修正效果"""

    # 配置
    val_images_dir = r'D:\Programing\C#\MauiScan\AnnotationTool\training_data\images\val'
    val_labels_dir = r'D:\Programing\C#\MauiScan\AnnotationTool\training_data\labels\val'
    model_output_dir = r'D:\Programing\C#\MauiScan\AnnotationTool\visualization'
    output_dir = r'D:\Programing\C#\MauiScan\AnnotationTool\test_homography'

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载验证集标签
    label_files = sorted(list(Path(val_labels_dir).glob('*.txt')))
    print(f"验证集样本数: {len(label_files)}")

    # 统计
    original_errors = []
    refined_errors = []
    original_scores = []
    refined_scores = []
    aspect_ratios = []

    print("\n测试单应性修正...")
    for label_file in tqdm(label_files):
        # 读取标签（作为 GT）
        with open(label_file, 'r') as f:
            label_str = f.read().strip()
        coords = label_str.split(',')
        corners_gt = []
        for i in range(0, len(coords), 2):
            corners_gt.append([float(coords[i]), float(coords[i + 1])])
        corners_gt = np.array(corners_gt)

        # 模拟模型输出：给 GT 加随机噪声（模拟模型误差）
        np.random.seed(hash(label_file.stem) % (2**32))
        noise_level = 30  # 30 像素的噪声
        corners_noisy = corners_gt + np.random.randn(4, 2) * noise_level

        # 计算修正前的误差
        original_error = np.mean([np.linalg.norm(corners_gt[i] - corners_noisy[i]) for i in range(4)])
        original_score = compute_geometry_score(corners_noisy)

        # 单应性修正
        corners_refined, aspect = refine_corners_homography(corners_noisy)

        # 计算修正后的误差
        refined_error = np.mean([np.linalg.norm(corners_gt[i] - corners_refined[i]) for i in range(4)])
        refined_score = compute_geometry_score(corners_refined)

        original_errors.append(original_error)
        refined_errors.append(refined_error)
        original_scores.append(original_score['score'])
        refined_scores.append(refined_score['score'])
        aspect_ratios.append(aspect)

        # 可视化
        image_path = Path(val_images_dir) / f"{label_file.stem}.jpg"
        if not image_path.exists():
            continue
        image = cv2.imread(str(image_path))

        # 绘制对比图（三栏：GT | 修正前 | 修正后）
        h, w = image.shape[:2]

        # 缩小图片以适应三栏显示
        scale = 0.6
        sw, sh = int(w * scale), int(h * scale)
        img_small = cv2.resize(image, (sw, sh))

        canvas = np.zeros((sh + 50, sw * 3 + 4, 3), dtype=np.uint8)
        canvas[:sh, :sw] = img_small
        canvas[:sh, sw+2:sw*2+2] = img_small
        canvas[:sh, sw*2+4:] = img_small

        def draw_quad(panel_offset_x, corners_arr, color, thickness=2):
            for i in range(4):
                p1 = (int(corners_arr[i][0] * scale) + panel_offset_x,
                      int(corners_arr[i][1] * scale))
                p2 = (int(corners_arr[(i+1)%4][0] * scale) + panel_offset_x,
                      int(corners_arr[(i+1)%4][1] * scale))
                cv2.line(canvas, p1, p2, color, thickness)
            for pt in corners_arr:
                cx = int(pt[0] * scale) + panel_offset_x
                cy = int(pt[1] * scale)
                cv2.circle(canvas, (cx, cy), 6, color, -1)

        # 左栏：GT（绿色）
        draw_quad(0, corners_gt, (0, 255, 0))

        # 中栏：修正前，模型输出（橙色）
        draw_quad(sw + 2, corners_noisy, (0, 165, 255))

        # 右栏：修正后（红色）+ GT（绿色虚线对比）
        draw_quad(sw*2 + 4, corners_refined, (0, 0, 255))
        draw_quad(sw*2 + 4, corners_gt, (0, 255, 0), thickness=1)

        # 分隔线
        cv2.line(canvas, (sw, 0), (sw, sh), (200, 200, 200), 2)
        cv2.line(canvas, (sw*2+2, 0), (sw*2+2, sh), (200, 200, 200), 2)

        # 底部标签
        label_y = sh + 35
        fs = 0.55
        cv2.putText(canvas, "GT (Ground Truth)", (5, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 255, 0), 1)
        cv2.putText(canvas, f"Before err={original_error:.1f}px score={original_score['score']:.3f}",
                   (sw + 7, label_y), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 165, 255), 1)
        cv2.putText(canvas, f"After  err={refined_error:.1f}px score={refined_score['score']:.3f}",
                   (sw*2 + 9, label_y), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 255), 1)

        cv2.imwrite(str(output_dir / f"result_{label_file.stem}.jpg"), canvas)

    # 打印统计
    print("\n" + "="*60)
    print("单应性修正结果")
    print("="*60)
    print(f"平均宽高比: {np.mean(aspect_ratios):.3f} (±{np.std(aspect_ratios):.3f})")
    print()
    print(f"修正前平均误差: {np.mean(original_errors):.2f} px")
    print(f"修正后平均误差: {np.mean(refined_errors):.2f} px")
    print(f"误差变化: {np.mean(refined_errors) - np.mean(original_errors):+.2f} px")
    print()
    print(f"修正前几何分数: {np.mean(original_scores):.4f}")
    print(f"修正后几何分数: {np.mean(refined_scores):.4f}")
    print(f"分数变化: {np.mean(refined_scores) - np.mean(original_scores):+.4f}")
    print()

    improved = sum(1 for i in range(len(refined_scores)) if refined_scores[i] > original_scores[i])
    print(f"几何质量提升: {improved}/{len(refined_scores)} ({improved/len(refined_scores)*100:.1f}%)")

    print(f"\n结果已保存到: {output_dir}")
    print("="*60)

    return np.mean(refined_errors) < np.mean(original_errors)


if __name__ == '__main__':
    test_homography_refinement()
