"""分析过拟合测试的详细误差"""

import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import json

sys.path.append(str(Path(__file__).parent / 'models'))
from corner_detector import PPTCornerDetector

# 加载模型
print("加载模型...")
checkpoint = torch.load('overfit_results/best_overfit_model.pth', map_location='cpu', weights_only=False)
model = PPTCornerDetector(pretrained=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"最佳 Epoch: {checkpoint['epoch']}")
print(f"最佳像素误差: {checkpoint['pixel_error']:.2f}px\n")
print("="*80)

# 测试3张图片
test_dir = Path("../AnnotationTool/data/test")
for img_path in sorted(test_dir.glob("*.png")) + sorted(test_dir.glob("*.jpg")):
    json_path = img_path.with_suffix('.json')
    if not json_path.exists():
        continue

    print(f"\n图片: {img_path.name}")

    # 加载标注
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    corners = data['Corners']
    gt_corners = np.array([[c['X'], c['Y']] for c in corners], dtype=np.float32)

    # 加载图片
    image = Image.open(img_path).convert('RGB')
    orig_w, orig_h = image.size

    # 归一化GT坐标
    gt_norm = gt_corners.copy()
    gt_norm[:, 0] /= orig_w
    gt_norm[:, 1] /= orig_h

    # 缩放并预处理
    image_512 = image.resize((512, 512), Image.BILINEAR)
    img_array = np.array(image_512).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)

    # 推理
    with torch.no_grad():
        pred_coords, _ = model(img_tensor)

    pred_norm = pred_coords[0].numpy()
    pred_pts = pred_norm.reshape(4, 2)

    # 计算误差（在归一化坐标上）
    errors_norm = np.linalg.norm(pred_pts - gt_norm, axis=1)

    # 转换为像素误差（512x512）
    errors_512 = errors_norm * 512

    # 转换为原图像素误差
    errors_orig = errors_norm * np.array([orig_w, orig_h]).mean()

    print(f"  原图尺寸: {orig_w}x{orig_h}")
    print(f"  角点误差（512x512图）:")
    for i, (err_512, err_orig) in enumerate(zip(errors_512, errors_orig)):
        corner_name = ['左上', '右上', '右下', '左下'][i]
        print(f"    {corner_name}: {err_512:5.1f}px (512x512) = {err_orig:6.1f}px (原图)")

    avg_512 = np.mean(errors_512)
    avg_orig = np.mean(errors_orig)
    print(f"  平均误差: {avg_512:.1f}px (512x512) = {avg_orig:.1f}px (原图)")

    # 检查order loss的满足情况
    violations = []

    # 左上X < 右上X
    if pred_pts[0, 0] >= pred_pts[1, 0]:
        violations.append(f"左上X({pred_pts[0,0]:.3f}) >= 右上X({pred_pts[1,0]:.3f})")

    # 左下X < 右下X
    if pred_pts[3, 0] >= pred_pts[2, 0]:
        violations.append(f"左下X({pred_pts[3,0]:.3f}) >= 右下X({pred_pts[2,0]:.3f})")

    # 左上Y < 左下Y
    if pred_pts[0, 1] >= pred_pts[3, 1]:
        violations.append(f"左上Y({pred_pts[0,1]:.3f}) >= 左下Y({pred_pts[3,1]:.3f})")

    # 右上Y < 右下Y
    if pred_pts[1, 1] >= pred_pts[2, 1]:
        violations.append(f"右上Y({pred_pts[1,1]:.3f}) >= 右下Y({pred_pts[2,1]:.3f})")

    if violations:
        print(f"  [!] Order约束违反:")
        for v in violations:
            print(f"    - {v}")
    else:
        print(f"  [OK] Order约束满足")

    # 检查是否受到margin限制
    print(f"  预测坐标范围: X[{pred_pts[:,0].min():.3f}, {pred_pts[:,0].max():.3f}], Y[{pred_pts[:,1].min():.3f}, {pred_pts[:,1].max():.3f}]")
    print(f"  真实坐标范围: X[{gt_norm[:,0].min():.3f}, {gt_norm[:,0].max():.3f}], Y[{gt_norm[:,1].min():.3f}, {gt_norm[:,1].max():.3f}]")

print("\n" + "="*80)
print("分析完成!")
