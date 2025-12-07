"""
预处理：批量生成数据增强
- 只增强图像（颜色、亮度、模糊、噪声）
- 不做几何变换（角点坐标保持不变）
- 生成到新目录，保留原始数据
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import json
from tqdm import tqdm
import shutil

sys.path.append(str(Path(__file__).parent / 'dataset'))
from prepare_data import AnnotationDataset


def augment_image(image, aug_type):
    """
    单次增强（不改变几何结构，角点坐标不变）

    aug_type: 增强类型编号
        1: 亮度+
        2: 亮度-
        3: 对比度+
        4: 对比度-
        5: 饱和度+
        6: 饱和度-
        7: 模糊
        8: 锐化
    """
    if aug_type == 1:  # 亮度+
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(1.3)

    elif aug_type == 2:  # 亮度-
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(0.7)

    elif aug_type == 3:  # 对比度+
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(1.3)

    elif aug_type == 4:  # 对比度-
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(0.7)

    elif aug_type == 5:  # 饱和度+
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(1.3)

    elif aug_type == 6:  # 饱和度-
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(0.7)

    elif aug_type == 7:  # 高斯模糊
        return image.filter(ImageFilter.GaussianBlur(radius=1.5))

    elif aug_type == 8:  # 锐化
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(1.5)

    else:
        return image


def main():
    import argparse

    parser = argparse.ArgumentParser(description='预处理数据增强')
    parser.add_argument('--data-root', type=str, default='../AnnotationTool/data',
                        help='原始数据根目录')
    parser.add_argument('--output-root', type=str, default='../AnnotationTool/data_augmented',
                        help='增强数据输出目录')
    parser.add_argument('--aug-per-image', type=int, default=3,
                        help='每张图生成几个增强版本')

    args = parser.parse_args()

    print("="*60)
    print("  数据增强预处理")
    print("="*60)
    print(f"原始数据: {args.data_root}")
    print(f"输出目录: {args.output_root}")
    print(f"每图增强: {args.aug_per_image} 个版本")
    print("="*60)

    # 加载原始数据
    print("\n[1/3] 加载原始数据...")
    dataset = AnnotationDataset(args.data_root)

    print(f"原始样本数: {len(dataset)}")
    print(f"增强后预计: {len(dataset) * (1 + args.aug_per_image)} 样本")

    # 创建输出目录
    output_root = Path(args.output_root)
    output_root.mkdir(exist_ok=True, parents=True)

    # 清空旧数据
    for item in output_root.iterdir():
        if item.is_file():
            item.unlink()

    print(f"\n[2/3] 生成增强数据...")

    total_count = 0
    aug_types = [1, 2, 3, 4, 5, 6, 7, 8]  # 8种增强类型

    for idx in tqdm(range(len(dataset)), desc="Processing"):
        sample = dataset.samples[idx]
        image_path = Path(sample['image_path'])
        corners = sample['corners']
        width = sample['width']
        height = sample['height']

        # 加载图片
        image = Image.open(image_path).convert('RGB')

        # 1. 保存原始图片
        orig_name = f"{total_count:04d}_orig{image_path.suffix}"
        orig_output_path = output_root / orig_name
        image.save(orig_output_path)

        # 保存标注（JSON）
        annotation = {
            'Corners': [
                {'X': float(corners[0, 0]), 'Y': float(corners[0, 1])},
                {'X': float(corners[1, 0]), 'Y': float(corners[1, 1])},
                {'X': float(corners[2, 0]), 'Y': float(corners[2, 1])},
                {'X': float(corners[3, 0]), 'Y': float(corners[3, 1])}
            ]
        }

        json_output_path = orig_output_path.with_suffix('.json')
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(annotation, f, indent=2)

        total_count += 1

        # 2. 生成增强版本
        # 随机选择 aug_per_image 种增强
        selected_augs = np.random.choice(aug_types, size=args.aug_per_image, replace=False)

        for aug_id in selected_augs:
            aug_image = augment_image(image, aug_id)

            aug_name = f"{total_count:04d}_aug{aug_id}{image_path.suffix}"
            aug_output_path = output_root / aug_name
            aug_image.save(aug_output_path)

            # ⭐ 角点坐标完全不变
            aug_json_path = aug_output_path.with_suffix('.json')
            with open(aug_json_path, 'w', encoding='utf-8') as f:
                json.dump(annotation, f, indent=2)

            total_count += 1

    print(f"\n[3/3] 完成！")
    print(f"总计生成: {total_count} 个样本")
    print(f"原始: {len(dataset)}")
    print(f"增强: {total_count - len(dataset)}")
    print(f"增倍: {total_count / len(dataset):.1f}x")
    print(f"输出目录: {output_root.absolute()}")
    print("="*60)

    print("\n下一步:")
    print(f"  python train_full_fixed.py --data-root {args.output_root} --epochs 300")


if __name__ == '__main__':
    main()
