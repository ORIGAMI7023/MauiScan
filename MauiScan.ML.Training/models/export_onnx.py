"""
导出 PyTorch 模型到 ONNX 格式
"""

import torch
import onnx
from pathlib import Path
from corner_detector import PPTCornerDetector


def export_to_onnx(
    model_path: str,
    output_path: str,
    input_size: tuple[int, int] = (512, 512),
    opset_version: int = 14
):
    """
    将 PyTorch 模型导出为 ONNX

    Args:
        model_path: PyTorch 模型权重文件路径 (.pth)
        output_path: 输出 ONNX 文件路径
        input_size: 输入图片尺寸 (H, W)
        opset_version: ONNX opset 版本
    """
    print(f"[Export] Starting ONNX export...")
    print(f"  - PyTorch model: {model_path}")
    print(f"  - Output path: {output_path}")
    print(f"  - Input size: {input_size}")

    # 1. 加载 PyTorch 模型
    model = PPTCornerDetector(pretrained=False)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    # 兼容不同的 checkpoint 格式
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # 2. 创建虚拟输入
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1])

    # 3. 导出 ONNX（使用旧版API避免编码问题）
    import warnings
    warnings.filterwarnings('ignore')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['coordinates', 'confidence'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'coordinates': {0: 'batch_size'},
                'confidence': {0: 'batch_size'}
            },
            dynamo=False  # 使用旧版导出器
        )

    print(f"[OK] ONNX model exported successfully!")

    # 4. 验证 ONNX 模型
    print(f"\n[Validate] Checking ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"[OK] ONNX model validation passed!")

    # 5. 显示模型信息
    import os
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n[Info] Model information:")
    print(f"  - File size: {file_size_mb:.2f} MB")
    print(f"  - Opset version: {opset_version}")
    print(f"  - Input: input [1, 3, {input_size[0]}, {input_size[1]}]")
    print(f"  - Output 1: coordinates [1, 8]")
    print(f"  - Output 2: confidence [1, 1]")

    # 6. 测试 ONNX Runtime 推理
    print(f"\n[Test] Testing ONNX Runtime inference...")
    import onnxruntime as ort
    import numpy as np

    session = ort.InferenceSession(output_path)
    test_input = np.random.randn(1, 3, input_size[0], input_size[1]).astype(np.float32)

    outputs = session.run(None, {'input': test_input})
    coords, conf = outputs

    print(f"[OK] ONNX Runtime inference successful!")
    print(f"  - Coordinates output: {coords.shape}, range [{coords.min():.3f}, {coords.max():.3f}]")
    print(f"  - Confidence output: {conf.shape}, value {conf[0, 0]:.3f}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='导出 ONNX 模型')
    parser.add_argument('model_path', type=str, help='PyTorch 模型路径 (.pth)')
    parser.add_argument('--output', type=str, default='ppt_corner_detector.onnx',
                        help='输出 ONNX 文件名')
    parser.add_argument('--input-size', type=int, default=512, help='输入图片尺寸')
    parser.add_argument('--opset', type=int, default=14, help='ONNX opset 版本')

    args = parser.parse_args()

    export_to_onnx(
        model_path=args.model_path,
        output_path=args.output,
        input_size=(args.input_size, args.input_size),
        opset_version=args.opset
    )

    print(f"\n🎉 完成! ONNX 模型已保存到: {args.output}")
    print(f"\n📌 下一步:")
    print(f"  1. 将 {args.output} 复制到 C# 项目:")
    print(f"     MauiScan/MauiScan/Resources/Raw/{args.output}")
    print(f"  2. 在 C# 中使用 OnnxInferenceService 加载模型")


if __name__ == '__main__':
    main()
