"""
从V2 checkpoint继续训练 - 使用更低的学习率
"""

import sys
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import json

sys.path.append(str(Path(__file__).parent / 'models'))
from corner_detector_v2 import PPTCornerDetectorV2, CornerDetectionLossV2


class SmallDataset:
    def __init__(self, data_dir, input_size=512):
        self.data_dir = Path(data_dir)
        self.input_size = input_size
        self.samples = []

        for img_path in self.data_dir.glob("*.png"):
            json_path = img_path.with_suffix('.json')
            if json_path.exists():
                self.samples.append((img_path, json_path))

        for img_path in self.data_dir.glob("*.jpg"):
            json_path = img_path.with_suffix('.json')
            if json_path.exists():
                self.samples.append((img_path, json_path))

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, json_path = self.samples[idx]

        image = Image.open(img_path).convert('RGB')
        original_width, original_height = image.size

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        corners = data['Corners']
        corners_array = np.array([[c['X'], c['Y']] for c in corners], dtype=np.float32)

        corners_norm = corners_array.copy()
        corners_norm[:, 0] /= original_width
        corners_norm[:, 1] /= original_height

        image = image.resize((self.input_size, self.input_size), Image.BILINEAR)

        image_tensor = self.to_tensor(image)
        corners_tensor = torch.from_numpy(corners_norm).flatten().float()

        return image_tensor, corners_tensor, str(img_path.name)


print("="*60)
print("  从V2 checkpoint继续训练（低LR）")
print("="*60)

# 加载checkpoint
checkpoint_path = Path("overfit_results_v2/best_overfit_model.pth")
if not checkpoint_path.exists():
    print(f"错误：找不到checkpoint: {checkpoint_path}")
    exit(1)

checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
print(f"加载checkpoint: epoch {checkpoint['epoch']}, error {checkpoint['pixel_error']:.2f}px")

# 加载数据
dataset = SmallDataset("../AnnotationTool/data/test", input_size=512)
dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

# 创建模型并加载权重
device = torch.device('cpu')
model = PPTCornerDetectorV2(pretrained=False).to(device)
model.load_state_dict(checkpoint['model_state_dict'])

# 使用更小的学习率
criterion = CornerDetectionLossV2(coord_weight=1.0, order_weight=0.0)
optimizer = optim.Adam(model.parameters(), lr=0.00001)  # ⭐ 10倍更小

print(f"新学习率: 0.00001 (原来是0.001的1/100)")
print("开始fine-tuning...")
print("-"*60)

best_error = checkpoint['pixel_error']
output_dir = Path("overfit_results_v2_resume")
output_dir.mkdir(exist_ok=True)

for epoch in range(1, 1001):
    model.train()

    for images, targets, filenames in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        pred_coords, pred_conf = model(images)
        losses = criterion(pred_coords, pred_conf, targets)
        loss = losses['total_loss']

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 计算误差
        pred_coords_np = pred_coords.detach().cpu().numpy()
        targets_np = targets.cpu().numpy()

        pred_pixels = pred_coords_np * 512
        target_pixels = targets_np * 512

        errors = []
        for i in range(len(images)):
            pred_pts = pred_pixels[i].reshape(4, 2)
            target_pts = target_pixels[i].reshape(4, 2)
            errs = np.linalg.norm(pred_pts - target_pts, axis=1)
            errors.append(np.mean(errs))

        pixel_error = np.mean(errors)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f} | Error: {pixel_error:.3f}px")

        if pixel_error < best_error:
            best_error = pixel_error
            torch.save({
                'epoch': checkpoint['epoch'] + epoch,
                'model_state_dict': model.state_dict(),
                'pixel_error': pixel_error,
            }, output_dir / 'best_model.pth')
            print(f"  -> 新最佳: {best_error:.2f}px")

        if pixel_error < 1.5:
            print(f"\n[SUCCESS] 达到 {pixel_error:.2f}px!")
            break

print(f"\n最终最佳误差: {best_error:.2f}px")
print(f"模型保存到: {output_dir / 'best_model.pth'}")
