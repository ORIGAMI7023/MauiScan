import torch

checkpoint = torch.load('overfit_results/best_overfit_model.pth', map_location='cpu')
print(f'Best epoch: {checkpoint["epoch"]}')
print(f'Best pixel error: {checkpoint["pixel_error"]:.2f}px')
