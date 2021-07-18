import torch
from models.unet_adaptive_bins import UnetAdaptiveBins


model = UnetAdaptiveBins.build(256)
x = torch.randn(2, 3, 480, 640)
model(x)
