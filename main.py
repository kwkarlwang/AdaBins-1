import torch
from models.unet_adaptive_bins import UnetAdaptiveBins

if __name__ == "__main__":
    model = UnetAdaptiveBins.build(100)
    x = torch.randn(2, 4, 640, 480)
    _, pred = model(x)
    print(pred.shape)
