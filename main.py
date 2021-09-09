import torch
from models.unet_adaptive_bins import UnetAdaptiveBins

if __name__ == "__main__":
    model = UnetAdaptiveBins.build(100)
    x = torch.randn((2, 4, 480, 640))
    _, res = model(x)
    print(res.shape)
