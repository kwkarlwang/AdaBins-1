from models.unet_adaptive_bins import UnetAdaptiveBins

if __name__ == "__main__":
    model = UnetAdaptiveBins.build(100)
    print(model.encoder.original_model)
