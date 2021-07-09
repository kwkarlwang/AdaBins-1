import torch
import torch.nn as nn
import torch.nn.functional as F

from .miniViT import mViT


class UpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSampleBN, self).__init__()

        self._net = nn.Sequential(
            nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_features),
            nn.LeakyReLU(),
            nn.Conv2d(
                output_features, output_features, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(output_features),
            nn.LeakyReLU(),
        )

    def forward(self, x, concat_with):
        up_x = F.interpolate(
            x,
            size=[concat_with.size(2), concat_with.size(3)],
            mode="bilinear",
            align_corners=True,
        )
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)


class ConvBN(nn.Module):
    def __init__(self, input_features, output_features):
        super(ConvBN, self).__init__()

        self._net = nn.Sequential(
            nn.Conv2d(
                input_features, output_features, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(output_features),
            nn.LeakyReLU(),
            nn.Conv2d(
                output_features, output_features, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(output_features),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self._net(x)


class DecoderBN(nn.Module):
    def __init__(
        self, num_features=2048, num_classes=1, bottleneck_features=2048, seg_classes=41
    ):
        super(DecoderBN, self).__init__()
        features = int(num_features)

        self.conv2 = nn.Conv2d(
            bottleneck_features, features, kernel_size=1, stride=1, padding=1
        )

        # for depth
        self.depth_up1 = UpSampleBN(
            skip_input=features // 1 + 112 + 64, output_features=features // 2
        )
        self.depth_up2 = UpSampleBN(
            skip_input=features // 2 + 40 + 24 + 128, output_features=features // 4
        )
        self.depth_up3 = UpSampleBN(
            skip_input=features // 4 + 24 + 16 + 64, output_features=features // 8
        )
        self.depth_up4 = UpSampleBN(
            skip_input=features // 8 + 16 + 8 + 64, output_features=features // 16
        )
        self.depth_conv3 = nn.Conv2d(
            features // 16 + 64, num_classes, kernel_size=3, stride=1, padding=1
        )

        self.depth_to_seg_up1 = ConvBN(features // 2, 128)
        self.depth_to_seg_up2 = ConvBN(features // 4, 64)
        self.depth_to_seg_up3 = ConvBN(features // 8, 64)
        self.depth_to_seg_up4 = ConvBN(features // 16, 64)

        self.seg_up1 = UpSampleBN(
            skip_input=features // 1 + 112 + 64, output_features=features // 2
        )
        self.seg_up2 = UpSampleBN(
            skip_input=features // 2 + 40 + 24 + 128, output_features=features // 4
        )
        self.seg_up3 = UpSampleBN(
            skip_input=features // 4 + 24 + 16 + 64, output_features=features // 8
        )
        self.seg_up4 = UpSampleBN(
            skip_input=features // 8 + 16 + 8 + 64, output_features=features // 16
        )

        self.seg_conv3 = nn.Conv2d(
            features // 16 + 64, seg_classes, kernel_size=3, stride=1, padding=1
        )

        self.seg_to_depth_up1 = ConvBN(features // 2, 128)
        self.seg_to_depth_up2 = ConvBN(features // 4, 64)
        self.seg_to_depth_up3 = ConvBN(features // 8, 64)
        self.seg_to_depth_up4 = ConvBN(features // 16, 64)

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = (
            features[4],
            features[5],
            features[6],
            features[8],
            features[11],
        )

        x_d0 = self.conv2(x_block4)  # x_d0: 2048

        x_d1 = self.depth_up1(x_d0, x_block3)
        x_s1 = self.seg_up1(x_d0, x_block3)
        x_s1_prime = self.seg_to_depth_up1(x_s1)
        x_d1_prime = self.depth_to_seg_up1(x_d1)

        x_d2 = self.depth_up2(torch.cat([x_d1, x_s1_prime], dim=1), x_block2)
        x_s2 = self.seg_up2(torch.cat([x_s1, x_d1_prime], dim=1), x_block2)
        x_s2_prime = self.seg_to_depth_up2(x_s2)
        x_d2_prime = self.depth_to_seg_up2(x_d2)

        x_d3 = self.depth_up3(torch.cat([x_d2, x_s2_prime], dim=1), x_block1)
        x_s3 = self.seg_up3(torch.cat([x_s2, x_d2_prime], dim=1), x_block1)
        x_s3_prime = self.seg_to_depth_up3(x_s3)
        x_d3_prime = self.depth_to_seg_up3(x_d3)

        x_d4 = self.depth_up4(torch.cat([x_d3, x_s3_prime], dim=1), x_block0)
        x_s4 = self.seg_up4(torch.cat([x_s3, x_d3_prime], dim=1), x_block0)
        x_s4_prime = self.seg_to_depth_up4(x_s4)
        x_d4_prime = self.depth_to_seg_up4(x_d4)

        depth_out = self.depth_conv3(torch.cat([x_d4, x_s4_prime], dim=1))
        seg_out = self.seg_conv3(torch.cat([x_s4, x_d4_prime], dim=1))
        return depth_out, seg_out


class Encoder(nn.Module):
    def __init__(self, backend):
        super(Encoder, self).__init__()
        self.original_model = backend

    def forward(self, x):
        features = [x]
        for k, v in self.original_model._modules.items():
            if k == "blocks":
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))
        return features


class UnetAdaptiveBins(nn.Module):
    def __init__(
        self,
        backend,
        n_bins=100,
        min_val=0.1,
        max_val=10,
        norm="linear",
        seg_classes=41,
    ):
        super(UnetAdaptiveBins, self).__init__()
        self.num_classes = n_bins
        self.min_val = min_val
        self.max_val = max_val
        self.encoder = Encoder(backend)
        self.adaptive_bins_layer = mViT(
            128,
            n_query_channels=128,
            patch_size=16,
            dim_out=n_bins,
            embedding_dim=128,
            norm=norm,
        )

        self.decoder = DecoderBN(num_classes=128, seg_classes=seg_classes)
        self.conv_out = nn.Sequential(
            nn.Conv2d(128, n_bins, kernel_size=1, stride=1, padding=0),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        unet_out, seg_out = self.decoder(features=self.encoder(x))
        bin_widths_normed, range_attention_maps = self.adaptive_bins_layer(unet_out)
        out = self.conv_out(range_attention_maps)

        # Post process
        # n, c, h, w = out.shape
        # hist = torch.sum(out.view(n, c, h * w), dim=2) / (h * w)  # not used for training

        bin_widths = (
            self.max_val - self.min_val
        ) * bin_widths_normed  # .shape = N, dim_out
        bin_widths = nn.functional.pad(
            bin_widths, (1, 0), mode="constant", value=self.min_val
        )
        bin_edges = torch.cumsum(bin_widths, dim=1)

        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        n, dout = centers.size()
        centers = centers.view(n, dout, 1, 1)

        pred = torch.sum(out * centers, dim=1, keepdim=True)
        return bin_edges, pred, seg_out

    def freeze_seg(self):
        d = self.decoder
        freeze_list = [
            d.seg_conv3,
            d.seg_to_depth_up1,
            d.seg_to_depth_up2,
            d.seg_to_depth_up3,
            d.seg_to_depth_up4,
            d.seg_up1,
            d.seg_up2,
            d.seg_up3,
            d.seg_up4,
        ]
        for m in freeze_list:
            for p in m.parameters():
                p.requires_grad = False

    def unfreeze_seg(self):
        d = self.decoder
        freeze_list = [
            d.seg_conv3,
            d.seg_to_depth_up1,
            d.seg_to_depth_up2,
            d.seg_to_depth_up3,
            d.seg_to_depth_up4,
            d.seg_up1,
            d.seg_up2,
            d.seg_up3,
            d.seg_up4,
        ]
        for m in freeze_list:
            for p in m.parameters():
                p.requires_grad = True

    def get_1x_lr_params(self):  # lr/10 learning rate
        return self.encoder.parameters()

    def get_10x_lr_params(self):  # lr learning rate
        modules = [self.decoder, self.adaptive_bins_layer, self.conv_out]
        for m in modules:
            yield from m.parameters()

    @classmethod
    def build(cls, n_bins, **kwargs):
        basemodel_name = "tf_efficientnet_b5_ap"

        print("Loading base model ()...".format(basemodel_name), end="")
        basemodel = torch.hub.load(
            "rwightman/gen-efficientnet-pytorch", basemodel_name, pretrained=True
        )
        print("Done.")

        # Remove last layer
        print("Removing last two layers (global_pool & classifier).")
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()

        # Building Encoder-Decoder model
        print("Building Encoder-Decoder model..", end="")
        m = cls(basemodel, n_bins=n_bins, **kwargs)
        print("Done.")
        return m
