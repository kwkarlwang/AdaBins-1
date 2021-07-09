# %%

from loss import BinsChamferLoss, SILogLoss
import torch
from models import UnetAdaptiveBins

model = UnetAdaptiveBins.build(n_bins=256, seg_classes=41)

# %%
model.freeze_seg()

# %%
for p in model.decoder.seg_up1.parameters():
    print(p.requires_grad)

# %%
model.unfreeze_seg()
# %%

# img = torch.randn(2, 3, 480, 640)
# depth = torch.randn(2, 480, 640, 1)
# optimizer = torch.optim.AdamW(model.parameters())

# has_seg = False
# criterion_ueff = SILogLoss()
# criterion_bins = BinsChamferLoss()
# ###########################################################

# optimizer.zero_grad()

# if has_seg:
#     bin_edges, pred, seg_out = model(img)
# else:
#     bin_edges, pred = model(img)

# l_dense = criterion_ueff(pred, depth, interpolate=True)

# l_chamfer = criterion_bins(bin_edges, depth)

# loss = l_dense + l_chamfer
# seg_loss = 0
# if has_seg:
#     seg = batch["seg"].to(torch.long).to(device)
#     seg = seg.squeeze()
#     seg_out = nn.functional.interpolate(seg_out, seg.shape[-2:], mode="nearest")
#     seg_loss = seg_criterion(seg_out, seg)
#     loss += args.w_seg * seg_loss

# loss.backward()
# nn.utils.clip_grad_norm_(model.parameters(), 0.1)  # optional
# optimizer.step()
# if should_log and step % 5 == 0:
#     wandb.log({f"Train/{criterion_ueff.name}": l_dense.item()}, step=step)
#     wandb.log({f"Train/{criterion_bins.name}": l_chamfer.item()}, step=step)
# if should_log and has_seg:
#     wandb.log({f"Train/CrossEntropyLoss": l_chamfer.item()}, step=step)

# step += 1
# scheduler.step()
