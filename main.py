# %%
import torch
from utils import VP
from dataloader import DepthDataLoader
from argparse import Namespace

args = {
    "filenames_file_vp": "./train_test_inputs/train_vp.npy",
    "mat_file_path": "./nyu_depth_v2_labeled.mat",
    "mode": "train_vp",
    "distributed": False,
    "batch_size": 2,
    "num_threads": 2,
}
args = Namespace(**args)
# %%
DDL = DepthDataLoader(args, "train_vp")
vp = DDL.data
ds = DDL.training_samples.dataset  #type:ignore

# %%
data = next(iter(vp))


# %%
class VP:
    @staticmethod
    def sample_points(lines: torch.Tensor, num_points: int = 2):
        x1 = lines[:, 0:2]
        x2 = lines[:, 2:4]

        direction = x2 - x1
        t1 = torch.rand((num_points, len(lines), 1))
        t2 = torch.rand((num_points, len(lines), 1))
        start = torch.round(x1 + t1 * direction).to(torch.long)
        end = torch.round(x1 + t2 * direction).to(torch.long)
        return torch.dstack((start, end)).reshape(-1, 4)

    @staticmethod
    def calc_vp_loss(lines: torch.Tensor, Kinv: torch.Tensor,
                     depth_map: torch.Tensor, vd: torch.Tensor):
        x1, y1, x2, y2 = (lines[:, 0], lines[:, 1], lines[:, 2], lines[:, 3])
        depth1, depth2 = depth_map[y1, x1], depth_map[y2, x2]
        # 3xn
        u = Kinv @ torch.vstack((lines[:, 0:2].T, torch.ones(len(lines))))
        v = Kinv @ torch.vstack((lines[:, 2:4].T, torch.ones(len(lines))))

        # nx3
        u3d = (depth1 * u).T
        # nx3
        v3d = (depth2 * v).T
        loss = torch.norm(
            torch.cross((u3d - v3d),
                        vd[None, :].repeat((len(lines), 1)).to(torch.float32),
                        dim=1)).sum()
        return loss


device = 'cpu'
for idx in data['idx']:
    # print(idx)
    lines_set = ds[idx]['labelled_lines']

    Kinv = torch.tensor(ds.Kinv).to(torch.float32)
    vds = torch.tensor(ds[idx]['VDs'])
    for j, line in enumerate(lines_set):
        line = torch.tensor(line).to(device)
        sample_lines = VP.sample_points(line, 4)
        depth_map = torch.randn((480, 640))
        loss = VP.calc_vp_loss(sample_lines, Kinv, depth_map, vds[j])
        print(loss)
        # print(p)
        # print(p.shape)

# %%
import matplotlib.pyplot as plt

plt.plot([1, 2, 3])
print('hi')
#%%
print("a")
#%%
print("a")
#%%
#%%
# import torch
# from models.unet_adaptive_bins import UnetAdaptiveBins

# model = UnetAdaptiveBins.build(256)
# x = torch.randn(2, 3, 480, 640)
# model(x)
