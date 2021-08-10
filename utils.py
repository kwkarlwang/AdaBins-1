import base64
import math
import re
from io import BytesIO

import matplotlib.cm
import numpy as np
import torch
import torch.nn
from PIL import Image


class VP:
    def __init__(self, device='cpu') -> None:
        self.count = 0
        self.loss = 0
        self.device = device
        self.total_count = 0

    def sample_points(
        self,
        lines: torch.Tensor,
        num_points: int = 2,
    ):
        x1 = lines[:, 0:2]  # startpoint
        x2 = lines[:, 2:4]  # endpoint

        direction = x2 - x1
        t1 = torch.rand((num_points, lines.shape[0], 1)).to(self.device)
        t2 = torch.rand((num_points, lines.shape[0], 1)).to(self.device)
        start = torch.round(x1 + t1 * direction).to(torch.long)
        end = torch.round(x1 + t2 * direction).to(torch.long)
        return torch.dstack((start, end)).reshape(-1, 4).to(self.device)

    def update(self, lines: torch.Tensor, Kinv: torch.Tensor,
               pred: torch.Tensor, vd: torch.Tensor, depth: torch.Tensor):
        x1, y1, x2, y2 = (lines[:, 0], lines[:, 1], lines[:, 2], lines[:, 3])
        depth1, depth2 = pred[y1, x1], pred[y2, x2]
        depth1_r, depth2_r = depth[y1, x1], depth[y2, x2]

        ones = torch.ones(lines.shape[0]).to(self.device)

        # 3xn
        u = Kinv @ torch.vstack((lines[:, 0:2].T, ones))
        v = Kinv @ torch.vstack((lines[:, 2:4].T, ones))

        # nx3
        u3d = (depth1 * u).T
        # nx3
        v3d = (depth2 * v).T

        u3d_r = (depth1_r * u).T
        v3d_r = (depth2_r * v).T
        vd_repeat = vd[None, :].repeat(
            (lines.shape[0], 1)).to(self.device).to(torch.float32)
        loss = torch.norm(torch.cross((u3d - v3d), vd_repeat, dim=1))
        loss_r = torch.norm(torch.cross((u3d_r - v3d_r), vd_repeat, dim=1))

        # only calculate back prop high loss
        invalid_loss = loss < loss_r
        loss[invalid_loss] *= 0
        self.loss += loss.sum()
        self.count += lines.shape[0] - invalid_loss.sum()
        self.total_count += len(lines)

    def compute(self):
        print(f'Total count: {self.total_count}')
        print(f'Valid count: {self.count}')
        return self.loss / self.count


class RunningAverage:
    def __init__(self):
        self.avg = 0
        self.count = 0

    def append(self, value):
        self.avg = (value + self.count * self.avg) / (self.count + 1)
        self.count += 1

    def get_value(self):
        return self.avg


def denormalize(x, device="cpu"):
    mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    return x * std + mean


class RunningAverageDict:
    def __init__(self):
        self._dict: dict = None  # type: ignore

    def update(self, new_dict):
        if self._dict is None:
            self._dict = dict()
            for key, value in new_dict.items():
                self._dict[key] = RunningAverage()

        for key, value in new_dict.items():
            self._dict[key].append(value)

    def get_value(self):
        return {key: value.get_value() for key, value in self._dict.items()}


def colorize(value, vmin=10, vmax=1000, cmap="magma_r"):
    value = value.cpu().numpy()[0, :, :]
    invalid_mask = value == -1

    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.0
    # squeeze last dim if it exists
    # value = value.squeeze(axis=0)
    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)  # (nxmx4)
    value[invalid_mask] = 255
    img = value[:, :, :3]

    #     return img.transpose((2, 0, 1))
    return img


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25**2).mean()
    a3 = (thresh < 1.25**3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred)**2) / gt)

    rmse = (gt - pred)**2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred))**2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err**2) - np.mean(err)**2) * 100

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return dict(
        a1=a1,
        a2=a2,
        a3=a3,
        abs_rel=abs_rel,
        rmse=rmse,
        log_10=log_10,
        rmse_log=rmse_log,
        silog=silog,
        sq_rel=sq_rel,
    )


class IoU:
    def __init__(self,
                 num_classes: int,
                 ignore_index: int = None,
                 device="cpu") -> None:
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.intersections = torch.zeros(num_classes).to(device)
        self.unions = torch.zeros(num_classes).to(device)

    def update(self, target, pred):
        # pred:   N, H, W
        # target: N, H, W

        # print(pred)
        # print(pred.max())
        # print(pred.min())
        # print(pred.shape)
        for i in range(self.num_classes):
            if i == self.ignore_index:
                continue
            # print(i)
            predMask = pred == i
            # print(predMask.sum())
            targetMask = target == i
            # print(targetMask.sum())
            intersection = (predMask & targetMask).sum()
            # print(intersection)
            # print(intersection.float())
            union = (predMask | targetMask).sum()
            self.intersections[i] += intersection.float()
            self.unions[i] += union.float()

    def compute(self):
        mask = self.unions != 0
        res = (self.intersections[mask] / self.unions[mask]).mean()
        return res.cpu().item()


#####################################################################################################
# https://github.com/VainF/DeepLabV3Plus-Pytorch/blob/master/metrics/stream_metrics.py


class _StreamMetrics(object):
    def __init__(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def update(self, gt, pred):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def to_str(self, metrics):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()


class StreamSegMetrics(_StreamMetrics):
    """
    Stream Metrics for Semantic Segmentation Task
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes))

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(),
                                                     lp.flatten())

    def compute(self):
        hist = self.confusion_matrix
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) -
                              np.diag(hist))
        mean_iu = np.nanmean(iu)
        # freq = hist.sum(axis=1) / hist.sum()
        # fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        # cls_iu = dict(zip(range(self.n_classes), iu))

        return mean_iu

    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            if k != "Class IoU":
                string += "%s: %f\n" % (k, v)

        # string+='Class IoU:\n'
        # for k, v in results['Class IoU'].items():
        #    string += "\tclass %d: %f\n"%(k, v)
        return string

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true > 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.num_classes**2,
        ).reshape(self.num_classes, self.num_classes)
        return hist

    def get_results(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) -
                              np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.num_classes), iu))

        return {
            "Overall Acc": acc,
            "Mean Acc": acc_cls,
            "FreqW Acc": fwavacc,
            "Mean IoU": mean_iu,
            "Class IoU": cls_iu,
        }

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))


#####################################################################################################


##################################### Demo Utilities ############################################
def b64_to_pil(b64string):
    image_data = re.sub("^data:image/.+;base64,", "", b64string)
    # image = Image.open(cStringIO.StringIO(image_data))
    return Image.open(BytesIO(base64.b64decode(image_data)))


# Compute edge magnitudes
from scipy import ndimage


def edges(d):
    dx = ndimage.sobel(d, 0)  # horizontal derivative
    dy = ndimage.sobel(d, 1)  # vertical derivative
    return np.abs(dx) + np.abs(dy)


class PointCloudHelper:
    def __init__(self, width=640, height=480):
        self.xx, self.yy = self.worldCoords(width, height)

    def worldCoords(self, width=640, height=480):
        hfov_degrees, vfov_degrees = 57, 43
        hFov = math.radians(hfov_degrees)
        vFov = math.radians(vfov_degrees)
        cx, cy = width / 2, height / 2
        fx = width / (2 * math.tan(hFov / 2))
        fy = height / (2 * math.tan(vFov / 2))
        xx, yy = np.tile(range(width), height), np.repeat(range(height), width)
        xx = (xx - cx) / fx
        yy = (yy - cy) / fy
        return xx, yy

    def depth_to_points(self, depth):
        depth[edges(depth) > 0.3] = np.nan  # Hide depth edges
        length = depth.shape[0] * depth.shape[1]
        # depth[edges(depth) > 0.3] = 1e6  # Hide depth edges
        z = depth.reshape(length)

        return np.dstack((self.xx * z, self.yy * z, z)).reshape((length, 3))


#####################################################################################################

if __name__ == "__main__":
    iou = IoU(4, 0)
    pred = torch.Tensor([[[5, 2], [3, 4]], [[5, 6], [7, 8]]])
    #%%
    pred
    #%%
    target = torch.Tensor([[[5, 1], [3, 3]], [[5, 5], [5, 5]]])
    #%%
    iou.update(pred, target)
    print(iou.compute())
