import base64
import math
import re
from io import BytesIO

import matplotlib.cm
import numpy as np
import torch
import torch.nn
from PIL import Image
from skimage.measure import label, regionprops


class RelDepth:
    def __init__(self,
                 args,
                 target_trainId=[5, 11, 12, 13, 14, 15, 16, 17, 18]) -> None:
        self.target_trainId = target_trainId
        self.relative_depth = args.relative_depth
        self.scale_coefficient = args.scale_coefficient

    def transform(self, seg_mask: np.ndarray, cat_map: np.ndarray):
        pass

    def validate(self,
                 seg_mask: np.ndarray,
                 cat_map: np.ndarray,
                 depth: np.ndarray,
                 use_mean=False) -> np.ndarray:

        seg_mask = seg_mask.copy()
        valid_id = []
        for i in range(len(cat_map)):
            if cat_map[i] in self.target_trainId:
                valid_id.append(i)

        for idx, x in np.ndenumerate(seg_mask):
            if seg_mask[idx] not in valid_id:
                seg_mask[idx] = 100
        seg_mask[seg_mask == 0] = 200
        seg_mask[seg_mask == 100] = 0
        label_img = label(seg_mask)
        regions = regionprops(label_img)
        # 3.
        rdm = np.zeros_like(seg_mask, dtype=np.float32)
        for region in regions:
            mean_depth = 0
            nnz_cnt = 0
            y = region.coords[:, 0]
            x = region.coords[:, 1]
            if use_mean:
                nnz_cnt = np.sum(depth[y, x] > 0)
                if nnz_cnt:
                    mean_depth = np.sum(depth[y, x]) / nnz_cnt
                    rdm[y, x] = mean_depth
            else:
                flatten_depth = depth[y, x].flatten()
                mask = flatten_depth > 0
                if mask.sum() != 0:
                    rdm[y, x] = np.median(flatten_depth[mask])

        if self.relative_depth and np.max(rdm) != 0:
            rdm: np.ndarray = rdm / np.max(rdm)
        rdm = rdm * self.scale_coefficient
        return rdm - 0.5


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
