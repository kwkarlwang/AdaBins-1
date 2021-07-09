import argparse
import os
import sys
import uuid
from datetime import datetime as dt

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.tensor import Tensor
import torch.utils.data.distributed
import wandb
from tqdm import tqdm

import model_io
import models
import utils
from dataloader import DepthDataLoader
from loss import SILogLoss, BinsChamferLoss
from utils import IoU, RunningAverage, StreamSegMetrics, colorize

import random

# os.environ['WANDB_MODE'] = 'dryrun'
PROJECT = "MDE-AdaBins"
logging = False


def is_rank_zero(args):
    return args.rank == 0


import matplotlib


def colorize(value, vmin=10, vmax=1000, cmap="plasma"):
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

    img = value[:, :, :3]

    #     return img.transpose((2, 0, 1))
    return img


def log_images(img, depth, pred, args, step):
    depth = colorize(depth, vmin=args.min_depth, vmax=args.max_depth)
    pred = colorize(pred, vmin=args.min_depth, vmax=args.max_depth)
    wandb.log(
        {
            "Input": [wandb.Image(img)],
            "GT": [wandb.Image(depth)],
            "Prediction": [wandb.Image(pred)],
        },
        step=step,
    )


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    ###################################### Load model ##############################################

    model = models.UnetAdaptiveBins.build(
        n_bins=args.n_bins,
        min_val=args.min_depth,
        max_val=args.max_depth,
        norm=args.norm,
        seg_classes=args.seg_classes,
    )

    ################################################################################################

    if args.gpu is not None:  # If a gpu is set by user: NO PARALLELISM!!
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    args.multigpu = False
    if args.distributed:
        # Use DDP
        args.multigpu = True
        args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
        args.batch_size = int(args.batch_size / ngpus_per_node)
        # args.batch_size = 8
        args.workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
        print(args.gpu, args.rank, args.batch_size, args.workers)
        torch.cuda.set_device(args.gpu)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            output_device=args.gpu,
            find_unused_parameters=False,
        )

    elif args.gpu is None:
        # Use DP
        args.multigpu = True
        model = model.cuda()
        model = torch.nn.DataParallel(model)
        print("USING DATA PARALLEL")

    args.epoch = 0
    args.last_epoch = -1
    train(
        model,
        args,
        epochs=args.epochs,
        lr=args.lr,
        device=args.gpu,
        root=args.root,
        experiment_name=args.name,
        optimizer_state_dict=None,
    )


def train(
    model,
    args,
    epochs=10,
    experiment_name="DeepLab",
    lr=0.0001,
    root=".",
    device=None,
    optimizer_state_dict=None,
):
    global PROJECT
    if device is None:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    ###################################### Logging setup #########################################
    print(f"Training {experiment_name}")

    run_id = f"{dt.now().strftime('%d-%h_%H-%M')}-nodebs{args.bs}-tep{epochs}-lr{lr}-wd{args.wd}-{uuid.uuid4()}"
    name = f"{experiment_name}_{run_id}"
    should_write = (not args.distributed) or args.rank == 0
    should_log = should_write and logging
    if should_log:
        tags = args.tags.split(",") if args.tags != "" else None
        if args.dataset != "nyu":
            PROJECT = PROJECT + f"-{args.dataset}"
        wandb.init(
            project=PROJECT,
            name=name,
            config=args,
            dir=args.root,
            tags=tags,
            notes=args.notes,
        )
        # wandb.watch(model)
    ################################################################################################

    test_loader = DepthDataLoader(args, "online_eval_seg").data
    train_loader = DepthDataLoader(args, "train").data
    train_seg_loader = DepthDataLoader(args, "train_seg").data

    ###################################### losses ##############################################
    criterion_ueff = SILogLoss()
    criterion_bins = BinsChamferLoss() if args.chamfer else None

    seg_criterion = nn.CrossEntropyLoss()
    ################################################################################################

    model.train()

    ###################################### Optimizer ################################################
    if args.same_lr:
        print("Using same LR")
        params = model.parameters()
    else:
        print("Using diff LR")
        m = model.module if args.multigpu else model
        params = [
            {"params": m.get_1x_lr_params(), "lr": lr / 10},
            {"params": m.get_10x_lr_params(), "lr": lr},
        ]

    optimizer = optim.AdamW(params, weight_decay=args.wd, lr=args.lr)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    ################################################################################################
    # some globals
    iters = len(train_loader) + len(train_seg_loader)
    step = args.epoch * iters
    best_loss = np.inf

    ###################################### Scheduler ###############################################
    steps_per_epoch = len(train_loader) + len(train_seg_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        lr,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        last_epoch=args.last_epoch,
        div_factor=args.div_factor,
        final_div_factor=args.final_div_factor,
    )
    if args.resume != "" and scheduler is not None:
        scheduler.step(args.epoch + 1)
    ################################################################################################

    ################################# DELETE ##################################################
    model.eval()
    metrics, val_si, miou, val_ce = validate(
        args, model, test_loader, criterion_ueff, 0, epochs, seg_criterion, device,
    )

    # print("Validated: {}".format(metrics))
    if should_log:
        wandb.log(
            {
                f"Test/{criterion_ueff.name}": val_si.get_value(),
                f"Test/CrossEntropyLoss": val_ce.get_value(),
                # f"Test/{criterion_bins.name}": val_bins.get_value()
            },
            step=step,
        )

        wandb.log({f"Metrics/{k}": v for k, v in metrics.items()}, step=step)
        wandb.log({f"Metrics/mIoU": miou}, step=step)

        model_io.save_checkpoint(
            model,
            optimizer,
            0,
            f"{experiment_name}_{run_id}_latest.pt",
            root=os.path.join(root, "checkpoints"),
        )

    if metrics["abs_rel"] < best_loss and should_write:
        model_io.save_checkpoint(
            model,
            optimizer,
            0,
            f"{experiment_name}_{run_id}_best.pt",
            root=os.path.join(root, "checkpoints"),
        )
        best_loss = metrics["abs_rel"]
    model.train()
    #################################################################################################
    # max_iter = len(train_loader) * epochs
    for epoch in range(args.epoch, epochs):

        ################################# Train loop ##########################################################
        if should_log:
            wandb.log({"Epoch": epoch}, step=step)
        train_loader_it = iter(train_loader)
        train_loader_is_done = False
        train_seg_loader_it = iter(train_seg_loader)
        train_seg_loader_is_done = False

        for i in (
            tqdm(
                range(steps_per_epoch),
                desc=f"Epoch: {epoch + 1}/{epochs}. Loop: Train",
                total=steps_per_epoch,
            )
            if is_rank_zero(args)
            else range(steps_per_epoch)
        ):

            #################### random select a loader ########################
            has_seg = False
            if train_loader_is_done or (
                train_seg_loader_is_done == False and random.random() < 0.2
            ):
                batch = next(train_seg_loader_it, None)
                if batch is not None:
                    has_seg = True
                else:
                    train_seg_loader_is_done = True
                    batch = next(train_loader_it, None)
                    if batch is None:
                        train_loader_is_done = True
            else:
                batch = next(train_loader_it, None)
                if batch is None:
                    train_loader_is_done = True
                    batch = next(train_seg_loader_it, None)
                    if batch is not None:
                        has_seg = True
                    else:
                        train_seg_loader_is_done = True

            if train_loader_is_done and train_seg_loader_is_done:
                break
            ###########################################################
            if has_seg:
                if isinstance(model, torch.nn.DataParallel) or isinstance(
                    model, torch.nn.parallel.DistributedDataParallel
                ):
                    model.module.unfreeze_seg()
                else:
                    model.unfreeze_seg()
            else:

                if isinstance(model, torch.nn.DataParallel) or isinstance(
                    model, torch.nn.parallel.DistributedDataParallel
                ):
                    model.module.freeze_seg()
                else:
                    model.freeze_seg()

            optimizer.zero_grad()

            img = batch["image"].to(device)
            depth = batch["depth"].to(device)
            if "has_valid_depth" in batch:
                if not batch["has_valid_depth"]:
                    continue
            bin_edges, pred, seg_out = model(img)

            mask = depth > args.min_depth
            l_dense = criterion_ueff(
                pred, depth, mask=mask.to(torch.bool), interpolate=True
            )

            if args.w_chamfer > 0:
                l_chamfer = criterion_bins(bin_edges, depth)
            else:
                l_chamfer = torch.Tensor([0]).to(img.device)

            loss = l_dense + args.w_chamfer * l_chamfer
            seg_loss = 0
            if has_seg:
                seg = batch["seg"].to(torch.long).to(device)
                seg = seg.squeeze()
                seg_out = nn.functional.interpolate(
                    seg_out, seg.shape[-2:], mode="nearest"
                )
                seg_loss = seg_criterion(seg_out, seg)
                loss += args.w_seg * seg_loss

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)  # optional
            optimizer.step()
            if should_log and step % 5 == 0:
                wandb.log({f"Train/{criterion_ueff.name}": l_dense.item()}, step=step)
                wandb.log({f"Train/{criterion_bins.name}": l_chamfer.item()}, step=step)
            if should_log and has_seg:
                wandb.log({f"Train/CrossEntropyLoss": l_chamfer.item()}, step=step)

            step += 1
            scheduler.step()

            ########################################################################################################

            if should_write and step % args.validate_every == 0:

                ################################# Validation loop ##################################################
                model.eval()
                metrics, val_si, miou, val_ce = validate(
                    args,
                    model,
                    test_loader,
                    criterion_ueff,
                    epoch,
                    epochs,
                    seg_criterion,
                    device,
                )

                # print("Validated: {}".format(metrics))
                if should_log:
                    wandb.log(
                        {
                            f"Test/{criterion_ueff.name}": val_si.get_value(),
                            f"Test/CrossEntropyLoss": val_ce.get_value(),
                            # f"Test/{criterion_bins.name}": val_bins.get_value()
                        },
                        step=step,
                    )

                    wandb.log(
                        {f"Metrics/{k}": v for k, v in metrics.items()}, step=step
                    )
                    wandb.log({f"Metrics/mIoU": miou}, step=step)

                    model_io.save_checkpoint(
                        model,
                        optimizer,
                        epoch,
                        f"{experiment_name}_{run_id}_latest.pt",
                        root=os.path.join(root, "checkpoints"),
                    )

                if metrics["abs_rel"] < best_loss and should_write:
                    model_io.save_checkpoint(
                        model,
                        optimizer,
                        epoch,
                        f"{experiment_name}_{run_id}_best.pt",
                        root=os.path.join(root, "checkpoints"),
                    )
                    best_loss = metrics["abs_rel"]
                model.train()
                #################################################################################################

    return model


def validate(
    args, model, test_loader, criterion_ueff, epoch, epochs, seg_criterion, device="cpu"
):
    with torch.no_grad():
        val_si = RunningAverage()
        # val_bins = RunningAverage()
        metrics = utils.RunningAverageDict()

        val_ce = RunningAverage()

        # iou = IoU(ignore_index=0, num_classes=41)
        # iou = StreamSegMetrics(num_classes=41)

        # i = 0
        for batch in (
            tqdm(test_loader, desc=f"Epoch: {epoch + 1}/{epochs}. Loop: Validation")
            if is_rank_zero(args)
            else test_loader
        ):
            img = batch["image"].to(device)
            depth = batch["depth"].to(device)
            if "has_valid_depth" in batch:
                if not batch["has_valid_depth"]:
                    continue
            depth = depth.squeeze().unsqueeze(0).unsqueeze(0)
            bins, pred, seg_out = model(img)

            mask = depth > args.min_depth

            l_dense = criterion_ueff(
                pred, depth, mask=mask.to(torch.bool), interpolate=True
            )
            val_si.append(l_dense.item())

            pred = nn.functional.interpolate(
                pred, depth.shape[-2:], mode="bilinear", align_corners=True
            )

            pred = pred.squeeze().cpu().numpy()
            pred[pred < args.min_depth_eval] = args.min_depth_eval
            pred[pred > args.max_depth_eval] = args.max_depth_eval
            pred[np.isinf(pred)] = args.max_depth_eval
            pred[np.isnan(pred)] = args.min_depth_eval

            gt_depth = depth.squeeze().cpu().numpy()
            valid_mask = np.logical_and(
                gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval
            )
            eval_mask = np.zeros(valid_mask.shape).astype(int)
            if args.garg_crop or args.eigen_crop:
                gt_height, gt_width = gt_depth.shape

                if args.garg_crop:
                    eval_mask[
                        int(0.40810811 * gt_height) : int(0.99189189 * gt_height),
                        int(0.03594771 * gt_width) : int(0.96405229 * gt_width),
                    ] = 1

                elif args.eigen_crop:
                    if args.dataset == "kitti":
                        eval_mask[
                            int(0.3324324 * gt_height) : int(0.91351351 * gt_height),
                            int(0.0359477 * gt_width) : int(0.96405229 * gt_width),
                        ] = 1
                    else:
                        eval_mask[45:471, 41:601] = 1
            valid_mask = np.logical_and(valid_mask, eval_mask)
            metrics.update(utils.compute_errors(gt_depth[valid_mask], pred[valid_mask]))

            seg = batch["seg"].to(torch.long).to(device)
            seg = seg.squeeze().unsqueeze(0)

            seg_out = nn.functional.interpolate(seg_out, seg.shape[-2:], mode="nearest")
            seg_loss = seg_criterion(seg_out, seg)
            val_ce.append(seg_loss)

            # seg_pred = seg_out.squeeze().argmax(dim=0).cpu().numpy()
            # seg = seg.squeeze().cpu().numpy()

            # iou.update(seg_pred[eval_mask], seg[eval_mask])

            # i += 1
            # if i > 50:
            #     break

        # miou = iou.compute()
        miou = 0

        return metrics.get_value(), val_si, miou, val_ce


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)


if __name__ == "__main__":

    # Arguments
    parser = argparse.ArgumentParser(
        description="Training script. Default values of all arguments are recommended for reproducibility",
        fromfile_prefix_chars="@",
        conflict_handler="resolve",
    )
    parser.convert_arg_line_to_args = convert_arg_line_to_args
    parser.add_argument(
        "--epochs", default=25, type=int, help="number of total epochs to run"
    )
    parser.add_argument(
        "--n-bins",
        "--n_bins",
        default=80,
        type=int,
        help="number of bins/buckets to divide depth range into",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.000357,
        type=float,
        help="max learning rate",
    )
    parser.add_argument(
        "--wd", "--weight-decay", default=0.1, type=float, help="weight decay"
    )
    parser.add_argument(
        "--w_chamfer",
        "--w-chamfer",
        default=0.1,
        type=float,
        help="weight value for chamfer loss",
    )

    parser.add_argument(
        "--w_seg",
        "--w-seg",
        default=0.3,
        type=float,
        help="weight value for chamfer loss",
    )

    parser.add_argument(
        "--div-factor",
        "--div_factor",
        default=25,
        type=float,
        help="Initial div factor for lr",
    )
    parser.add_argument(
        "--final-div-factor",
        "--final_div_factor",
        default=100,
        type=float,
        help="final div factor for lr",
    )

    parser.add_argument("--bs", default=16, type=int, help="batch size")
    parser.add_argument(
        "--validate-every",
        "--validate_every",
        default=100,
        type=int,
        help="validation period",
    )
    parser.add_argument("--gpu", default=None, type=int, help="Which gpu to use")
    parser.add_argument("--name", default="UnetAdaptiveBins")
    parser.add_argument(
        "--norm",
        default="linear",
        type=str,
        help="Type of norm/competition for bin-widths",
        choices=["linear", "softmax", "sigmoid"],
    )
    parser.add_argument(
        "--same-lr",
        "--same_lr",
        default=False,
        action="store_true",
        help="Use same LR for all param groups",
    )
    parser.add_argument(
        "--distributed", default=False, action="store_true", help="Use DDP if set"
    )
    parser.add_argument(
        "--root", default=".", type=str, help="Root folder to save data in"
    )
    parser.add_argument("--resume", default="", type=str, help="Resume from checkpoint")

    parser.add_argument("--notes", default="", type=str, help="Wandb notes")
    parser.add_argument("--tags", default="sweep", type=str, help="Wandb tags")

    parser.add_argument(
        "--workers", default=11, type=int, help="Number of workers for data loading"
    )
    parser.add_argument(
        "--dataset", default="nyu", type=str, help="Dataset to train on"
    )

    parser.add_argument(
        "--data_path", default="../dataset/nyu/sync/", type=str, help="path to dataset"
    )
    parser.add_argument(
        "--gt_path", default="../dataset/nyu/sync/", type=str, help="path to dataset"
    )

    parser.add_argument(
        "--filenames_file",
        default="./train_test_inputs/nyudepthv2_train_files_with_gt.txt",
        type=str,
        help="path to the filenames text file",
    )

    parser.add_argument(
        "--filenames_file_seg",
        default="./train_test_inputs/nyudepthv2_train_files_with_gt_seg.txt",
        type=str,
        help="path to the filenames text file with segmentation",
    )

    parser.add_argument("--input_height", type=int, help="input height", default=416)
    parser.add_argument("--input_width", type=int, help="input width", default=544)
    parser.add_argument(
        "--max_depth", type=float, help="maximum depth in estimation", default=10
    )
    parser.add_argument(
        "--min_depth", type=float, help="minimum depth in estimation", default=1e-3
    )

    parser.add_argument(
        "--do_random_rotate",
        default=True,
        help="if set, will perform random rotation for augmentation",
        action="store_true",
    )
    parser.add_argument(
        "--degree", type=float, help="random rotation maximum degree", default=2.5
    )
    parser.add_argument(
        "--do_kb_crop",
        help="if set, crop input images as kitti benchmark images",
        action="store_true",
    )
    parser.add_argument(
        "--use_right",
        help="if set, will randomly use right images when train on KITTI",
        action="store_true",
    )

    parser.add_argument(
        "--data_path_eval",
        default="../dataset/nyu/official_splits/test/",
        type=str,
        help="path to the data for online evaluation",
    )
    parser.add_argument(
        "--gt_path_eval",
        default="../dataset/nyu/official_splits/test/",
        type=str,
        help="path to the groundtruth data for online evaluation",
    )
    parser.add_argument(
        "--filenames_file_eval",
        default="./train_test_inputs/nyudepthv2_test_files_with_gt.txt",
        type=str,
        help="path to the filenames text file for online evaluation",
    )

    parser.add_argument(
        "--min_depth_eval",
        type=float,
        help="minimum depth for evaluation",
        default=1e-3,
    )
    parser.add_argument(
        "--max_depth_eval", type=float, help="maximum depth for evaluation", default=10
    )
    parser.add_argument(
        "--eigen_crop",
        default=True,
        help="if set, crops according to Eigen NIPS14",
        action="store_true",
    )
    parser.add_argument(
        "--garg_crop",
        help="if set, crops according to Garg  ECCV16",
        action="store_true",
    )

    parser.add_argument(
        "--seg_classes",
        default=41,
        type=int,
        help="Number of classes in the segmentation",
    )

    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = "@" + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()

    args.batch_size = args.bs
    args.num_threads = args.workers
    args.mode = "train"
    args.chamfer = args.w_chamfer > 0
    if args.root != "." and not os.path.isdir(args.root):
        os.makedirs(args.root)

    try:
        node_str = os.environ["SLURM_JOB_NODELIST"].replace("[", "").replace("]", "")
        nodes = node_str.split(",")

        args.world_size = len(nodes)
        args.rank = int(os.environ["SLURM_PROCID"])

    except KeyError as e:
        # We are NOT using SLURM
        args.world_size = 1
        args.rank = 0
        nodes = ["127.0.0.1"]

    if args.distributed:
        mp.set_start_method("forkserver")

        print(args.rank)
        port = np.random.randint(15000, 15025)
        args.dist_url = "tcp://{}:{}".format(nodes[0], port)
        print(args.dist_url)
        args.dist_backend = "nccl"
        args.gpu = None

    ngpus_per_node = torch.cuda.device_count()
    args.num_workers = args.workers
    args.ngpus_per_node = ngpus_per_node

    # first download the model to cache the result
    basemodel_name = "tf_efficientnet_b5_ap"
    torch.hub.load(
        "rwightman/gen-efficientnet-pytorch", basemodel_name, pretrained=True
    )

    if args.distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        if ngpus_per_node == 1:
            args.gpu = 0
        main_worker(args.gpu, ngpus_per_node, args)
