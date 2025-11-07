import datetime
import logging
import math
import os
import random
import shutil
import sys
import time
from types import SimpleNamespace

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import amfinder_config as AmfConfig
import amfinder_log as AmfLog
import amfinder_model as AmfModel
from fixmatch.dataset.amf_fixmatch import get_amf
from fixmatch.utils import AverageMeter, accuracy, get_confusion_matrix

# TODO constants that use AmfConfig need to be moved to local variables to work with FastAPI
COLONISATION_TYPE = AmfConfig.get("colonisation_type")
CLASS_NAMES = AmfConfig.get("class_names")[COLONISATION_TYPE]

best_acc = 0

try:
    wd = sys._MEIPASS
except AttributeError:
    wd = os.getcwd()


def save_checkpoint(state, is_best, checkpoint, filename="checkpoint.pth.tar"):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, "model_best.pth.tar"))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    num_cycles=7.0 / 16.0,
    last_epoch=-1,
):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def run(image_path):
    with open(os.path.join(wd, "config/base_config.yml"), "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    args = SimpleNamespace(**config)

    # Overwrite the default args with the cl args
    args.train = True
    args.config = os.path.join(wd, "config/base_config.yml")
    curr_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

    # Overwrite root
    args.root_path = image_path

    # Overwrite the default args with the cl args
    args.out = os.path.join(
        AmfConfig.get("outdir"), f"{args.out}/{curr_datetime}_results"
    )

    global best_acc

    if args.local_rank == -1:
        device = AmfConfig.get("device")
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
        args.device = device
    else:
        torch.cuda.set_device(args.local_rank)
        device = AmfConfig.get("device")
        torch.distributed.init_process_group(backend="nccl")
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1
        args.device = device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )

    AmfLog.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.amp}",
    )

    AmfLog.info(dict(args.__dict__.items()))

    if args.seed is not None:
        set_seed(args)
    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)

        # Assuming args.out is the directory where you want to save the YAML file
        yaml_file_path = os.path.join(args.out, "config.yaml")

    def serialize_object(obj):
        # If the object is a torch.device, convert it to its string representation
        if hasattr(obj, "type"):
            return str(obj)
        # Add other custom serialization rules as needed
        # Return the object itself if no custom rule applies
        return obj

    # Assuming `args` is an object with attributes you want to dump
    args_dict_serializable = {k: serialize_object(v) for k, v in args.__dict__.items()}

    # Write the serializable dict to the YAML file
    with open(yaml_file_path, "w") as file:
        yaml.safe_dump(args_dict_serializable, file, default_flow_style=False)

    args.num_classes = len(CLASS_NAMES)

    if args.arch == "wideresnet":
        args.model_depth = 16
        args.model_width = 4
    elif args.arch == "resnext":
        args.model_cardinality = 8
        args.model_depth = 29
        args.model_width = 64

    labeled_dataset, unlabeled_dataset, val_dataset = get_amf(args)

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=AmfConfig.get("num_workers"),
        drop_last=True,
    )

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
        batch_size=int(args.batch_size * args.mu),
        num_workers=AmfConfig.get("num_workers"),
        drop_last=True,
    )

    test_loader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=args.batch_size,
        num_workers=AmfConfig.get("num_workers"),
    )

    print(f"Length of the unlabelled dataset: {len(unlabeled_dataset)}")
    print(f"Batch length: {len(unlabeled_trainloader)}")
    if len(unlabeled_trainloader) == 0:
        print("Unlabeled trainloader is empty!")

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    model = AmfModel.load()

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

    no_decay = ["bias", "bn"]
    grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.wdecay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = optim.SGD(
        grouped_parameters, lr=args.lr, momentum=0.9, nesterov=args.nesterov
    )
    args.epochs = AmfConfig.get("epochs")
    args.eval_step = math.ceil(args.num_labeled / args.batch_size)
    args.total_steps = args.epochs * args.eval_step

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.total_steps
    )

    args.start_epoch = 0

    if args.resume:
        AmfLog.info("==> Resuming from checkpoint..")
        assert os.path.isfile(args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint["best_acc"]
        args.start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    AmfLog.info("***** Running training *****")
    AmfLog.info(f"  Task = {args.dataset}@{args.num_labeled}")
    AmfLog.info(f"  Num Epochs = {args.epochs}")
    AmfLog.info(f"  Batch size per GPU = {args.batch_size}")
    AmfLog.info(f"  Total train batch size = {args.batch_size * args.world_size}")
    AmfLog.info(f"  Total optimization steps = {args.total_steps}")

    model.zero_grad()
    train(
        args,
        labeled_trainloader,
        unlabeled_trainloader,
        test_loader,
        model,
        optimizer,
        scheduler,
    )


def train(
    args,
    labeled_trainloader,
    unlabeled_trainloader,
    test_loader,
    model,
    optimizer,
    scheduler,
):
    global best_acc
    test_accs = []
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    mask_probs = AverageMeter()
    end = time.time()

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)

    model.train()
    if AmfConfig.get("mlflow_flag"):
        mlflow.start_run()

    for epoch in range(args.start_epoch, args.epochs):
        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step), disable=args.local_rank not in [-1, 0])
        for batch_idx in range(args.eval_step):
            try:
                inputs_x, targets_x = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_trainloader)
                inputs_x, targets_x = next(labeled_iter)

            try:
                (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)

            data_time.update(time.time() - end)
            batch_size = inputs_x.shape[0]
            inputs = interleave(
                torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2 * args.mu + 1
            ).to(args.device)
            targets_x = targets_x.to(args.device)
            logits = model(inputs)
            logits = de_interleave(logits, 2 * args.mu + 1)
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
            del logits

            Lx = F.cross_entropy(logits_x, targets_x, reduction="mean")

            pseudo_label = torch.softmax(logits_u_w.detach() / args.T, dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()

            Lu = (
                F.cross_entropy(logits_u_s, targets_u, reduction="none") * mask
            ).mean()

            loss = Lx + args.lambda_u * Lu

            loss.backward()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            mask_probs.update(mask.mean().item())
            if not args.no_progress:
                p_bar.set_description(
                    "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Mask: {mask:.2f}. ".format(
                        epoch=epoch + 1,
                        epochs=args.epochs,
                        batch=batch_idx + 1,
                        iter=args.eval_step,
                        lr=scheduler.get_last_lr()[0],
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        loss_x=losses_x.avg,
                        loss_u=losses_u.avg,
                        mask=mask_probs.avg,
                    )
                )
                p_bar.update()

        if not args.no_progress:
            p_bar.close()

        test_model = model

        if args.local_rank in [-1, 0]:
            test_loss, test_acc, f1, fp, fn, tp, tn = test(
                args, test_loader, test_model, epoch
            )
            epsilon = 1e-10  # A small number to prevent division by zero

            # Log metrics
            if AmfConfig.get("mlflow_flag"):
                mlflow.log_metric("train_loss", losses.avg)
                mlflow.log_metric("train_loss_x", losses_x.avg)
                mlflow.log_metric("train_loss_u", losses_u.avg)
                mlflow.log_metric("mask", mask_probs.avg)
                mlflow.log_metric("test_acc", test_acc)
                mlflow.log_metric("test_loss", test_loss)
                mlflow.log_metric("fpr", fp / (fp + tn + epsilon))
                mlflow.log_metric("fnr", fn / (tp + fn + epsilon))
                mlflow.log_metric("recall", tp / (tp + fn + epsilon))
                mlflow.log_metric("precision", tp / (tp + fp + epsilon))

            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)

            model_to_save = model.module if hasattr(model, "module") else model
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model_to_save.state_dict(),
                    "acc": test_acc,
                    "best_acc": best_acc,
                    "f1": f1,
                    "fp": fp,
                    "fn": fn,
                    "tp": tp,
                    "tn": tn,
                    "precision": tp / (tp + fp),
                    "recall": tp / (tp + fn),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                is_best,
                args.out,
            )

            test_accs.append(test_acc)
            AmfLog.info("Best top-1 acc: {:.2f}".format(best_acc))
            AmfLog.info("Mean top-1 acc: {:.2f}\n".format(np.mean(test_accs[-20:])))

        if AmfConfig.get("mlflow_flag"):
            AmfLog.info(f"MLflow runs are saved in: {mlflow.get_tracking_uri()}")

        AmfLog.info(f"Main modelling output is saved in: {args.out}")


def test(args, test_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    tp = AverageMeter()
    fp = AverageMeter()
    tn = AverageMeter()
    fn = AverageMeter()
    end = time.time()

    def calculate_tp_fp_tn_fn(conf_matrix):
        TPs = np.diag(conf_matrix)
        FPs = np.sum(conf_matrix, axis=0) - TPs
        FNs = np.sum(conf_matrix, axis=1) - TPs
        TNs = np.sum(conf_matrix) - (FPs + FNs + TPs)

        # Sum across all classes
        total_tp = np.sum(TPs)
        total_fp = np.sum(FPs)
        total_fn = np.sum(FNs)
        total_tn = np.sum(TNs)

        return total_tp, total_fp, total_fn, total_tn

    if not args.no_progress:
        test_loader = tqdm(test_loader, disable=args.local_rank not in [-1, 0])

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            confusion_dict = get_confusion_matrix(outputs=outputs, targets=targets)
            fn_total, tn_total, tp_total, fp_total = calculate_tp_fp_tn_fn(
                confusion_dict
            )

            prec1 = accuracy(outputs, targets)
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1[0].item(), inputs.shape[0])
            fn.update(fn_total)
            tn.update(tn_total)
            tp.update(tp_total)
            fp.update(fp_total)
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description(
                    "Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. ".format(
                        batch=batch_idx + 1,
                        iter=len(test_loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        top1=top1.avg,
                    )
                )
        if not args.no_progress:
            test_loader.close()

    AmfLog.info("top-1 acc: {:.2f}".format(top1.avg))
    AmfLog.info(f"true positives: {tp.sum}")
    AmfLog.info(f"true negatives: {tn.sum}")
    AmfLog.info(f"false positives: {fp.sum}")
    AmfLog.info(f"false negatives: {fn.sum}")
    return (
        losses.avg,
        top1.avg,
        tp.sum / (tp.sum + 0.5 * (fp.sum + fn.sum)),
        fp.sum,
        fn.sum,
        tp.sum,
        tn.sum,
    )
