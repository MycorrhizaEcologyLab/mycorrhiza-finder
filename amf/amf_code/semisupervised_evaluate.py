import os
from types import SimpleNamespace

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from fixmatch.utils import (
    AverageMeter,
    accuracy,
    get_confusion_matrix,
    get_per_class_accuracies,
)

matplotlib.use("Agg")
import math

import matplotlib.pyplot as plt
import pandas as pd

import amfinder_config as AmfConfig
import amfinder_log as AmfLog
import amfinder_model as AmfModel
from fixmatch.dataset.amf_fixmatch import get_amf

# TODO note that these should be updated to mean and std of channels in dataset you are using
amf_mean = (0.6936627221601291, 0.7870194843667774, 0.8169391664031584)
amf_std = (0.24046182473804545, 0.11873623743912588, 0.07242399828535838)


def run():
    fixmatch_results_directory = AmfConfig.get("fixmatch_results_directory")
    with open(f"{fixmatch_results_directory}/config.yaml", "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    args = SimpleNamespace(**config)
    args.out = fixmatch_results_directory
    args.test_out = os.path.join(args.out, "test_results")
    if not os.path.isdir(args.test_out):
        os.mkdir(args.test_out)

    args.train = False

    if args.local_rank == -1:
        device = torch.device("cuda")
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        device = "cpu"

    args.device = device

    test_dataset = get_amf(args)

    if args.arch == "wideresnet":
        args.model_depth = 16
        args.model_width = 4
        args.num_classes = 6

    def create_model(args):
        model = AmfModel.load()
        return model.to(args.device)

    model = create_model(args)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=8,
        num_workers=AmfConfig.get("num_workers"),
    )

    checkpoint = torch.load(os.path.join(args.out, "checkpoint.pth.tar"))
    best_acc = checkpoint["best_acc"]
    AmfLog.info(f"best accuracy = {best_acc}")
    model.load_state_dict(checkpoint["state_dict"])
    test_model = model

    def find_optimal_layout(num_items):
        """
        Find the number of rows and columns for a grid layout with num_items,
        aiming for a layout that is as square as possible.
        """
        sqrt_val = int(math.sqrt(num_items))
        if sqrt_val * sqrt_val == num_items:
            # Perfect square, equal rows and columns
            return sqrt_val, sqrt_val

        for extra in range(1, sqrt_val + 1):
            # Attempt to find factors by incrementally checking numbers larger than the square root
            if num_items % (sqrt_val + extra) == 0:
                return num_items // (sqrt_val + extra), sqrt_val + extra

        # If no factors found that make a perfect rectangle (e.g., prime numbers),
        # use the closest square higher than the number of items
        nearest_square = (sqrt_val + 1) ** 2
        row, col = find_optimal_layout(nearest_square)
        # Adjust rows if the number of items doesn't require all rows in the nearest square layout
        if num_items <= row * (col - 1):
            return row, col - 1
        return row, col

    def plot_images(images, titles, out_name="test_img"):
        nrows, ncols = find_optimal_layout(len(images))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 5))
        fig.suptitle(out_name.replace("_", ""), fontsize=30)
        if nrows * ncols > 1:
            axes = axes.flatten()
        else:
            axes = [axes]

        mean = torch.tensor(amf_mean).view(3, 1, 1)
        std = torch.tensor(amf_std).view(3, 1, 1)
        for ax, img, title in zip(axes, images, titles):
            img = img * std + mean
            img = torch.clip(img, 0, 1)
            ax.imshow(img.permute(1, 2, 0).numpy())  # Convert from Tensor image
            ax.set_title(title)
            ax.axis("off")
        for ax in axes[len(images) :]:
            ax.axis("off")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"{args.test_out}/{out_name}.png")
        plt.close(fig)

    def test(args, test_loader, model):
        incorrect_samples = {}  # Dictionary to store incorrect samples by class
        losses = AverageMeter()
        top1 = AverageMeter()
        tp = AverageMeter()
        fp = AverageMeter()
        tn = AverageMeter()
        fn = AverageMeter()

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
            test_loader = tqdm(test_loader)

        target_list = []
        filename_list = []
        prediction_list = []

        with torch.no_grad():
            for batch_idx, (inputs, targets, filenames) in enumerate(test_loader):
                model.eval()
                inputs = inputs.to(args.device)
                targets = targets.to(args.device)
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, targets)
                _, predicted = torch.max(outputs, 1)
                incorrects = predicted != targets
                for i in range(inputs.size(0)):
                    if incorrects[i]:
                        label = targets[i].item()
                        if label not in incorrect_samples:
                            incorrect_samples[label] = []
                        # Store the misclassified image and its incorrect prediction
                        incorrect_samples[label].append(
                            (inputs[i].cpu(), predicted[i].item())
                        )

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
                target_list.extend(targets.cpu().tolist())
                filename_list.extend(list(filenames))
                prediction_list.extend(predicted.cpu().tolist())

            if not args.no_progress:
                test_loader.close()

        class_names = AmfConfig.get("class_names")[AmfConfig.get("colonisation_type")]
        for label, samples in incorrect_samples.items():
            images, preds = zip(
                *samples
            )  # Unzip the tuples to get the images and predictions lists
            predictions = [class_names[pred] for pred in preds]
            titles = [
                f"Label: {class_names[label]}, Pred: {predictions[i]}"
                for i in range(len(predictions))
            ]
            print(f"Class {class_names[label]} - Incorrect Predictions")
            plot_images(
                images, titles, out_name=f"{class_names[label]}_incorrect_predictions"
            )

        conf_matrix = get_confusion_matrix(
            outputs=prediction_list,
            targets=target_list,
            to_cpu=False,
        )

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
            prediction_list,
            target_list,
            filename_list,
            conf_matrix,
        )

    (
        test_loss,
        test_acc,
        f1,
        fp,
        fn,
        tp,
        tn,
        predicted_labels,
        target_labels,
        filenames,
        confusion_dict,
    ) = test(args, test_loader, test_model)

    print("num labelled", args.num_labeled)

    df_results = pd.DataFrame({"Test Accuracy": [test_acc], "Test F1-Score": [f1]})

    accuracies = {}
    class_names = AmfConfig.get("class_names")[AmfConfig.get("colonisation_type")]
    per_class_accuracies = get_per_class_accuracies(
        confusion_dict, class_names, args.test_out
    )
    for i, name in enumerate(class_names[:-1]):
        accuracies[name] = per_class_accuracies[i] * 100
    for key, value in accuracies.items():
        df_results[f"{key} Accuracy"] = value
    df_results.to_csv(f"{args.test_out}/test_metrics.csv", index=False)
    df = get_perfile_metrics(target_labels, predicted_labels, filenames)
    df.to_csv(f"{args.test_out}/file_metrics.csv")


def get_perfile_metrics(actual_labels, predicted_labels, filenames):
    split_files = [
        filename.split("/")[-1].split("_Default_Extended")[0] for filename in filenames
    ]

    data = {
        "Filename": split_files,
        "Actual Label": actual_labels,
        "Predicted Label": predicted_labels,
        # Include other relevant data as needed
    }
    df = pd.DataFrame(data)

    def calculate_colonized_percentage(series):
        """Calculate % colonized for a series of labels."""
        colonized = series.value_counts()
        total_root = colonized.get(0, 0) + colonized.get(1, 0)
        perc_col = (colonized.get(0, 0) / total_root) * 100 if total_root > 0 else None
        # perc_root = (total_root / colonized.sum()) * 100 if colonized.sum() > 0 else None
        return perc_col

    def calculate_accuracy_for_class(group, class_label):
        """Calculate accuracy for a specific class within a grouped object."""
        correct_predictions = (group["Actual Label"] == group["Predicted Label"]) & (
            group["Actual Label"] == class_label
        )
        total_instances = group["Actual Label"] == class_label
        if total_instances.sum() == 0:  # Avoid division by zero
            return None
        return (correct_predictions.sum() / total_instances.sum()) * 100

    def calculate_class_count(group, class_label):
        # classes = [0,1,2]
        # classes.remove(class_label)
        num_actual = (group["Actual Label"] == class_label).sum()
        num_pred = (group["Predicted Label"] == class_label).sum()

        return num_actual, num_pred

    # Calculate % colonized for actual and predicted labels
    colonization_metrics = df.groupby("Filename").apply(
        lambda x: pd.Series(
            {
                "Actual % Colonized": calculate_colonized_percentage(x["Actual Label"]),
                "Predicted % Colonized": calculate_colonized_percentage(
                    x["Predicted Label"]
                ),
            }
        )
    )

    # Calculate per-class accuracy for each filename
    class_names = AmfConfig.get("class_names")[AmfConfig.get("colonisation_type")]
    class_accuracies = df.groupby("Filename").apply(
        lambda group: pd.Series(
            {
                f"Accuracy Class {class_names[0]}": calculate_accuracy_for_class(
                    group, 0
                ),
                f"Accuracy Class {class_names[1]}": calculate_accuracy_for_class(
                    group, 1
                ),
            }
        )
    )

    gt_class_quantities = df.groupby("Filename").apply(
        lambda group: pd.Series(
            {
                **{
                    f"Num Actual {class_name}": calculate_class_count(group, i)[0]
                    for i, class_name in enumerate(class_names)
                },
                **{
                    f"Num Pred {class_name}": calculate_class_count(group, i)[1]
                    for i, class_name in enumerate(class_names)
                },
            }
        )
    )

    # Combine both metrics into a single DataFrame
    file_metrics = colonization_metrics.join(class_accuracies).join(gt_class_quantities)
    file_metrics_with_index = file_metrics.reset_index()
    file_metrics_with_index.rename(columns={"index": "Filename"}, inplace=True)

    return file_metrics_with_index
