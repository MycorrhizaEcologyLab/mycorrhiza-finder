"""Some helper functions for PyTorch."""

import logging

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)


def get_confusion_matrix(outputs, targets, to_cpu=True):
    if to_cpu:
        _, pred = outputs.cpu().topk(1, dim=1)
        pred = pred.squeeze(1)
    else:
        pred = outputs
    return confusion_matrix(
        targets.cpu() if to_cpu else targets,
        pred.cpu() if to_cpu else pred,
        labels=list(range(6)),
    )


def get_per_class_accuracies(conf_matrix, class_names, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix", size=20)
    plt.ylabel("Actual Class", size=18)
    plt.xlabel("Predicted Class", size=18)
    plt.show()
    plt.savefig(f"{save_path}/confusion_matrix.png")

    # Calculate per-class accuracies
    per_class_accuracies = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)

    # Print per-class accuracies
    print("\nPer-Class Accuracies:")
    for class_name, accuracy in zip(class_names, per_class_accuracies):
        print(f"{class_name}: {accuracy * 100:.2f}%")

    return per_class_accuracies


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
