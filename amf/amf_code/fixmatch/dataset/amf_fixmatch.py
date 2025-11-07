import logging
import math
import os
from typing import Union

import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms

import amfinder_load as AmfLoad

from .randaugment import RandAugmentMC

logger = logging.getLogger(__name__)
CLASS_NAMES = [
    "AMColonised",
    "Uncolonised",
    "Background",
    "Unreadable",
    "DSE",
    "Hybrid",
]
# mean and std calculated based on the kew dataset not the authors
amf_mean = (
    0.6936627221601291,
    0.7870194843667774,
    0.8169391664031584,
)
amf_std = (
    0.24046182473804545,
    0.11873623743912588,
    0.07242399828535838,
)

fix_match_loader = AmfLoad.FixMatchLoader()


def get_amf(args):
    """Gets the train, test and validation datasets based on a specific file structure

    Args:
        args : arguments

    Returns:
        Pytorch Dataset Objects: train, test and validation Dataset Modules
    """

    num_labelled = args.num_labeled
    fixed_dataset = args.fixed_dataset
    image_size = args.image_size
    input_files = fix_match_loader.get_input_files(args.root_path)

    train_list = []
    val_list = []
    test_list = []

    for path in input_files:
        if "train" in path:
            train_list.append(path)
        elif "val" in path:
            val_list.append(path)
        elif "test" in path:
            test_list.append(path)
        else:
            raise ValueError(
                f"File path does not belong to a recognised category: {path}"
            )

    transform_labeled = transforms.Compose(
        [
            transforms.Resize(size=(image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(
                size=image_size, padding=int(image_size * 0.125), padding_mode="reflect"
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=amf_mean, std=amf_std),
        ]
    )
    transform_val = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=amf_mean, std=amf_std)]
    )

    if args.train:
        train_labelled_dataset = AmfDatasetLabelled(
            args=args,
            train_list=train_list,
            n_labels=num_labelled,
            transform=transform_labeled,
            fixed_dataset=fixed_dataset,
            image_size=image_size,
        )
        train_unlabelled_dataset = AmfDatasetUnlabelled(
            prop_unlabelled=args.prop_unlabelled,
            transform=TransformFixMatch(
                mean=amf_mean, std=amf_std, image_size=image_size
            ),
            labelled_indices=train_labelled_dataset.labelled_indices,
            image_size=image_size,
            x=train_labelled_dataset.x,
        )
        val_dataset = AmfDatasetLabelled(
            args=args, val_list=val_list, transform=transform_val, image_size=image_size
        )
        return train_labelled_dataset, train_unlabelled_dataset, val_dataset
    else:
        test_dataset = AmfDatasetLabelled(
            args=args,
            test_list=test_list,
            transform=transform_val,
            image_size=image_size,
        )

        return test_dataset


class TransformFixMatch(object):
    # Grabbed directly from cifar.py
    def __init__(self, mean, std, image_size):
        self.image_size = image_size
        # TODO these augmentations should be tailored to the dataset before use
        self.weak = transforms.Compose(
            [
                transforms.Resize(size=(image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(
                    size=self.image_size,
                    padding=int(self.image_size * 0.125),
                    padding_mode="reflect",
                ),
            ]
        )
        self.strong = transforms.Compose(
            [
                transforms.Resize(size=(image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(
                    size=self.image_size,
                    padding=int(self.image_size * 0.125),
                    padding_mode="reflect",
                ),
                RandAugmentMC(n=2, m=10),
            ]
        )  # n is number of augmentations in the Randaugment sequence, m is magnitude for each of the transformations - >M implies stronger augmentations
        self.normalize = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
        )

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class AmfDatasetLabelled:
    def __init__(
        self,
        args=None,
        train_list: list = None,
        test_list: list = None,
        val_list: list = None,
        n_labels: Union[int, float] = -1,
        transform=None,
        fixed_dataset=False,
        image_size=252,
    ):
        """
        root (str): Root directory
        n_labels (Union[None, int, float]): indication of number of datapoints that should be labelled.
        n_labels < 0 indicates all labels should be used.
        n_labels 0<x<1 indicated that n_labels*100% of the data should be labelled.
        n_labels >= 1 indicated that exactly n_labels labelled pieces of data should be used.
        train (bool): whether the dataset should be a training set
        """
        self.train = args.train
        self.image_size = image_size
        self.train_list = train_list
        self.transform = transform
        self.fixed_dataset = fixed_dataset
        self.save_path = args.out
        self.root = args.root_path

        if train_list is not None:
            if n_labels >= 1:
                self.images, self.targets, labelled_indices, x = (
                    fix_match_loader.load_train_dataset(self.train_list, n_labels, args)
                )
                self.labelled_indices = labelled_indices
                self.x = x
                # self.plot_labelled_images(self.images, self.targets, len(labelled_indices))
            else:
                raise ValueError("num_labels cannot be 0 for labelled set")
        elif (
            val_list is not None
        ):  # Test sets always have labels, and we use the full set
            self.images, self.targets = fix_match_loader.load_val_dataset(
                val_list, self.root
            )
        elif test_list is not None:
            self.images, self.targets, self.filenames = (
                fix_match_loader.load_test_dataset(test_list, self.root)
            )
        else:
            raise ValueError("no train or test data provided")

    def __getitem__(self, idx):
        # Get id and target
        im = self.images[idx]
        target = self.targets[idx]

        im = np.transpose(im, (1, 2, 0))

        im = Image.fromarray(im.astype("uint8"), mode="RGB")

        # Add augmentation if relevant
        if self.transform is not None:
            im = self.transform(im)

        if self.train:
            return im, target
        else:
            filename = self.filenames[idx]
            return im, target, filename

    def __len__(self):
        # Find dataset size
        dataset_length = len(self.images)

        return dataset_length

    def find_optimal_layout(self, num_items):
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
        row, col = self.find_optimal_layout(nearest_square)
        # Adjust rows if the number of items doesn't require all rows in the nearest square layout
        if num_items <= row * (col - 1):
            return row, col - 1
        return row, col

    def plot_labelled_images(self, images, labels, num_labelled):
        unique_labels = np.unique(labels)
        num_labelled_per_class = num_labelled // len(unique_labels)

        # Create results directory if it does not exist

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # Plot and save images for each class
        for label in unique_labels:
            class_images = [img for img, lbl in zip(images, labels) if lbl == label][
                :num_labelled_per_class
            ]
            num_images = len(class_images)
            # Dynamically determine the grid size
            rows, cols = self.find_optimal_layout(num_images)

            plt.figure(figsize=(cols * 4, rows * 4))
            for i, image in enumerate(class_images, start=1):
                image = image / 255
                plt.subplot(rows, cols, i)
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)
                if image.shape[-1] == 1:  # if the image is grayscale
                    plt.imshow(image.squeeze(), cmap="gray")
                else:
                    plt.imshow(image)
                plt.xlabel(CLASS_NAMES[label])
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_path, f"{CLASS_NAMES[label]}.png"))
            plt.close()


class AmfDatasetUnlabelled:
    def __init__(
        self,
        prop_unlabelled=None,
        transform=None,
        labelled_indices=None,
        image_size=252,
        x=None,
    ):
        self.transform = transform
        self.image_size = image_size

        self.images, num_examples = fix_match_loader.load_unlabelled_dataset(
            labelled_indices, prop_unlabelled, x
        )

        self.targets = [torch.tensor(float("nan"))] * len(self.images)

        print(
            f"Unlabelled dataset set up with {prop_unlabelled * 100}% of the available unlabelled data, totalling {num_examples} examples."
        )

    def __getitem__(self, idx):
        im = self.images[idx]
        target = self.targets[idx]

        im = np.transpose(im, (1, 2, 0))
        im = Image.fromarray(im.astype("uint8"), mode="RGB")

        # Add augmentation if relevant
        if self.transform is not None:
            im = self.transform(im)

        return im, target

    def __len__(self):
        # Find dataset size
        dataset_length = len(self.images)

        return dataset_length
