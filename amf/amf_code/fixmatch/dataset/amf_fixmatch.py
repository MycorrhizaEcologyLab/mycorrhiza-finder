# TODO I think some of the logic in here must be broken

import logging
import math
from types import SimpleNamespace
from typing import Callable

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from torchvision import transforms

from ... import amfinder_load as AmfLoad
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
amf_mean = [
    0.6936627221601291,
    0.7870194843667774,
    0.8169391664031584,
]
amf_std = [
    0.24046182473804545,
    0.11873623743912588,
    0.07242399828535838,
]

fix_match_loader = AmfLoad.FixMatchLoader()


class TransformFixMatch(object):
    # Grabbed directly from cifar.py
    def __init__(self, mean: list[float], std: list[float], image_size: int):
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
        )  # n is number of augmentations in the Randaugment sequence, m is magnitude
        #    for each of the transformations - >M implies stronger augmentations
        self.normalize = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
        )

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class AmfDatasetLabelled:
    def __init__(
        self,
        args: SimpleNamespace,
        train_list: list[str] | None = None,
        test_list: list[str] | None = None,
        val_list: list[str] | None = None,
        n_labels: int = -1,
        transform: Callable[[NDArray[np.uint8]], NDArray[np.float32]] | None = None,
        fixed_dataset: bool = False,
        image_size: int = 252,
    ):
        """
        root (str): Root directory
        n_labels (Union[None, int, float]): indication of number of datapoints that
            should be labelled.
        n_labels < 0 indicates all labels should be used.
        n_labels 0<x<1 indicated that n_labels*100% of the data should be labelled.
        n_labels >= 1 indicated that exactly n_labels labelled pieces of data should be
            used.
        train (bool): whether the dataset should be a training set
        """
        self.train = args.train
        self.image_size = image_size
        self.train_list = train_list
        self.transform = transform
        self.fixed_dataset = fixed_dataset
        self.save_path = args.out
        self.root = args.root_path
        self.images: NDArray[np.float32] | list[NDArray[np.float32]]
        self.targets: NDArray[np.uint8] | list[np.uint8]

        if train_list is not None:
            if n_labels >= 1:
                self.images, self.targets, labelled_indices, x = (
                    fix_match_loader.load_train_dataset(train_list, n_labels, args)
                )
                self.labelled_indices = labelled_indices
                self.x = x
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

    def __getitem__(
        self, idx: int
    ) -> (
        tuple[NDArray[np.float32], np.uint8] | tuple[NDArray[np.float32], np.uint8, str]
    ):
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

    def __len__(self) -> int:
        # Find dataset size
        dataset_length = len(self.images)

        return dataset_length

    def find_optimal_layout(self, num_items: int) -> tuple[int, int]:
        """
        Find the number of rows and columns for a grid layout with num_items,
        aiming for a layout that is as square as possible.
        """
        sqrt_val = int(math.sqrt(num_items))
        if sqrt_val * sqrt_val == num_items:
            # Perfect square, equal rows and columns
            return sqrt_val, sqrt_val

        for extra in range(1, sqrt_val + 1):
            # Attempt to find factors by incrementally checking numbers larger than the
            # square root
            if num_items % (sqrt_val + extra) == 0:
                return num_items // (sqrt_val + extra), sqrt_val + extra

        # If no factors found that make a perfect rectangle (e.g., prime numbers),
        # use the closest square higher than the number of items
        nearest_square = (sqrt_val + 1) ** 2
        row, col = self.find_optimal_layout(nearest_square)
        # Adjust rows if the number of items doesn't require all rows in the nearest
        # square layout
        if num_items <= row * (col - 1):
            return row, col - 1
        return row, col


class AmfDatasetUnlabelled:
    def __init__(
        self,
        prop_unlabelled: float,
        transform: TransformFixMatch,
        labelled_indices: list[int],
        image_size: int,
        x: list[NDArray[np.float32]] | NDArray[np.float32],
    ) -> None:
        self.transform = transform
        self.image_size = image_size

        self.images, num_examples = fix_match_loader.load_unlabelled_dataset(
            labelled_indices, prop_unlabelled, x
        )

        self.targets = [torch.tensor(float("nan"))] * len(self.images)

        print(
            f"Unlabelled dataset set up with {prop_unlabelled * 100}% of the available "
            f"unlabelled data, totalling {num_examples} examples."
        )

    def __getitem__(self, idx: int):  # type: ignore[no-untyped-def]
        image = self.images[idx]
        target = self.targets[idx]

        image = np.transpose(image, (1, 2, 0))
        image_int = Image.fromarray(image.astype("uint8"), mode="RGB")

        # Add augmentation if relevant
        # TODO this can't make sense as it stands since transform returns a tuple,
        # so it is impossible to type this module correctly
        image_transformed = self.transform(image_int)
        return image_transformed, target

    def __len__(self) -> int:
        # Find dataset size
        dataset_length = len(self.images)

        return dataset_length


def get_amf(
    args: SimpleNamespace,
) -> (
    tuple[AmfDatasetLabelled, AmfDatasetUnlabelled, AmfDatasetLabelled]
    | AmfDatasetLabelled
):
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
