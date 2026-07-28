import glob
import io
import json
import os
import random
import re
import sys
from collections import defaultdict
from typing import cast

import h5py
import numpy as np
import pandas as pd

# Torch functionalities
import torch
from loguru import logger
from numpy.typing import ArrayLike, NDArray
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

import amf.helper.config as AmfConfig
import amf.helper.segmentation as AmfSegm
from amf.helper.api_utils import (
    check_entries_for_id,
    download_entries_as_csv,
    get_enabled,
    get_tile_edge,
)
from amf.helper.db_config import connect

FLUSH_EVERY = 4096  # tiles buffered in RAM before a write (~60 MB at 15 KB/tile)


class CustomNormalisedDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """
    Manages dataset loading and normalizes image data to the range [0, 1].
    This is intended to be used by DataLoaders primarily for efficient memory handling.

    Expected Inputs:
    x: Iterable of image data (e.g., list of tiles as NumPy arrays or as simple list),
        or row indices into an HDF5 tile file if dynamic loading is enabled.
    y: Iterable of labels corresponding to the image data (e.g., list of NumPy arrays or
        lists representing the labels).
    h5_path: Path to the HDF5 tile file, or to a directory containing exactly one.
        Required when dynamic loading is enabled, ignored otherwise.

    Methods:
    __init__(self, x, y, transform, h5_path): Initializes the dataset.
    __len__(self): Returns the number of samples in the dataset.
    __getitem__(self, index): Returns a tuple of image and label at the given index as
        PyTorch tensors, with the image normalized to the range [0, 1].

    Expected Outputs:
    Returns normalized data as PyTorch tensors with on-demand data conversion.
    x_tensor are the image tiles which are normalised and converted into torch tensors
        whenever __getitem__() is invoked.
    y_tensor are one_hot_encoded labels which are converted into torch tensors whenever
        __getitem__() is invoked.
    """

    def __init__(
        self,
        x: ArrayLike,
        y: ArrayLike,
        transform=None,
        h5_path: str | None = None,
    ):
        # Store data as NumPy arrays or lists
        self.dynamic_loading = AmfConfig.get("dynamic_loading")
        self.h5: h5py.File | None = None
        self.pid: int | None = None

        if transform is not None:
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ColorJitter(
                        brightness=(0.75, 1.25), saturation=(0.75, 1.25)
                    ),
                    transforms.RandomApply(
                        [
                            transforms.RandomResizedCrop(
                                (
                                    AmfConfig.get("tile_edge"),
                                    AmfConfig.get("tile_edge"),
                                ),
                                scale=(0.9, 0.9),
                                ratio=(1.0, 1.0),
                                interpolation=InterpolationMode.BILINEAR,
                            ),
                            transforms.Resize(
                                (AmfConfig.get("tile_edge"), AmfConfig.get("tile_edge"))
                            ),
                        ],
                        p=0.3,
                    ),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.transform = None

        if self.dynamic_loading:
            if h5_path is None:
                logger.error(
                    "dynamic_loading is enabled but no tile dataset path was provided."
                )
                sys.exit(32)
            self.h5_path = h5_path
            self.x = np.asarray(x, dtype=np.int64)  # HDF5 row numbers
        else:
            self.h5_path = None
            self.x = np.array(x) if not isinstance(x, np.ndarray) else x

        self.y = np.array(y) if not isinstance(y, np.ndarray) else y

    def _file(self) -> h5py.File:
        """
        Return a handle owned by the current process.
        """
        pid = os.getpid()
        if self.h5 is None or self.pid != pid:
            self.h5 = h5py.File(self.h5_path, "r")
            self.pid = pid
        return self.h5

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["h5"] = None
        state["pid"] = None
        return state

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Convert to torch tensors only when accessed
        if self.dynamic_loading:
            # Load tile image from the HDF5 file
            row = int(self.x[index])
            raw = self._file()["jpg"][row].tobytes()
            tile = np.array(Image.open(io.BytesIO(raw)), dtype=np.uint8)
            tile = np.transpose(tile.astype(np.uint8), (2, 0, 1))  # Convert to CxHxW
            x_tensor = cast(
                torch.Tensor, torch.tensor(tile, dtype=torch.float32) / 255.0
            )  # Normalisation to [0, 1]
        else:
            x_tensor = cast(
                torch.Tensor, torch.tensor(self.x[index], dtype=torch.float32) / 255.0
            )  # Normalisation to [0, 1]

        if self.transform:
            x_tensor = self.transform(x_tensor)
        if AmfConfig.get("resize_dim") is not None:
            resize_transform = transforms.Resize(
                (AmfConfig.get("resize_dim"), AmfConfig.get("resize_dim"))
            )
            x_tensor = resize_transform(x_tensor)
        y_tensor = torch.tensor(self.y[index], dtype=torch.float32)
        return x_tensor, y_tensor


class TileDatasetLoader(Dataset[tuple[NDArray[np.uint8], NDArray[np.uint8]]]):
    """
    Handles loading of training tile sets and annotations, with optional data
    augmentation and dataset balancing.

    Expected Inputs:
    input_files: List of file paths to the images used for training.
    use_augmentation: Boolean indicating whether to apply data augmentation.
    balance_factor: Float multiplier for balancing datasets based on class sizes and
        oriented on Colonised classes.
    calculate_distribution: Boolean flag to calculate class distribution statistics.

    Methods and Workflow:
    __init__(self, input_files, use_augmentation=False, balance_factor=1.0,
             calculate_distribution=False):
    Initializes the class with input images, configurations, and optional parameters for
    augmentation and balancing.
    Calls _prepare_dataset to load and augment data if necessary. Optionally computes
    class distribution statistics via _prepare_stats.

    _prepare_dataset(input_files, use_augmentation):
    Iterates through each input file, extracting image tiles and annotations. Normalizes
    image tiles by scaling pixel values to [0, 1].
    Optionally applies data augmentation using specified transformations.

        if use_augmentation is set to true:
        Utilizes a set of transformations (random flips, color jitter, etc.) to augment
        specified classes multiple times.
        For these operations tiles and labels are temporarilty converted to torch
        tensors. Balanced representation is ensured through
        class_augmentations, inidicating the number of augmentation loops per given
        class in a dictionary.

    _balance_dataset(self, x, y, factor=1.0):
    Balances classes by clipping larger clases to the number of tiles obtained from
    smaller classes. Smaller classes are retrained fully.

    factor (float): The factor to determine the clipping size for classes larger than
                    class 0.
    - If factor = 1.0, classes are clipped to match class 0's size.
    - If factor > 1.0, classes are clipped to `factor` times class 0's size.

    Notes:
    - Class 0 (AM Colonised) is treated as the reference class.
    - Classes smaller than class 0 are fully retained.
    - Classes larger than class 0 are clipped to `factor` times class 0's size, but not
        exceeding the class size.

    _prepare_stats(self):
    Calculates and prints class distribution statistics, as a check and overview that
    all classes are present within a given dataset.

    __len__(self):
    Returns the number of samples in the dataset.

    __getitem__(self, idx):
    Retrieves data samples and their labels from the dataset, ensuring that images are
    appropriately scaled.

    Expected Outputs
    dataset = List of tuples containing pairs of image tiles and labels.

    The loader facilitates the generation of a dataset from input images and
    annotations, outputting:
    Tile Images: Image tiles processed with optional augmentations and balacing,
        converted into uint8 numpy arrays and stored in a list.
    Labels: Corresponding one-hot encoded labels for each tile, converted into uint8
        numpy arrays and stored in an according list.

    """

    def __init__(
        self,
        input_files: list[str],
        use_augmentation: bool = False,
        balance_factor: float = 1.0,
        calculate_distribution: bool = False,
    ):
        self.use_augmentation = use_augmentation
        self.balance_factor = balance_factor
        self.dataset = self._prepare_dataset(input_files, self.use_augmentation)
        self.labels = [label for _, label in self.dataset]  # Ensuring labels are stored
        self.calculate_distribution = calculate_distribution

        # Calculate statistics if needed
        if self.calculate_distribution:
            logger.info(f"Total images for train/val split: {len(input_files)}")
            self._prepare_stats()

    def _prepare_dataset(
        self, input_files: list[str], use_augmentation: bool
    ) -> list[tuple[NDArray[np.uint8], NDArray[np.uint8]]]:
        all_tiles = []
        all_labels = []
        # Integration of augmentation
        tile_edge = AmfConfig.get("tile_edge")
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(
                    brightness=(0.75, 1.25), saturation=(0.75, 1.25)
                ),
                transforms.RandomApply(
                    [
                        transforms.RandomResizedCrop(
                            (tile_edge, tile_edge),
                            scale=(0.9, 0.9),
                            ratio=(1.0, 1.0),
                            interpolation=InterpolationMode.BILINEAR,
                        ),
                        transforms.Resize((tile_edge, tile_edge)),
                    ],
                    p=0.3,
                ),
                transforms.ToTensor(),
            ]
        )

        colonisation_type = AmfConfig.get("colonisation_type")
        if colonisation_type == "am":
            class_augmentations = {
                0: 2,
                1: 1,
                3: 3,
                4: 3,
                5: 3,
            }  # Example augmentation setup for AM
        else:
            class_augmentations = {
                0: 2,
                1: 2,
                2: 2,
                3: 1,
                6: 3,
                7: 3,
                8: 3,
                9: 3,
            }  # Example augmentation setup for ErM

        for path in input_files:
            annotations = import_annotations(path)
            settings = import_settings(path)
            if annotations is not None:
                image = AmfSegm.load(path)
                for annot in annotations.itertuples():
                    # Grab tile and label
                    tile = AmfSegm.tile(
                        image, annot.row, annot.col, settings["tile_edge"]
                    )
                    label = np.array(annot[3:], dtype=np.uint8)

                    # Apply augmentation during dataset preparation if specified
                    if use_augmentation:
                        # Convert tile and label
                        tile = cast(
                            torch.Tensor,
                            torch.tensor(tile, dtype=torch.float32) / 255.0,
                        )  # Normalize images
                        label = torch.tensor(list(annot[3:]), dtype=torch.float32)

                        # Run augmentation
                        label_class = torch.argmax(label).item()
                        augment_count = class_augmentations.get(label_class, 0)
                        for _ in range(augment_count):
                            augmented_tile = transform(tile)
                            # Resize again if specified in config, as some augmentations may change the tile size
                            if AmfConfig.get("resize_dim") is not None:
                                resize_transform = transforms.Resize(
                                    (
                                        AmfConfig.get("resize_dim"),
                                        AmfConfig.get("resize_dim"),
                                    )
                                )
                                tile = resize_transform(tile)
                            # Convert back to uint8 to stay memory efficient
                            augmented_tile = np.clip(
                                (augmented_tile.numpy() * 255).round(), 0, 255
                            ).astype(np.uint8)
                            augmented_label = label.numpy().astype(np.uint8)
                            # Append augmented tiles and labels to dataset
                            all_tiles.append(augmented_tile)
                            all_labels.append(augmented_label)

                        tile = np.clip((tile.numpy() * 255).round(), 0, 255).astype(
                            np.uint8
                        )

                        label = label.numpy().astype(np.uint8)

                    # Also resize original tile if specified in config, to ensure consistency with augmented tiles
                    if AmfConfig.get("resize_dim") is not None:
                        resize_transform = transforms.Resize(
                            (
                                AmfConfig.get("resize_dim"),
                                AmfConfig.get("resize_dim"),
                            )
                        )
                        tile = resize_transform(tile)
                    all_tiles.append(tile)
                    all_labels.append(label)  # Always add original
                del image

        if len(all_tiles) == 0:
            logger.error("None of the training images had annotations, so fail")
            # ERR_MISSING_ANNOTATIONS = 32
            sys.exit(32)

        if self.balance_factor != 0.0:
            # Apply balancing
            balanced_tiles, balanced_labels = self._balance_dataset(
                all_tiles, all_labels, self.balance_factor
            )
            # Create balanced dataset
            logger.info(
                f"Balance factor set to: {self.balance_factor}. "
                f"Generating balanced dataset."
            )
            dataset = list(zip(balanced_tiles, balanced_labels))
        elif self.balance_factor == 0.0:
            # Create unbalanced dataset
            logger.info(
                f"Balance factor set to: {self.balance_factor}. "
                f"Generating unbalanced dataset."
            )
            dataset = list(zip(all_tiles, all_labels))

        return dataset

    def _balance_dataset(
        self,
        x: list[NDArray[np.uint8]],
        y: list[NDArray[np.uint8]],
        factor: float = 1.0,
    ) -> tuple[
        list[NDArray[np.uint8]], list[NDArray[np.uint8]]
    ]:  # factor is a hyperparameter
        # Count the number of samples in class 0 (assumes one-hot encoding)
        class_0_samples = sum(array[0] for array in y)

        # Initialize a list to keep track of indices to retain
        indices_to_keep: list[int] = []

        # Convert the one-hot encoded labels to class indices
        y_indices = np.argmax(y, axis=1)

        colonisation_type = AmfConfig.get("colonisation_type")

        # Retrieve the number of classes based on configuration
        num_classes = len(AmfConfig.get("class_names")[colonisation_type])

        for class_id in range(num_classes):
            # Get the indices of samples belonging to the current class
            idx_class_samples = np.where(y_indices == class_id)[0]
            num_class_samples = len(idx_class_samples)

            if class_id == 0:
                # Retain all samples from class 0
                indices_to_keep.extend(idx_class_samples)
            else:
                if num_class_samples == 0:
                    # Skip if no samples are available for the current class
                    continue

                # Clip samples to the minimum of `factor * class_0_samples` or the class
                # size
                # Factor = 1.0: Same size as AM Colonised.
                # Factor > 1.0: Other classes are becoming larger than AM Colonised.
                # Factor < 1.0: AM Colonised is the biggest class.
                num_samples_to_select = min(
                    int(class_0_samples * factor), num_class_samples
                )

                # Select a subset of indices if needed
                if num_samples_to_select < num_class_samples:
                    idx_selected_samples = random.sample(
                        list(idx_class_samples), num_samples_to_select
                    )
                    indices_to_keep.extend(idx_selected_samples)
                else:
                    # Retain all samples if the number to select exceeds or equals the
                    # class size
                    indices_to_keep.extend(idx_class_samples)

        # Update the input data (x) and labels (y) to retain only the selected indices
        indices_to_keep = sorted(indices_to_keep)  # Ensure indices are in order
        x = [x[i] for i in indices_to_keep]
        y = [y[i] for i in indices_to_keep]

        return x, y

    def _prepare_stats(self) -> None:
        # Collect labels for statistics
        samples = np.stack([self.dataset[i][1] for i in range(len(self.dataset))])
        y = torch.from_numpy(samples)
        y = y.type(torch.float32)
        class_counts = y.sum(dim=0)
        total_samples = y.size(0)
        percentages = (class_counts / total_samples) * 100

        logger.debug("Class Distribution before train/val split:")
        for i, count in enumerate(class_counts):
            logger.debug(f"Class {i}: {count} samples ({percentages[i]:.2f}%)")

        uniqueargs_probe_hot_indexes = (
            (class_counts > 0).nonzero(as_tuple=True)[0].numpy()
        )
        uniqueargs_headers = np.arange(len(class_counts))

        if not np.array_equal(uniqueargs_probe_hot_indexes, uniqueargs_headers):
            logger.error(
                "Training data does not represent all classes. Please reconsider "
                "training dataset curation."
            )
            # ERR_NO_DATA = 10
            sys.exit(10)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
        tile, label = self.dataset[idx]
        return tile, label


class PreTiledHDF5Loader(Dataset[tuple[NDArray[np.uint8], NDArray[np.uint8]]]):
    def __init__(
        self,
        h5_path: str,
        use_augmentation: bool = False,
        balance_factor: float = 1.0,
        calculate_distribution: bool = False,
    ):
        self.h5_path = h5_path
        self.use_augmentation = use_augmentation
        self.h5: h5py.File | None = None  # opened lazily, per worker process
        self.pid: int | None = None

        colonisation_type = AmfConfig.get("colonisation_type")
        self.class_names = AmfConfig.get("class_names")[colonisation_type]
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.eye = np.eye(len(self.class_names), dtype=np.uint8)

        tile_edge = AmfConfig.get("tile_edge")

        if not os.path.isfile(self.h5_path):
            h5_files = glob.glob(os.path.join(self.h5_path, "*.h5"))
            if not h5_files:
                logger.error(f"No image tile dataset file found at '{self.h5_path}'.")
                sys.exit(32)
            elif len(h5_files) > 1:
                logger.error(
                    f"Multiple HDF5 files found in '{self.h5_path}'. Please specify a single "
                    f"file path."
                )
                sys.exit(32)
            self.h5_path = h5_files[0]

        with h5py.File(self.h5_path, "r") as f:
            if "jpg" not in f:
                logger.error(f"'{self.h5_path}' contains no tile datasets.")
                sys.exit(32)

            file_classes = [
                s.decode() if isinstance(s, bytes) else str(s)
                for s in f.attrs["class_names"]
            ]
            if file_classes != list(self.class_names):
                logger.error(
                    f"Class mismatch: file has {file_classes}, "
                    f"config expects {list(self.class_names)}. Aborting."
                )
                sys.exit(33)

            file_edge = int(f.attrs["tile_edge"]) if "tile_edge" in f.attrs else None

            # Tiles past the commit point belong to a source image that was only
            # part-processed before an interrupted run; they are not valid data.
            n = int(f.attrs["n_committed"])
            if n == 0:
                logger.error("No tiles found in HDF5 file, aborting.")
                sys.exit(32)

            self.cls = f["cls"][:n]

        if file_edge is not None and file_edge != int(tile_edge):
            logger.warning(
                f"File was tiled at edge {file_edge}, config says {tile_edge}. "
                f"Tiles will not match the expected input size."
            )

        self.sample_index = np.arange(n, dtype=np.int64)

        if balance_factor != 0.0:
            self.sample_index = self._balance_dataset(self.sample_index, balance_factor)
            logger.info(
                f"Balance factor set to: {balance_factor}. Generating balanced dataset."
            )
        else:
            logger.info(
                f"Balance factor set to: {balance_factor}. "
                f"Generating unbalanced dataset."
            )

        # One-hot labels for the balanced sample set, in dataset order.
        self.labels = self.eye[self.cls[self.sample_index]]

        if calculate_distribution:
            logger.info(f"Total tiles after balancing: {len(self.sample_index)}")
            self._prepare_stats()

        # Pipeline takes a PIL image and returns a CHW float tensor in [0, 1].
        self.transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomChoice(
                    [
                        transforms.RandomRotation((0, 0)),
                        transforms.RandomRotation((90, 90)),
                        transforms.RandomRotation((180, 180)),
                        transforms.RandomRotation((270, 270)),
                    ]
                ),
                transforms.ColorJitter(
                    brightness=(0.75, 1.25), saturation=(0.75, 1.25)
                ),
                transforms.RandomApply(
                    [
                        transforms.RandomResizedCrop(
                            (tile_edge, tile_edge),
                            scale=(0.9, 0.9),
                            ratio=(1.0, 1.0),
                            interpolation=InterpolationMode.BILINEAR,
                        ),
                    ],
                    p=0.3,
                ),
                transforms.ToTensor(),
            ]
        )

    def _file(self) -> h5py.File:
        """
        Return a handle owned by the current process.
        """
        pid = os.getpid()
        if self.h5 is None or self.pid != pid:
            self.h5 = h5py.File(self.h5_path, "r")
            self.pid = pid
        return self.h5

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["h5"] = None
        state["pid"] = None
        return state

    def _balance_dataset(
        self, index: NDArray[np.int64], factor: float = 1.0
    ) -> NDArray[np.int64]:
        y = self.cls[index]
        class_0_samples = int((y == 0).sum())
        keep: list[NDArray[np.int64]] = []

        for class_id in range(len(self.class_names)):
            idx_class_samples = index[y == class_id]
            num_class_samples = len(idx_class_samples)

            if class_id == 0:
                keep.append(idx_class_samples)
                continue
            if num_class_samples == 0:
                continue

            num_samples_to_select = min(
                int(class_0_samples * factor), num_class_samples
            )
            if num_samples_to_select < num_class_samples:
                keep.append(
                    np.random.choice(
                        idx_class_samples, num_samples_to_select, replace=False
                    )
                )
            else:
                keep.append(idx_class_samples)

        return np.sort(np.concatenate(keep))

    def _prepare_stats(self) -> None:
        y = self.cls[self.sample_index]
        counts = np.bincount(y, minlength=len(self.class_names))
        total = int(counts.sum())

        logger.debug("Class distribution after balancing:")
        for i, name in enumerate(self.class_names):
            logger.debug(
                f"  {name}: {int(counts[i])} samples ({counts[i] / total * 100:.2f}%)"
            )

        if (counts == 0).any():
            logger.error(
                "Training data does not represent all classes. Please reconsider "
                "training dataset curation."
            )
            sys.exit(10)

    def get_split_arrays(self) -> tuple[NDArray[np.int64], NDArray[np.uint8]]:
        return self.sample_index.copy(), self.labels

    def __getitem__(self, idx: int) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
        real = int(self.sample_index[idx])
        f = self._file()

        img = Image.open(io.BytesIO(f["jpg"][real].tobytes()))
        label = self.eye[self.cls[real]].copy()

        if self.use_augmentation:
            # ToTensor() already yields CHW float in [0, 1].
            tile_tensor = cast(torch.Tensor, self.transform(img))
            tile = np.clip((tile_tensor.numpy() * 255).round(), 0, 255).astype(np.uint8)
        else:
            tile = np.transpose(np.array(img, dtype=np.uint8), (2, 0, 1))  # HWC -> CHW

        return tile, label

    def __len__(self) -> int:
        return len(self.sample_index)

    def get_tile_meta(self, idx: int) -> tuple[str, int, int]:
        """Source image name, row, col — for mapping predictions back."""
        real = int(self.sample_index[idx])
        f = self._file()
        src = f["source"][real]
        return (
            src.decode() if isinstance(src, bytes) else str(src),
            int(f["row"][real]),
            int(f["col"][real]),
        )


class StratifiedDatasetSplitter:
    """
    Designed to split a dataset into training and validation sets while maintaining the
    distribution of class labels.
    This is particularly useful in imbalanced datasets where ensuring the proportional
    representation of classes in both training and validation sets improves model
    evaluation and training performance.
    Efficient data handling using lists and NumPy ensures that the class can manage
    datasets of varying size and complexity without significant performance degradation.

    Expected Inputs:
    x: Iterable of image data (e.g., list of tiles as NumPy arrays or as simple list).
    y: Iterable of labels corresponding to the image data (e.g., list of NumPy arrays or
        lists representing the labels).
    val_size (float): Fraction of the dataset to allocate to the validation set. By
        default, this is set to 0.2 (20% validation).
    random_state (int): Seed used for random number generation to ensure reproducibility
        of data splits. Default is 42.

    Methods and Workflow:
    __init__(self, x, y, val_size=0.2, random_state=42):
    Takes input features and labels, along with parameters for validation size and
    random state.
    Calls _split_data to perform the stratified split.

    _split_data(self, val_size, random_state):
    Converts one-hot encoded labels to single integer indices using NumPy operations for
    efficient computation, including unique labels.
    Subsequently, organises indices of samples into a dictionary keyed by class labels,
    creating a mapping of data sample locations.
    For each class label:
    - Collects all indices associated with that class.
    - Determines the split index based on the specified val_size.
    - Shuffles indices within each class to ensure randomness in split selection.
    - Accumulates indices for training and validation splits based on this shuffled
        order.

    _get_dataset(self, dataset_type="train"):
    Provides access to data splits by returning features and labels for either the
    training or validation set, based on the input parameter.
    IMPORTANT: The datasets are returned in their original format, i. e. lists of numpy
    arrays respectively. Torch tensors are not used.

    Expected Outputs:
    x_train, y_train: Lists of image tiles and one-hot-encoded labels for training,
        stratified to mirror overall class distribution.
    x_val, y_val: Lists of image tiles and one-hot-encoded labels for validation,
        stratified to mirror overall class distribution.
    """

    def __init__(
        self,
        x: list[NDArray[np.uint8]],
        y: list[NDArray[np.uint8]],
        val_size: float = 0.2,
        random_state: int = 42,
    ):
        self.x = x  # Ensure x is a numpy array
        self.y = y  # Ensure y is a numpy array

        # Perform stratified split while keeping the split index structures internal
        self._split_data(val_size, random_state)

    def _split_data(self, val_size: float, random_state: int) -> None:
        # Set random seed for reproducibility
        random.seed(random_state)

        # Use NumPy to convert one-hot encoded labels to single integer labels
        y_np = np.array(self.y)
        y_labels = np.argmax(y_np, axis=1).tolist()  # Convert NumPy array to list

        # Use NumPy to get unique class labels
        unique_labels = np.unique(y_labels).tolist()  # Convert NumPy array to list

        # Create a dictionary to hold the indices of each class
        label_indices = defaultdict(list)
        for idx, label in enumerate(y_labels):
            label_indices[label].append(idx)

        # Prepare lists to store train and validation indices
        train_indices = []
        val_indices = []

        # Perform stratified sampling for each label
        for label in unique_labels:
            indices = label_indices[label]
            num_samples = len(indices)

            # Calculate the index to split the data based on val_size
            split_idx = int(num_samples * (1 - val_size))

            # Shuffle the indices for randomness
            shuffled_indices = list(indices)
            random.shuffle(shuffled_indices)

            # Split the indices into training and validation sets
            train_indices.extend(shuffled_indices[:split_idx])
            val_indices.extend(shuffled_indices[split_idx:])

        self.train_indices = train_indices
        self.val_indices = val_indices

    def _get_dataset(
        self, dataset_type: str = "train"
    ) -> tuple[
        list[NDArray[np.uint8]], list[NDArray[np.uint8]]
    ]:  # TODO should use an enum
        if dataset_type == "train":
            x_train = [self.x[i] for i in self.train_indices]
            y_train = [self.y[i] for i in self.train_indices]
            return x_train, y_train
        elif dataset_type == "val":
            x_val = [self.x[i] for i in self.val_indices]
            y_val = [self.y[i] for i in self.val_indices]
            return x_val, y_val
        else:
            raise ValueError("Invalid dataset type. Choose 'train' or 'val'.")


class TileFilesandData(
    Dataset[
        tuple[
            NDArray[np.uint8],
            NDArray[np.uint8],
            str,
            int,
            int,
        ]
    ]
):
    """
    Responsible for loading image tiles and their associated annotations from a dataset.
    This class effectively handles the extraction and preparation of data needed for
    machine learning tasks, specifically image-based analysis, by organizing and
    managing the annotations and image data efficiently.

    Expected Inputs:
    input_files: List of file paths to the images used for training. These images should
        ideally have corresponding annotation files or data.

    Methods and Workflow:
    __init__(self, input_files):
    Uses input_files as input and calls _load_and_process_data method to aggregate image
    tiles, annotations, and related metadata, filtering out entries without annotations.

    _load_and_process_data(self):
    Loads annotations and settings for each provided image file unsing
    import_annotations and import_settings. Filters the image files to ensure only those
    with annotations are processed.
    Calls _process_dataset to process each image and extract tiles and metadata.

    _process_dataset(self, dataset):
    Iterates through a filtered datasetimage file, accessing configurations and
    annotations. Tiles are extracted based on annotations.
    Collects tile data, labels, file names, and their row/column position data into
    lists and compiles them into a tuple containing tiles, labels, filenames,
    row indices, and column indices.

    Expected Outputs:
    x (torch.FloatTensor): Normalized tiles as a PyTorch tensor, with pixel values
        scaled to [0, 1].
    y (torch.FloatTensor): One-hot encoded annotations corresponding to the tiles.
    file_names (list): List of file names for each processed tile.
    rows (list): Row indices for each extracted tile.
    cols (list): Column indices for each extracted tile.
    """

    def __init__(self, input_files: list[str], is_bald_folder: bool = False):
        self.input_files = input_files
        if AmfConfig.get("dynamic_loading"):
            pretiled_dir = AmfConfig.get("pretiled_dir")
            preprocess_to_tiles(input_files, pretiled_dir)
            self.x, self.y, self.file_names, self.rows, self.cols = get_tile_metadata(
                pretiled_dir
            )
        else:
            self.x, self.y, self.file_names, self.rows, self.cols = (
                self._load_and_process_data(is_bald_folder)  # type: ignore[misc] # TODO fix better
            )
        if is_bald_folder:
            (
                self.x_unlabelled,
                self.file_names_unlabelled,
                self.rows_unlabelled,
                self.cols_unlabelled,
            ) = self._load_and_process_data(is_bald_folder, is_unlabelled=True)  # type: ignore[misc] # TODO fix better

        else:
            self.x_unlabelled = None
            self.file_names_unlabelled = None
            self.rows_unlabelled = None
            self.cols_unlabelled = None

    def _load_and_process_data(
        self, is_bald_folder: bool = False, is_unlabelled: bool = False
    ) -> (
        tuple[
            list[NDArray[np.uint8]],
            list[NDArray[np.uint8]],
            list[str],
            list[int],
            list[int],
        ]
        | tuple[
            list[NDArray[np.uint8]],
            list[str],
            list[int],
            list[int],
        ]
    ):
        logger.info("Tile extraction.")

        # Load image settings and annotations.
        annotations = [
            import_annotations(path, is_bald_folder) for path in self.input_files
        ]
        settings = [import_settings(path) for path in self.input_files]

        dataset = zip(self.input_files, settings, annotations)

        if is_unlabelled:
            dataset_list = list(dataset)
            logger.info(f"{len(dataset_list)} images in BALD dataset.")

            # Process and normalize dataset
            x, file_names, rows, cols = self._process_dataset_unlabelled(dataset_list)

            return x, file_names, rows, cols

        else:
            filtered_dataset = [x for x in dataset if x[2] is not None]
            if len(filtered_dataset) == 0 and not is_bald_folder:
                logger.error(
                    "Input images do not contain tile annotations. "
                    "Use amfbrowser to annotate tiles before training."
                )
                # ERR_NO_DATA = 10
                sys.exit(10)

            logger.info(f"{len(filtered_dataset)} images in filtered dataset.")

            # Process and normalize dataset
            x, y, file_names, rows, cols = self._process_dataset(filtered_dataset)

            return x, y, file_names, rows, cols

    def _process_dataset(
        self, dataset: list[tuple[str, dict[str, int], pd.DataFrame]]
    ) -> tuple[
        list[NDArray[np.uint8]],
        list[NDArray[np.uint8]],
        list[str],
        list[int],
        list[int],
    ]:
        tiles = []
        hot_labels = []
        file_names = []
        rows = []
        cols = []

        for path, config, annots in dataset:
            edge = config["tile_edge"]
            AmfConfig.set_("tile_edge", edge)

            image = AmfSegm.load(path)
            for annot in annots.itertuples():
                if AmfConfig.get("dynamic_loading"):
                    # Store file path for dynamic loading
                    tile = path  # Save path, actual tile will be loaded in __getitem__
                else:
                    tile = AmfSegm.tile(image, annot.row, annot.col)
                    tile = np.array(tile, dtype=np.uint8)
                label = np.array(annot[3:], dtype=np.uint8)
                rows.append(annot.row)
                cols.append(annot.col)
                tiles.append(tile)
                hot_labels.append(label)
                file_names.append(path)

            del image

        return (
            tiles,
            hot_labels,
            file_names,
            rows,
            cols,
        )

    def _process_dataset_unlabelled(
        self, dataset: list[tuple[str, dict[str, int], pd.DataFrame]]
    ) -> tuple[
        list[NDArray[np.uint8]],
        list[str],
        list[int],
        list[int],
    ]:
        tiles = []
        file_names = []
        rows = []
        cols = []
        for path, config, annots in dataset:
            edge = config["tile_edge"]
            AmfConfig.set_("tile_edge", edge)
            image = AmfSegm.load(path)
            width, height = image.size
            nrows = int(height // edge)
            ncols = int(width // edge)

            if annots is None:
                for r in range(nrows):
                    for c in range(ncols):
                        tile = AmfSegm.tile(image, r, c)
                        tile = np.array(tile, dtype=np.uint8)
                        rows.append(r)
                        cols.append(c)
                        tiles.append(tile)
                        file_names.append(path)

            else:
                labelled_coords = set(zip(annots.row, annots.col))
                for r in range(nrows):
                    for c in range(ncols):
                        if (r, c) not in labelled_coords:
                            tile = AmfSegm.tile(image, r, c)
                            tile = np.array(tile, dtype=np.uint8)
                            rows.append(r)
                            cols.append(c)
                            tiles.append(tile)
                            file_names.append(path)

            del image

        return (
            tiles,
            file_names,
            rows,
            cols,
        )

    def get_all_data(
        self,
    ) -> tuple[
        list[NDArray[np.uint8]],
        list[NDArray[np.uint8]],
        list[str],
        list[int],
        list[int],
    ]:
        """Returns all processed data from the dataset."""
        return self.x, self.y, self.file_names, self.rows, self.cols

    def get_all_unlabelled_data(
        self,
    ) -> tuple[list[NDArray[np.uint8]], list[str], list[int], list[int]]:
        """Returns all unlabelled data from the dataset."""
        return (
            self.x_unlabelled,
            self.file_names_unlabelled,
            self.rows_unlabelled,
            self.cols_unlabelled,
        )

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(
        self, idx: int
    ) -> tuple[
        NDArray[np.uint8],
        NDArray[np.uint8],
        str,
        int,
        int,
    ]:
        # Validate index
        if idx >= len(self.x) or idx < 0:
            raise IndexError("Index out of bounds")

        # Get the item at the specified index
        tile = self.x[idx]
        label = self.y[idx]
        file_name = self.file_names[idx]
        row = self.rows[idx]
        col = self.cols[idx]

        # Return a tuple of the data
        return tile, label, file_name, row, col


def import_settings(path: str) -> dict[str, int]:
    """
    Imports image settings stored in the auxiliary ZIP archive
    associated with the given image.

    :param path: Path to an input image.
    :return: Dictionary containing image settings
    :rtype: dict
    """
    image_name = os.path.splitext(os.path.basename(path))[0]
    try:
        if AmfConfig.get("use_db"):
            conn = connect("amf")
            with conn, conn.cursor() as crsr:
                id_ = get_enabled(crsr, image_name)

                # Make sure there is an enabled image
                if id_ is None:
                    raise ValueError(f"No enabled image found for {image_name}")

                tile_edge = get_tile_edge(crsr, id_)
                return {"tile_edge": tile_edge[0]}
        else:
            dirname = os.path.split(path)[0]
            settings_path = f"{image_name}_settings.json"

            if settings_path in os.listdir(dirname):
                with open(os.path.join(dirname, settings_path)) as json_file:
                    return cast(dict[str, int], json.load(json_file))

        # Default to value in settings
        return {"tile_edge": AmfConfig.get("tile_edge")}
    except AssertionError as e:
        logger.warning(f"Failed to import settings for {image_name}: {e}")
        # Default to value in settings
        return {"tile_edge": AmfConfig.get("tile_edge")}


def import_annotations(path: str, is_bald_folder: bool = False) -> pd.DataFrame | None:
    """
    Imports tile annotations from the auxiliary ZIP archive
    associated with the given input image.

    :param path: Path to an input image.
    :return: Pandas dataframe containing annotations
    :rtype: pd.DataFrame
    """
    try:
        image_name = os.path.splitext(os.path.basename(path))[0]

        if AmfConfig.get("use_db"):
            colonisation_type = AmfConfig.get("colonisation_type")

            conn = connect("amf")
            with conn, conn.cursor() as crsr:
                id_ = get_enabled(crsr, image_name)

                if id_ is None and is_bald_folder:
                    return None

                if id_ is None:
                    raise ValueError(f"No enabled image found for {image_name}")

                existing_entries = check_entries_for_id(crsr, id_, colonisation_type)

                # Check that annotations exist

                if (
                    not existing_entries[id_]["cnn1_annotations_exist"]
                    and is_bald_folder
                ):
                    return None

                if not existing_entries[id_]["cnn1_annotations_exist"]:
                    raise ValueError(f"No annotations found for {image_name}")

                csv = download_entries_as_csv(
                    crsr, id_, "Annotations", colonisation_type
                )
                output = pd.read_csv(io.StringIO(csv), sep=",")

                # Drop question marks from the CSV
                if "Question" in output.columns:
                    # logger.info(f"Dropping questions from annotations for {image_name}")
                    output.drop("Question", axis=1, inplace=True, errors="ignore")

                # Drop question comments from the CSV
                if "QuestionComment" in output.columns:
                    # logger.info(
                    #     f"Dropping question comments from annotations for {image_name}"
                    # )
                    output.drop(
                        "QuestionComment", axis=1, inplace=True, errors="ignore"
                    )

                # Further check that csv is not empty
                if is_bald_folder and output.empty:
                    return None

                if output.empty:
                    raise ValueError("Annotation file is empty")

                return output
        else:
            directory = os.path.dirname(path)
            files = os.listdir(directory)

            regex_pattern = f"{image_name}_.+cnn_1_annotations.+"

            # Collect matching annotation files
            matching_annotations = [
                file for file in files if re.match(regex_pattern, file)
            ]

            # Ensure exactly one matching annotation file exists
            if len(matching_annotations) != 1:
                if is_bald_folder:
                    return None
                else:
                    raise ValueError(
                        f"Expected exactly one annotation file, found: "
                        f"{len(matching_annotations)}"
                    )

            output = pd.read_csv(os.path.join(directory, matching_annotations[0]))

            # If question does not exist then do not error when dropping
            # (for legacy CSVs)
            if "Question" in output.columns:
                # logger.info(f"Dropping questions from annotations for {image_name}")
                output.drop("Question", axis=1, inplace=True, errors="ignore")

            # Drop question comments from the CSV
            if "QuestionComment" in output.columns:
                # logger.info(
                #     f"Dropping question comments from annotations for {image_name}"
                # )
                output.drop("QuestionComment", axis=1, inplace=True, errors="ignore")

            # Further check that csv is not empty
            if output.empty:
                raise ValueError("Annotation file is empty")

            return output

    except (AssertionError, ValueError, KeyError) as e:
        logger.error(f"Error in importing annotations: {e}")
        return None


def categorise_path(paths: list[str]) -> dict[str, list[str]]:
    categories = ["train", "test", "BALD"]
    categorised_paths: dict[str, list[str]] = {"train": [], "test": [], "BALD": []}
    for path in paths:
        # Split the path based on '/'
        parts = path.split(os.path.sep)

        # Check each part of the path to see if it matches the categories
        for part in parts[::-1]:
            if part in categories:
                categorised_paths[part].append(path)
                break  # Stop searching once a match is found

    return categorised_paths


def is_blank(tile: NDArray[np.uint8]) -> bool:
    """Returns True if the tile is entirely black or entirely white."""
    return bool(np.all(tile == 0) or np.all(tile == 255))


def init_h5(f: h5py.File, class_names: list[str], colonisation_type: str) -> None:
    """Create the empty, resizable datasets for a fresh tile file."""
    vlen = h5py.vlen_dtype(np.uint8)
    strd = h5py.string_dtype(encoding="utf-8")

    f.create_dataset("jpg", (0,), maxshape=(None,), dtype=vlen, chunks=(256,))
    f.create_dataset(
        "cls", (0,), maxshape=(None,), dtype=np.uint8, chunks=(FLUSH_EVERY,)
    )
    f.create_dataset(
        "row", (0,), maxshape=(None,), dtype=np.int32, chunks=(FLUSH_EVERY,)
    )
    f.create_dataset(
        "col", (0,), maxshape=(None,), dtype=np.int32, chunks=(FLUSH_EVERY,)
    )
    f.create_dataset(
        "source", (0,), maxshape=(None,), dtype=strd, chunks=(FLUSH_EVERY,)
    )
    f.create_dataset(
        "completed_sources", (0,), maxshape=(None,), dtype=strd, chunks=(64,)
    )

    f.attrs["class_names"] = np.array(class_names, dtype=strd)
    f.attrs["colonisation_type"] = colonisation_type
    f.attrs["n_committed"] = 0


def preprocess_to_tiles(
    input_files: list[str], output_dir: str, h5_name: str = "tiles.h5"
) -> None:
    colonisation_type = AmfConfig.get("colonisation_type")
    class_names = AmfConfig.get("class_names")[colonisation_type]

    os.makedirs(output_dir, exist_ok=True)
    h5_path = os.path.join(output_dir, h5_name)

    # Source images whose blank tiles should always be fully retained
    special_cases = {"high_res_black_image_AM", "high_res_white_image_AM"}
    logger.info(f"Number of input files: {len(input_files)}")

    with h5py.File(h5_path, "a", libver="latest") as f:
        if "jpg" not in f:
            init_h5(f, class_names, colonisation_type)
            logger.info(f"Created new tile file: {h5_path}")
        else:
            existing = [
                s.decode() if isinstance(s, bytes) else str(s)
                for s in f.attrs["class_names"]
            ]
            if existing != list(class_names):
                logger.error(
                    f"Existing file uses classes {existing}, config expects "
                    f"{list(class_names)}. Aborting rather than mixing schemas."
                )
                sys.exit(33)

        d_jpg, d_cls = f["jpg"], f["cls"]
        d_row, d_col, d_src = f["row"], f["col"], f["source"]
        d_done = f["completed_sources"]

        # Roll back any tiles written after the last completed source image.
        n = int(f.attrs["n_committed"])
        if d_jpg.shape[0] != n:
            logger.warning(
                f"Discarding {d_jpg.shape[0] - n} uncommitted tiles from a "
                f"previous interrupted run."
            )
            for d in (d_jpg, d_cls, d_row, d_col, d_src):
                d.resize((n,))

        completed = {s.decode() if isinstance(s, bytes) else str(s) for s in d_done[:]}
        if completed:
            logger.info(
                f"Resuming: {len(completed)} source images already committed, "
                f"{n} tiles on disk."
            )

        wanted = {os.path.splitext(os.path.basename(p))[0] for p in input_files}
        missing = wanted - completed
        if not missing:
            logger.info(
                f"All {len(wanted)} source images already tiled "
                f"({n} tiles in {h5_path}). Skipping tile extraction."
            )
            counts = np.bincount(d_cls[:n], minlength=len(class_names))
            for i, class_name in enumerate(class_names):
                logger.info(f"  {class_name}: {int(counts[i])} tiles")
            return

        buf: list[tuple[bytes, int, int, int, str]] = []

        def flush() -> None:
            nonlocal n, buf
            if not buf:
                return
            m = len(buf)
            for d in (d_jpg, d_cls, d_row, d_col, d_src):
                d.resize((n + m,))

            jpgs = np.empty(m, dtype=object)
            for i, rec in enumerate(buf):
                jpgs[i] = np.frombuffer(rec[0], dtype=np.uint8)

            try:
                d_jpg[n : n + m] = jpgs
            except (TypeError, ValueError):
                # Every JPEG in this batch is the same byte length (uniform-colour
                # source image), so numpy collapses the object array into a 2-D
                # array that h5py cannot map onto a 1-D vlen selection.
                for i in range(m):
                    d_jpg[n + i] = jpgs[i]
            d_cls[n : n + m] = np.fromiter((r[1] for r in buf), np.uint8, m)
            d_row[n : n + m] = np.fromiter((r[2] for r in buf), np.int32, m)
            d_col[n : n + m] = np.fromiter((r[3] for r in buf), np.int32, m)
            d_src[n : n + m] = [r[4] for r in buf]

            n += m
            buf = []

        pb = tqdm(
            input_files, total=len(input_files), desc="Processing images", unit="image"
        )

        blank_count = 0

        for path in pb:
            # Extract the stem of the source filename e.g. "Image_1"
            source_name = os.path.splitext(os.path.basename(path))[0]
            if source_name in completed:
                continue

            annotations = import_annotations(path)
            settings = import_settings(path)

            # Per-image counter so numbering restarts cleanly for each source file
            image_tile_counter = 0

            if annotations is None:
                logger.info(f"Skipping {source_name}, no annotations found.")
                continue
            else:
                logger.info(
                    f"Processing {source_name} with {len(annotations)} annotated tiles."
                )

            tile_edge = settings["tile_edge"]
            if "tile_edge" not in f.attrs:
                f.attrs["tile_edge"] = int(tile_edge)
            elif int(f.attrs["tile_edge"]) != int(tile_edge):
                logger.warning(
                    f"{source_name} has tile_edge {tile_edge}, file was started "
                    f"with {int(f.attrs['tile_edge'])}. Tiles would not be uniform, skipping."
                )

            image = AmfSegm.load(path)
            for annot in annotations.itertuples():
                tile = AmfSegm.tile(image, annot.row, annot.col, tile_edge)
                label = np.array(annot[3:], dtype=np.uint8)
                tile = np.transpose(tile.astype(np.uint8), (1, 2, 0))
                if source_name not in special_cases and is_blank(tile):
                    blank_count += 1
                    continue  # Skip blank tiles for non-special cases

                class_idx = int(np.argmax(label))

                bio = io.BytesIO()
                Image.fromarray(tile).save(bio, format="JPEG", quality=95)
                buf.append(
                    (
                        bio.getvalue(),
                        class_idx,
                        int(annot.row),
                        int(annot.col),
                        source_name,
                    )
                )

                image_tile_counter += 1
                if len(buf) >= FLUSH_EVERY:
                    flush()

            del image

            # Commit point: everything from this source image is now durable.
            flush()
            d_done.resize((d_done.shape[0] + 1,))
            d_done[-1] = source_name
            completed.add(source_name)
            f.attrs["n_committed"] = n
            f.flush()

            logger.debug(
                f"Finished {source_name}: kept {image_tile_counter} of "
                f"{len(annotations)} tiles."
            )
            logger.debug(f"Total blank tiles skipped so far: {blank_count}")
            pb.set_postfix(tiles=n)

        logger.info("Tile extraction complete. Class distribution:")
        counts = np.bincount(d_cls[:n], minlength=len(class_names))
        for i, class_name in enumerate(class_names):
            logger.info(f"  {class_name}: {int(counts[i])} tiles")
        logger.info(f"  total: {n} tiles -> {h5_path}")
        logger.info(
            f"  blank tiles skipped: {blank_count}, total tiles: {n + blank_count}, "
            f"proportion skipped: {blank_count / (n + blank_count):.2%}"
        )


def get_tile_metadata(
    h5_path: str,
) -> tuple[list[int], list[NDArray[np.uint8]], list[str], list[int], list[int]]:
    """
    Reads per-tile metadata from a preprocessed HDF5 tile file without loading
    any image data.

    Returns:
        x: list of dataset row indices identifying each tile
        y: list of one-hot encoded labels
        file_names: list of source image stems
        rows: list of row positions in the source image
        cols: list of column positions in the source image
    """
    colonisation_type = AmfConfig.get("colonisation_type")
    class_names = AmfConfig.get("class_names")[colonisation_type]

    if not os.path.isfile(h5_path):
        logger.error(f"No image tile dataset file found at '{h5_path}'.")
        sys.exit(32)

    with h5py.File(h5_path, "r") as f:
        if "jpg" not in f:
            logger.error(f"'{h5_path}' contains no tile datasets.")
            sys.exit(32)

        existing = [
            s.decode() if isinstance(s, bytes) else str(s)
            for s in f.attrs["class_names"]
        ]
        if existing != list(class_names):
            logger.error(
                f"Image tile dataset file uses classes {existing}, config expects "
                f"{list(class_names)}. Aborting."
            )
            sys.exit(33)

        n = int(f.attrs["n_committed"])
        if n == 0:
            logger.error("No tiles found in image tile dataset file.")
            sys.exit(32)

        cls = f["cls"][:n]
        rows_arr = f["row"][:n]
        cols_arr = f["col"][:n]
        src_arr = f["source"][:n]

    eye = np.eye(len(class_names), dtype=np.uint8)

    x = list(range(n))
    y = [eye[c].copy() for c in cls]
    file_names = [s.decode() if isinstance(s, bytes) else str(s) for s in src_arr]
    rows = rows_arr.tolist()
    cols = cols_arr.tolist()

    logger.info(f"Retrieved metadata for {n} tiles.")
    return x, y, file_names, rows, cols
