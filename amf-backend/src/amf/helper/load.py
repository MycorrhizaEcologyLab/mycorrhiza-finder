"""Functionality for loading and tiling images and importing annotations."""

import io
import json
import os
import random
import re
import sys
from collections import defaultdict
from typing import cast

import numpy as np
import pandas as pd

# Torch functionalities
import torch
from loguru import logger
from numpy.typing import ArrayLike, NDArray
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import amf.helper.config as AmfConfig
import amf.helper.segmentation as AmfSegm
from amf.helper.api_utils import (
    check_entries_for_id,
    download_entries_as_csv,
    get_enabled,
    get_tile_edge,
)
from amf.helper.db_config import connect


class CustomNormalisedDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dataset that normalises images to [0, 1] on access."""

    def __init__(self, x: ArrayLike, y: ArrayLike):
        """Initialise dataset with data.

        Args:
            x: Input features (image tiles).
            y: Corresponding labels (one-hot encoded).
        """
        # Store data as NumPy arrays or lists
        self.x = np.array(x) if not isinstance(x, np.ndarray) else x
        self.y = np.array(y) if not isinstance(y, np.ndarray) else y

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.x)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return item (x, y) at given index, normalising x to [0, 1].

        Args:
            index: Index of the sample to retrieve.

        Returns: Tuple of normalised image tensor and label tensor.
        """
        # Convert to torch tensors only when accessed
        x_tensor = cast(
            torch.Tensor, torch.tensor(self.x[index], dtype=torch.float32) / 255.0
        )  # Normalisation to [0, 1]
        y_tensor = torch.tensor(self.y[index], dtype=torch.float32)
        return x_tensor, y_tensor


class TileDatasetLoader(Dataset[tuple[NDArray[np.uint8], NDArray[np.uint8]]]):
    """Dataset with data augmentation and balancing functionality."""

    def __init__(
        self,
        input_files: list[str],
        use_augmentation: bool = False,
        balance_factor: float = 1.0,
        calculate_distribution: bool = False,
    ):
        """Initialise dataset.

        Args:
            input_files: List of input image file paths.
            use_augmentation: Whether to apply data augmentation.
            balance_factor: Factor to balance classes in the dataset. This is the
               multiple of the size of the anchor class to clip other classes to.
            calculate_distribution: Whether to calculate and print class distribution
               statistics.
        """
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
        """Prepare dataset, including applying augmentation and balancing.

        Args:
            input_files: List of input image file paths.
            use_augmentation: Whether to apply data augmentation.

        Returns: Prepared dataset as a list of (tile, label) tuples.
        """
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
            dataset = list(zip(balanced_tiles, balanced_labels, strict=True))
        elif self.balance_factor == 0.0:
            # Create unbalanced dataset
            logger.info(
                f"Balance factor set to: {self.balance_factor}. "
                f"Generating unbalanced dataset."
            )
            dataset = list(zip(all_tiles, all_labels, strict=True))

        return dataset

    def _balance_dataset(
        self,
        x: list[NDArray[np.uint8]],
        y: list[NDArray[np.uint8]],
        factor: float = 1.0,
    ) -> tuple[list[NDArray[np.uint8]], list[NDArray[np.uint8]]]:
        """Balance dataset based on specified factor.

        This works by multiplying the size of the anchor class by the balancing factor
        and limiting all other classes to this size.

        Args:
            x: List of input features (image tiles).
            y: List of corresponding labels (one-hot encoded).
            factor: Balancing factor to determine class sizes.

        Returns: Balanced dataset as lists of (tiles, labels).
        """
        # Count the number of samples in class 0 (assumes one-hot encoding)
        class_0_samples = sum(array[0] for array in y)

        # Initialise a list to keep track of indices to retain
        indices_to_keep: list[int] = []

        # Convert the one-hot encoded labels to class indices
        y_indices = np.argmax(y, axis=1)

        # Retrieve the number of classes based on configuration
        colonisation_type = AmfConfig.get("colonisation_type")
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
        """Display class distribution statistics."""
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
        """Return the number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
        """Return the tile and label at the specified index."""
        tile, label = self.dataset[idx]
        return tile, label


class StratifiedDatasetSplitter:
    """Dataset splitter that maintains class distribution in train/val sets.

    Useful in imbalanced datasets where ensuring the proportional representation of
    classes in both training and validation sets improves model evaluation and training
    performance.
    """

    def __init__(
        self,
        x: list[NDArray[np.uint8]],
        y: list[NDArray[np.uint8]],
        val_size: float = 0.2,
        random_state: int = 42,
    ):
        """Initialise stratified dataset splitter with data.

        Args:
            x: List of input features (image tiles).
            y: List of corresponding labels (one-hot encoded).
            val_size: Fraction of dataset for validation set.
            random_state: Seed for random number generator.
        """
        self.x = x  # Ensure x is a numpy array
        self.y = y  # Ensure y is a numpy array

        # Perform stratified split while keeping the split index structures internal
        self._split_data(val_size, random_state)

    def _split_data(self, val_size: float, random_state: int) -> None:
        """Apply training/validation split while maintaining class distribution.

        Args:
            val_size: Fraction of dataset for validation set.
            random_state: Seed for random number generator.
        """
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
        self,
        dataset_type: str = "train",  # TODO use enum?
    ) -> tuple[list[NDArray[np.uint8]], list[NDArray[np.uint8]]]:
        """Return the specified dataset split.

        Args:
            dataset_type: Type of dataset to return ("train" or "val").

        Returns: Tuple of (x, y) for the specified dataset split.
        """
        if dataset_type == "train":
            x_train = [self.x[i] for i in self.train_indices]
            y_train = [self.y[i] for i in self.train_indices]
            return x_train, y_train
        if dataset_type == "val":
            x_val = [self.x[i] for i in self.val_indices]
            y_val = [self.y[i] for i in self.val_indices]
            return x_val, y_val
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
    """Class to handle loading and tiling of images and annotations."""

    def __init__(self, input_files: list[str], is_bald_folder: bool = False):
        """Initialise dataset with data from input files.

        Loads and processes data by extracting tiles and their annotations.

        Args:
            input_files: List of input image file paths
            is_bald_folder: Whether the data is from a BALD folder
        """
        self.input_files = input_files
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
        """Load and process image data, extracting tiles with or without labels.

        Loads images from the configured input files, imports their settings and
        annotations, and extracts tiles. The processing behavior varies based on whether
        unlabelled data is requested and whether annotations are available.

        Args:
            is_bald_folder: Whether data is from a BALD folder.
            is_unlabelled: Whether to extract only unlabelled tiles (True) or only
                labelled tiles (False).

        Returns:
            If is_unlabelled is True:
                tiles: List of image tile data
                file_names: List of source file paths for each tile
                rows: List of row indices for each tile
                cols: List of column indices for each tile

            If is_unlabelled is False:
                tiles: List of image tile data
                labels: List of label arrays for each tile
                file_names: List of source file paths for each tile
                rows: List of row indices for each tile
                cols: List of column indices for each tile
        """
        logger.info("Tile extraction.")

        # Load image settings and annotations.
        annotations = [
            import_annotations(path, is_bald_folder) for path in self.input_files
        ]
        settings = [import_settings(path) for path in self.input_files]

        dataset = zip(self.input_files, settings, annotations, strict=True)

        if is_unlabelled:
            dataset_list = list(dataset)
            logger.info(f"{len(dataset_list)} images in BALD dataset.")

            # Process and normalize dataset
            x, file_names, rows, cols = self._process_dataset_unlabelled(dataset_list)

            return x, file_names, rows, cols

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
        """Extract labelled tiles and their labels from the dataset.

        Args:
            dataset: List of tuples, each containing:
                - path: File path to the image
                - config: Config dictionary with 'tile_edge' key specifying tile size in
                    pixels
                - annots: Annotations DataFrame where each row contains row index,
                    column index and one-hot annotations

        Returns:
            tiles: List of image tiles
            hot_labels: List of label arrays extracted from annotation columns
            file_names: List of source file paths for each tile
            rows: List of row indices
            cols: List of column indices
        """
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
        """Extract unlabelled tiles from dataset.

        Args:
            dataset: A list of tuples, each containing:
                - path: File path to the image
                - config: Config dictionary with 'tile_edge' key specifying tile size
                    in pixels
                - annots: Annotations DataFrame with 'row' and 'col' columns indicating
                    labelled tile coordinates, or None if no tiles are labelled

        Returns:
            tiles: List of image tiles
            file_names: List of source image file paths
            rows: List of row indices
            cols: List of column indices
        """
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
                labelled_coords = set(zip(annots.row, annots.col, strict=True))
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
        """Return all data from the dataset.

        Returns:
            x: List of image tile data.
            y: List of label arrays for each tile.
            file_names: List of source file paths for each tile.
            rows: List of row indices for each tile.
            cols: List of column indices for each tile.
        """
        return self.x, self.y, self.file_names, self.rows, self.cols

    def get_all_unlabelled_data(
        self,
    ) -> tuple[list[NDArray[np.uint8]], list[str], list[int], list[int]]:
        """Return all unlabelled data from the dataset.

        Returns:
            x_unlabelled: List of unlabelled image tiles.
            file_names_unlabelled: List of file names for each unlabelled tile.
            rows_unlabelled: List of row indices for each unlabelled tile.
            cols_unlabelled: List of column indices for each unlabelled tile.
        """
        return (
            self.x_unlabelled,
            self.file_names_unlabelled,
            self.rows_unlabelled,
            self.cols_unlabelled,
        )

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
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
        """Get a specific tile by its index.

        Args:
            idx: Index of tile in dataset.

        Returns:
            tile: Image tile.
            label: One-hot encoded label for the tile.
            file_name: Name of the file the tile was extracted from.
            row: Row index of the tile in the original image.
            col: Column index of the tile in the original image.
        """
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
    """Determine tile edge length in pixels for a given image.

    Attempts to read value from the database or from a JSON settings file, or otherwise
    falls back to the configured default.

    Args:
        path: Image path.

    Returns: Dictionary containing tile edge length under `tile_edge` key.
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
    except ValueError as e:
        logger.warning(f"Failed to import settings for {image_name}: {e}")
        # Default to value in settings
        return {"tile_edge": AmfConfig.get("tile_edge")}


def import_annotations(path: str, is_bald_folder: bool = False) -> pd.DataFrame | None:
    """Load annotations from the database or CSV file (most recent annotations).

    In the case of the database, the enabled annotations are used. For CSV annotations,
    the most recent annotations are used.

    Args:
        path: Path to input image.
        is_bald_folder: Whether the data is from a BALD folder.

    Returns: DataFrame containing selected annotations, or None if no annotations exist.
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
                    logger.info(f"Dropping questions from annotations for {image_name}")
                    output.drop("Question", axis=1, inplace=True, errors="ignore")

                # Drop question comments from the CSV
                if "QuestionComment" in output.columns:
                    logger.info(
                        f"Dropping question comments from annotations for {image_name}"
                    )
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
                raise ValueError(
                    "Expected exactly one annotation file, found: "
                    "{len(matching_annotations)}"
                )

            output = pd.read_csv(os.path.join(directory, matching_annotations[0]))

            # If question does not exist then do not error when dropping
            # (for legacy CSVs)
            if "Question" in output.columns:
                logger.info(f"Dropping questions from annotations for {image_name}")
                output.drop("Question", axis=1, inplace=True, errors="ignore")

            # Drop question comments from the CSV
            if "QuestionComment" in output.columns:
                logger.info(
                    f"Dropping question comments from annotations for {image_name}"
                )
                output.drop("QuestionComment", axis=1, inplace=True, errors="ignore")

            # Further check that csv is not empty
            if output.empty:
                raise ValueError("Annotation file is empty")

            return output

    except (AssertionError, ValueError, KeyError) as e:
        logger.error(f"Error in importing annotations: {e}")
        return None


def categorise_path(paths: list[str]) -> dict[str, list[str]]:
    """Group paths into `train`, `test` and `BALD` categories based on directory names.

    For each path, traverses the directory structure backwards until it finds one of the
    target directory names. If not, the path is ignored.

    Args:
        paths: List of file paths to categorise.

    Returns: Dictionary with keys `train`, `test` and `BALD`, each containing a list of
        corresponding paths.
    """
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
