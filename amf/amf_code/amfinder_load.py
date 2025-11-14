import glob
import io
import json
import os
import random
import re
from collections import defaultdict

import numpy as np
import pandas as pd

# Torch functionalities
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import amfinder_config as AmfConfig
import amfinder_log as AmfLog
import amfinder_segmentation as AmfSegm
from api_utils import (
    check_entries_for_id,
    download_entries_as_csv,
    get_enabled,
    get_tile_edge,
)
from db_config import connect


# Defining class for Datasetloader
class CustomNormalisedDataset(Dataset):
    """
    Manages dataset loading and normalizes image data to the range [0, 1].
    This is intended to be used by DataLoaders primarily for efficient memory handling.

    Expected Inputs:
    x: Iterable of image data (e.g., list of tiles as NumPy arrays or as simple list).
    y: Iterable of labels corresponding to the image data (e.g., list of NumPy arrays or
        lists representing the labels).

    Methods:
    __init__(self, x, y): Initializes the dataset, storing provided data as NumPy
        arrays.
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

    def __init__(self, x, y):
        # Store data as NumPy arrays or lists
        self.x = np.array(x) if not isinstance(x, np.ndarray) else x
        self.y = np.array(y) if not isinstance(y, np.ndarray) else y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        # Convert to torch tensors only when accessed
        x_tensor = (
            torch.tensor(self.x[index], dtype=torch.float32) / 255.0
        )  # Normalisation to [0, 1]
        y_tensor = torch.tensor(self.y[index], dtype=torch.float32)
        return x_tensor, y_tensor


class TileDatasetLoader(Dataset):
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
        input_files,
        use_augmentation=False,
        balance_factor=1.0,
        calculate_distribution=False,
    ):
        self.use_augmentation = use_augmentation
        self.balance_factor = balance_factor
        self.dataset = self._prepare_dataset(input_files, self.use_augmentation)
        self.labels = [label for _, label in self.dataset]  # Ensuring labels are stored
        self.calculate_distribution = calculate_distribution

        # Calculate statistics if needed
        if self.calculate_distribution:
            AmfLog.info(f"Total images for train/val split: {len(input_files)}")
            self._prepare_stats()

    def _prepare_dataset(self, input_files, use_augmentation):
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
                        tile = (
                            torch.tensor(tile, dtype=torch.float32) / 255.0
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
            AmfLog.error(
                "None of the training images had annotations, so fail",
                AmfLog.ERR_MISSING_ANNOTATIONS,
            )

        if self.balance_factor != 0.0:
            # Apply balancing
            balanced_tiles, balanced_labels = self._balance_dataset(
                all_tiles, all_labels, self.balance_factor
            )
            # Create balanced dataset
            AmfLog.text(
                f"Balance factor set to: {self.balance_factor}. "
                "Generating balanced dataset."
            )
            dataset = list(zip(balanced_tiles, balanced_labels))
        elif self.balance_factor == 0.0:
            # Create unbalanced dataset
            AmfLog.text(
                f"Balance factor set to: {self.balance_factor}. "
                "Generating unbalanced dataset."
            )
            dataset = list(zip(all_tiles, all_labels))

        return dataset

    def _balance_dataset(self, x, y, factor=1.0):  # factor is a hyperparameter
        # Count the number of samples in class 0 (assumes one-hot encoding)
        class_0_samples = sum(array[0] for array in y)

        # Initialize a list to keep track of indices to retain
        indices_to_keep = []

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

    def _prepare_stats(self):
        # Collect labels for statistics
        samples = np.stack([self.dataset[i][1] for i in range(len(self.dataset))])
        y = torch.from_numpy(samples)
        y = y.type(torch.float32)
        class_counts = y.sum(dim=0)
        total_samples = y.size(0)
        percentages = (class_counts / total_samples) * 100

        print("Class Distribution before train/val split:")
        for i, count in enumerate(class_counts):
            print(f"Class {i}: {count} samples ({percentages[i]:.2f}%)")

        uniqueargs_probe_hot_indexes = (
            (class_counts > 0).nonzero(as_tuple=True)[0].numpy()
        )
        uniqueargs_headers = np.arange(len(class_counts))

        if not np.array_equal(uniqueargs_probe_hot_indexes, uniqueargs_headers):
            AmfLog.error(
                "Training data does not represent all classes. Please reconsider "
                "training dataset curation",
                AmfLog.ERR_NO_DATA,
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        tile, label = self.dataset[idx]
        return tile, label


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

    def __init__(self, x, y, val_size=0.2, random_state=42):
        self.x = x  # Ensure x is a numpy array
        self.y = y  # Ensure y is a numpy array

        # Perform stratified split while keeping the split index structures internal
        self._split_data(val_size, random_state)

    def _split_data(self, val_size, random_state):
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

    def _get_dataset(self, dataset_type="train"):
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


class TileFilesandData(Dataset):
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

    def __init__(self, input_files, is_bald_folder=False):
        self.input_files = input_files
        self.x, self.y, self.file_names, self.rows, self.cols = (
            self._load_and_process_data(is_bald_folder)
        )
        if is_bald_folder:
            (
                self.x_unlabelled,
                self.file_names_unlabelled,
                self.rows_unlabelled,
                self.cols_unlabelled,
            ) = self._load_and_process_data(is_bald_folder, is_unlabelled=True)

        else:
            self.x_unlabelled = None
            self.file_names_unlabelled = None
            self.rows_unlabelled = None
            self.cols_unlabelled = None

    def _load_and_process_data(self, is_bald_folder=False, is_unlabelled=False):
        print(f"[{AmfConfig.invite()}] Tile extraction.")

        # Load image settings and annotations.
        annotations = [
            import_annotations(path, is_bald_folder) for path in self.input_files
        ]
        settings = [import_settings(path) for path in self.input_files]

        dataset = zip(self.input_files, settings, annotations)

        if is_unlabelled:
            dataset = list(dataset)
            print(f"[{AmfConfig.invite()}] {len(dataset)} images in BALD dataset.")

            # Process and normalize dataset
            x, file_names, rows, cols = self._process_dataset_unlabelled(dataset)

            return x, file_names, rows, cols

        else:
            filtered_dataset = [x for x in dataset if x[2] is not None]
            if len(filtered_dataset) == 0 and not is_bald_folder:
                AmfLog.error(
                    "Input images do not contain tile annotations. "
                    "Use amfbrowser to annotate tiles before training",
                    AmfLog.ERR_NO_DATA,
                )

            print(
                f"[{AmfConfig.invite()}] {len(filtered_dataset)} images in filtered "
                "dataset."
            )

            # Process and normalize dataset
            x, y, file_names, rows, cols = self._process_dataset(filtered_dataset)

            return x, y, file_names, rows, cols

    def _process_dataset(self, dataset):
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

    def _process_dataset_unlabelled(self, dataset):
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

    def get_all_data(self):
        """Returns all processed data from the dataset."""
        return self.x, self.y, self.file_names, self.rows, self.cols

    def get_all_unlabelled_data(self):
        """Returns all unlabelled data from the dataset."""
        return (
            self.x_unlabelled,
            self.file_names_unlabelled,
            self.rows_unlabelled,
            self.cols_unlabelled,
        )

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
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


# TODO fix match needs to be updated to work with ErM model
class FixMatchLoader:
    def get_input_files(self, root):
        """
        Filter input file list and keep valid JPEG or PNG images.
        Return the paths relative to the root directory.
        """
        # Define patterns for JPEG and PNG files
        patterns = ["*.jpg", "*.jpeg", "*.png"]
        images = []

        # Walk through the directory
        for pattern in patterns:
            # Use glob to find all files matching the pattern
            for path in glob.glob(os.path.join(root, "**", pattern), recursive=True):
                # Make the path relative to root
                relative_path = os.path.relpath(path, start=root)
                images.append(relative_path)

        return images

    def _load_data(self, input_files, root):
        abs_paths = [os.path.join(root, path) for path in input_files]

        annotations = [import_annotations(path) for path in abs_paths]

        # Remove images without annotations.
        dataset = zip(abs_paths, annotations)
        filtered_dataset = [x for x in dataset if x[1] is not None]
        return filtered_dataset

    def load_val_dataset(self, input_files: list, root: str):
        """
        Loads validation tile set and their corresponding annotations.

        :param input_files: List of input images to use for training.
        :return: Numpy arrays containing tiles and one-hot encoded annotations.
        :rtype: tuple
        """

        filtered_dataset = self._load_data(input_files, root)

        # Determine the required amount (in %) of background subsampling (if active).
        def process_dataset(dataset: list = None):
            tiles = []
            hot_labels = []

            def get_tile(image, annot):
                tile = AmfSegm.tile(image, annot.row, annot.col)
                tiles.append(tile)
                hot_labels.append(list(annot[3:-1]))

            for path, annots in dataset:
                # FIXME: Random access is inefficient. To achieve better
                # efficiency we would have to load tiles row by row.
                # set the number of not colonised to = the number of colonised
                # get the max examples based on the max number of examples in the
                # minority class
                max_examples = min(
                    annots.sum().AMColonised,
                    annots.sum().Uncolonised,
                    annots.sum().Unreadable,
                    annots.sum().DSE,
                    annots.sum().Hybrid,
                )
                image = AmfSegm.load(path)

                # Extract tile sets (= original tile and augmented versions).
                # Repeat one-hot encoded annotations for each tile.
                num_AM = 0
                num_Uncolonised = 0
                num_Unreadable = 0
                num_DSE = 0
                num_Hybrid = 0
                for annot in annots.itertuples():
                    if annot.Uncolonised == 1 and num_Uncolonised < max_examples:
                        get_tile(image, annot)
                        num_Uncolonised += 1
                    elif annot.AMColonised == 1 and num_AM < max_examples:
                        get_tile(image, annot)
                        num_AM += 1
                    elif annot.Unreadable == 1 and num_Unreadable < max_examples:
                        get_tile(image, annot)
                        num_Unreadable += 1
                    elif annot.DSE == 1 and num_DSE < max_examples:
                        get_tile(image, annot)
                        num_DSE += 1
                    elif annot.Hybrid == 1 and num_Hybrid < max_examples:
                        get_tile(image, annot)
                        num_Hybrid += 1

                del image

            return np.array(tiles, np.float32), np.array(hot_labels, np.uint8)

        x, y = process_dataset(filtered_dataset)
        indices = np.random.permutation(x.shape[0])
        x_shuffled = x[indices]
        y_shuffled = np.argmax(y[indices], axis=1)

        return x_shuffled, y_shuffled

    def load_test_dataset(self, input_files, root):
        """
        Loads training tile set and their corresponding annotations.

        :param input_files: List of input images to use for training.
        :return: Numpy arrays containing tiles and one-hot encoded annotations.
        :rtype: tuple
        """

        filtered_dataset = self._load_data(input_files, root)

        # Determine the required amount (in %) of background subsampling (if active).
        def process_dataset(dataset: list = None):
            tiles = []
            hot_labels = []
            file_names = []

            for path, annots in dataset:
                image = AmfSegm.load(path)

                for annot in annots.itertuples():
                    if annot.Background == 1:
                        continue
                    else:
                        tile = AmfSegm.tile(image, annot.row, annot.col)
                        tiles.append(tile)
                        hot_labels.append(list(annot[3:-1]))
                        file_names.append(path)

                del image

            return (
                np.array(tiles, np.float32),
                np.array(hot_labels, np.uint8),
                file_names,
            )

        x, y, file_names = process_dataset(filtered_dataset)

        return x, np.argmax(y, axis=1), file_names

    def load_unlabelled_dataset(self, labelled_indices, prop_unlabelled, x):
        """
        Loads training tile set and their corresponding annotations.

        :param input_files: List of input images to use for training.
        :return: Numpy arrays containing tiles and one-hot encoded annotations.
        :rtype: tuple
        """

        all_indices = set(range(len(x)))
        unlabelled_indices = list(all_indices - set(labelled_indices))
        num_examples = int(np.floor(len(unlabelled_indices) * prop_unlabelled))
        sampled_unlabelled_indices = np.random.choice(
            unlabelled_indices, num_examples, replace=False
        )
        unlabelled_images = [x[i] for i in sampled_unlabelled_indices]
        return unlabelled_images, num_examples

    def load_train_dataset(self, input_files, num_labelled, args):
        """
        Loads training tile set and their corresponding annotations, and stores the file
        name for each tile.

        :param input_files: List of input images to use for training.
        :return: Numpy arrays containing tiles, one-hot encoded annotations, and list of
            file names.
        :rtype: tuple
        """

        filtered_dataset = self._load_data(input_files, args.root_path)

        # Terminate if there is no data to process.
        if len(filtered_dataset) == 0:
            raise ValueError("No files to process")

        def sample_data(images, y, num_labelled, filenames, cols, rows):
            # Convert one-hot encodings to class labels
            labels = np.argmax(y, axis=1)
            unique_labels = np.unique(labels)
            print("number of classes: ", len(unique_labels))

            labelled_indices = []

            num_labelled_per_class = num_labelled // len(unique_labels)

            # Sample labelled data
            for label in unique_labels:
                indices = np.where(labels == label)[0]
                if num_labelled_per_class > len(indices):
                    sampled_indices = indices
                else:
                    sampled_indices = np.random.choice(
                        indices, num_labelled_per_class, replace=False
                    )
                labelled_indices.extend(sampled_indices)

            # Prepare output
            labelled_images = [images[i] for i in labelled_indices]
            labelled_files = [filenames[i] for i in labelled_indices]
            labelled_cols = [cols[i] for i in labelled_indices]
            labelled_rows = [rows[i] for i in labelled_indices]
            labels = [labels[i] for i in labelled_indices]
            df = pd.DataFrame(
                {
                    "Filename": labelled_files,
                    "Column": labelled_cols,
                    "Row": labelled_rows,
                    "Label": labels,
                }
            )

            # Step 2: Save the DataFrame to a CSV file
            csv_file_path = os.path.join(
                args.out, "labelled_data.csv"
            )  # Specify the path and name of the CSV file
            df.to_csv(csv_file_path, index=False)
            # unlabelled_images = [images[i] for i in sampled_unlabelled_indices]

            return labelled_images, labels, labelled_indices

        # Determine the required amount (in %) of background subsampling (if active).
        def process_dataset(dataset: list = None):
            tiles = []
            hot_labels = []
            file_names = []  # List to store the file name for each tile
            cols = []
            rows = []
            for path, annots in dataset:
                image = AmfSegm.load(path)
                for annot in annots.itertuples():
                    if annot.Background == 1:
                        continue
                    else:
                        tile = AmfSegm.tile(image, annot.row, annot.col)
                        tiles.append(tile)
                        cols.append(annot.col)
                        rows.append(annot.row)
                        hot_labels.append(list(annot[3:-1]))
                        file_names.append(path)  # Append the file name

                del image

            return (
                np.array(tiles, np.float32),
                np.array(hot_labels, np.uint8),
                file_names,
                cols,
                rows,
            )

        def shuffle_dataset(images, targets, labelled_indices):
            combined = list(zip(images, targets, labelled_indices))
            random.shuffle(combined)
            images, targets, labelled_indices = zip(*combined)
            return list(images), list(targets), list(labelled_indices)

        x, y, file_names, cols, rows = process_dataset(filtered_dataset)

        x_labelled, labels, labelled_indices = sample_data(
            x, y, num_labelled, file_names, cols, rows
        )

        # This function assumes images have been properly sampled and labels extracted
        # beforehand.
        x_labelled, labels, labelled_indices = shuffle_dataset(
            x_labelled, labels, labelled_indices
        )

        return x_labelled, labels, labelled_indices, x


def import_settings(path):
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
                    return json.load(json_file)

        # Default to value in settings
        return {"tile_edge": AmfConfig.get("tile_edge")}
    except AssertionError as e:
        AmfLog.warning(f"Failed to import settings for {image_name}: {e}")
        # Default to value in settings
        return {"tile_edge": AmfConfig.get("tile_edge")}


def import_annotations(path, is_bald_folder=False):
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
                    print(f"Dropping questions from annotations for {image_name}")
                    output.drop("Question", axis=1, inplace=True, errors="ignore")

                # Drop question comments from the CSV
                if "QuestionComment" in output.columns:
                    print(
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
                else:
                    raise ValueError(
                        "Expected exactly one annotation file, found: "
                        "{len(matching_annotations)}"
                    )

            output = pd.read_csv(os.path.join(directory, matching_annotations[0]))

            # If question does not exist then do not error when dropping
            # (for legacy CSVs)
            if "Question" in output.columns:
                print(f"Dropping questions from annotations for {image_name}")
                output.drop("Question", axis=1, inplace=True, errors="ignore")

            # Drop question comments from the CSV
            if "QuestionComment" in output.columns:
                print(f"Dropping question comments from annotations for {image_name}")
                output.drop("QuestionComment", axis=1, inplace=True, errors="ignore")

            # Further check that csv is not empty
            if output.empty:
                raise ValueError("Annotation file is empty")

            return output

    except (AssertionError, ValueError, KeyError) as e:
        print(f"Error in importing annotations: {e}")
        return None


def categorise_path(paths):
    categories = ["train", "test", "BALD"]
    categorised_paths = {"train": [], "test": [], "BALD": []}
    for path in paths:
        # Split the path based on '/'
        parts = path.split(os.path.sep)

        # Check each part of the path to see if it matches the categories
        for part in parts[::-1]:
            if part in categories:
                categorised_paths[part].append(path)
                break  # Stop searching once a match is found

    return categorised_paths
