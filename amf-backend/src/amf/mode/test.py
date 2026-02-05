# AMFinder - test.py
#
# Copyright (c) 2024-2025 Royal Botanic Gardens, Kew
#
# Evaluates the model on the test roots to obtain metrics.

import os
import random
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from loguru import logger
from numpy.typing import NDArray
from torch.utils.data import DataLoader
from tqdm import tqdm

import amf.helper.config as AmfConfig
import amf.helper.load as AmfLoad
import amf.helper.model as AmfModel
import amf.helper.save as AmfSave
import amf.helper.segmentation as AmfSegm
from amf.helper.metrics_collector import MetricsCollector
from amf.helper.test_metrics import TestMetrics

random.seed(42)


def get_test_results(
    x_test: list[NDArray[np.uint8]],  # Test images
    y_test: list[NDArray[np.uint8]],  # Labels for images
    filenames: list[str],  # filenames as required in load_data_and_files()
    results_dir: str,  # Directory to save results
    model: torch.nn.Module,  # Selected model as pre config file
    test_loader: DataLoader[
        tuple[torch.Tensor, torch.Tensor]
    ],  # Pre-defined test-set DataLoader
    metrics_collector: MetricsCollector,  # Object to collect metrics
    rows: list[int],  # Rows for visualisation (if needed)
    cols: list[int],  # Cols for visualisation (if needed)
) -> None:
    """
    Evaluate a trained model on a test dataset, calculate metrics, and save the results.

    This function performs the following steps:
    1. Sets the model to evaluation mode and assigns the computational device.
    2. Iterates through the test data to compute predictions, calculate loss, and
        evaluate accuracy.
    3. Collects predicted labels, one-hot encoded predictions, and probabilities for all
        test samples.
    4. Converts predictions and probabilities to numpy arrays for further processing.
    5. For predictions with low confidence, applies max voting using surrounding tiles.
    6. Initializes a `TestMetrics` object to calculate and save various performance
        metrics, such as confusion matrices, per-file and per-class metrics.

    Parameters:
    - x_test: np.ndarray
        Array of test images.
    - y_test: np.ndarray
        Ground truth labels for the test images, one-hot encoded or as class indices.
    - filenames: list of str
        Filenames corresponding to the test images, used for saving results.
    - results_dir: str
        Directory path where the metrics and results will be saved.
    - model: torch.nn.Module
        Pre-trained model to be evaluated.
    - test_dataset: torch.utils.data.DataSet
        Datset object containing the test tiles alongside corresponding labels.
    - test_loader: torch.utils.data.DataLoader
        DataLoader object providing batches of test data. Expecting numpy arrays as
        input but using normalised torch tensors in batches.
    - metrics_collector: MetricsCollector
        Object responsible for storing and managing calculated metrics.
    - rows: int
        Number of rows for visualization (if applicable).
    - cols: int
        Number of columns for visualization (if applicable).
    - confidence_threshold: float, optional (default=0.7)
        Threshold below which max voting with surrounding tiles will be applied.

    Returns:
    - None
        Saves the calculated metrics and results to the specified directory.

    Notes:
    - The function uses CrossEntropyLoss as the loss criterion.
    - Outputs are processed to obtain class predictions and probabilities.
    - Low confidence predictions are refined using max voting from surrounding tiles.
    - Several metrics are calculated using the `TestMetrics` class, including
      confusion matrices, per-file predictions, and per-class predictions.
    """

    # Start timing
    start_time = time.time()

    # Assigning correct computational resource and setting model to evaluation mode.
    device = AmfConfig.get("device")
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    # Grab temperature scaling
    colonisation_type = AmfConfig.get("colonisation_type")
    temperature_factor_file = (
        AmfConfig.get("temperature_factor_path")
        if colonisation_type == "am"
        else AmfConfig.get("temperature_factor_path_erm")
    )

    # Check in trained_networks
    temperature_factor = 1.0
    if temperature_factor_file is not None:
        path = os.path.join(AmfConfig.get_model_dir(), temperature_factor_file)

        # Otherwise return path to allow for absolute paths
        if not os.path.isfile(path):
            path = temperature_factor_file

        try:
            with open(path) as f:
                tf = f.readline()
                temperature_factor = float(tf)
        except Exception:
            logger.warning(
                f"Failed to read temperature factor from file {path}, setting to 1"
            )

    logger.info(f"Using temperature factor of {temperature_factor}")

    # Defining loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Lists to store predictions, probabilities, and labels
    predicted_labels_list = []
    true_labels_list = []
    all_probs_list = []

    batch_count = len(test_loader)

    # Loop over test data
    with torch.no_grad():  # Disabling gradient calculations
        for batch_x, batch_y in tqdm(
            test_loader, total=batch_count, desc="Processing batches for test"
        ):
            # Load to correct device
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Forward pass
            outputs = model(batch_x)

            # Determine what batch_y should be
            if batch_y.ndimension() > 1:  # Check if batch_y is one-hot encoded
                batch_y = torch.argmax(batch_y, dim=1)  # Convert to class indices

            # Calculate loss
            loss = criterion(outputs, batch_y)
            test_loss += loss.item()

            # Get predictions and probabilities
            probabilities = torch.softmax(outputs / temperature_factor, dim=1)
            class_indices = torch.argmax(probabilities, dim=1)
            num_classes = probabilities.size(1)

            # Accumulate metrics
            total += batch_y.size(0)
            correct += (class_indices == batch_y).sum().item()

            # Collect predictions and true labels
            predicted_labels_list.extend(class_indices.cpu().numpy())
            true_labels_list.extend(
                F.one_hot(batch_y, num_classes=num_classes).cpu().numpy()
            )
            all_probs_list.extend(probabilities.cpu().numpy())

    # Calculate average loss and accuracy
    avg_loss = test_loss / batch_count
    test_accuracy = correct / total

    # print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    logger.info(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # General metrics collection
    metrics_collector.add_generic_metric("Test accuracy", test_accuracy)
    metrics_collector.add_generic_metric("Loss", avg_loss)

    # Convert to numpy arrays
    # print("Converting to numpy arrays")
    logger.debug("Converting to numpy arrays")
    predicted_labels = np.array(predicted_labels_list)
    true_labels = np.array(true_labels_list)
    all_probs = np.array(all_probs_list)
    y_test_labels = np.argmax(y_test, axis=1)

    use_contextual_confidence = AmfConfig.get("use_contextual_confidence")
    if use_contextual_confidence:
        contextual_confidence_threshold = AmfConfig.get(
            "contextual_confidence_threshold"
        )  # Threshold for applying max voting
        # Apply max voting for low confidence predictions
        max_confidences = np.max(all_probs, axis=1)
        low_confidence_indices = np.where(
            max_confidences <= contextual_confidence_threshold
        )[0]

        # Initialize a dictionary to track class changes
        class_changes = {}
        # Initialize a list to track individual tile changes
        individual_tile_changes = []

        if len(low_confidence_indices) > 0:
            logger.info(
                f"Found {len(low_confidence_indices)} predictions with confidence \
                lower than {contextual_confidence_threshold}. Applying max voting"
            )

            # Group by image to minimize image loading
            file_to_indices: dict[str, list[tuple[int, int, int]]] = {}
            for idx in low_confidence_indices:
                file_path = filenames[idx]
                if file_path not in file_to_indices:
                    file_to_indices[file_path] = []
                file_to_indices[file_path].append((idx, rows[idx], cols[idx]))

            # Create a mapping from class indices to class names before processing
            header = AmfConfig.get("header")
            class_names_map = {i: name for i, name in enumerate(header)}

            # Process each image
            for file_path, indices_with_coords in file_to_indices.items():
                # Load the image only once per file
                image = AmfSegm.load(file_path)
                edge = AmfConfig.get("tile_edge")

                # Batch process all low-confidence tiles from this image
                all_contextual_tile_sets = []
                # Number of contextual tiles for a given tile index
                num_contextual_tiles_mapping = []

                for idx, r, c in indices_with_coords:
                    # Get surrounding tiles
                    tiles = AmfSegm.get_contextual_tiles(image, r, c, edge)

                    # Filter out None values (beyond image boundaries)
                    valid_tiles = [t for t in tiles if t is not None]

                    if valid_tiles:  # Only process if we have valid tiles
                        all_contextual_tile_sets.extend(valid_tiles)
                        num_contextual_tiles_mapping.append(
                            (idx, len(valid_tiles), r, c)
                        )  # Store row and col for logging

                if not all_contextual_tile_sets:
                    continue

                # Normalize and convert to torch tensor
                all_contextual_tiles_array = np.array(
                    all_contextual_tile_sets, dtype=np.uint8
                )

                contextual_tile_dataset = AmfLoad.CustomNormalisedDataset(
                    all_contextual_tiles_array,
                    np.zeros((len(all_contextual_tiles_array), num_classes)),
                )
                bs = AmfConfig.get("batch_size")
                num_workers = AmfConfig.get("num_workers")
                contextual_tile_loader = DataLoader(
                    contextual_tile_dataset,
                    batch_size=bs,
                    shuffle=False,
                    num_workers=num_workers,
                )

                tile_preds = []
                with torch.no_grad():  # Disabling gradient calculations
                    for batch_x, batch_y in tqdm(
                        contextual_tile_loader,
                        total=len(contextual_tile_loader),
                        desc=(
                            "Processing contextual predictions for "
                            f"{os.path.splitext(os.path.basename(file_path))[0]}"
                        ),
                    ):
                        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                        outputs = model(batch_x)
                        probs = torch.softmax(outputs / temperature_factor, dim=1)
                        class_indices = torch.argmax(probs, dim=1)
                        tile_preds.extend(class_indices.cpu().numpy())

                # Apply max voting for each original low-confidence tile
                start_idx = 0
                for (
                    orig_idx,
                    num_tiles,
                    r,
                    c,
                ) in num_contextual_tiles_mapping:  # Unpack with row and col
                    if num_tiles > 0:
                        # Get predictions for this tile's neighborhood
                        neighborhood_preds = tile_preds[
                            start_idx : start_idx + num_tiles
                        ]
                        central_tile_with_neighborhood_preds = [
                            predicted_labels[orig_idx]
                        ] + neighborhood_preds

                        # Apply max vote
                        final_class = Counter(
                            central_tile_with_neighborhood_preds
                        ).most_common(1)[0][0]

                        # Update prediction
                        if predicted_labels[orig_idx] != final_class:
                            # Get class names for better readability
                            from_class_name = class_names_map[
                                predicted_labels[orig_idx]
                            ]
                            to_class_name = class_names_map[final_class]

                            # Update the tracking dictionary
                            change_key = f"{from_class_name} -> {to_class_name}"
                            if change_key not in class_changes:
                                class_changes[change_key] = 0
                            class_changes[change_key] += 1

                            # Store individual tile change details
                            individual_tile_changes.append(
                                {
                                    "filename": filenames[orig_idx],
                                    "row": r,
                                    "col": c,
                                    "from_class": from_class_name,
                                    "to_class": to_class_name,
                                    "from_confidence": max_confidences[orig_idx],
                                }
                            )

                            # Update the predicted label for the original tile
                            predicted_labels[orig_idx] = final_class

                    start_idx += num_tiles

                # Free memory
                del image, all_contextual_tiles_array

            # Print summary of all class changes
            if class_changes:
                logger.info("\n=== Summary of Class Changes ===")
                total_changes = 0
                for change, count in sorted(class_changes.items()):
                    logger.info(f"{change}: {count} changes")
                    total_changes += count
                logger.info(f"Total changes: {total_changes}")
                logger.info("===============================\n")

                # Save individual tile changes to CSV
                if individual_tile_changes:
                    # Convert list of dictionaries to DataFrame
                    changes_df = pd.DataFrame(individual_tile_changes)

                    # Save to CSV in the results directory
                    csv_path = os.path.join(results_dir, "tile_class_changes.csv")
                    changes_df.to_csv(csv_path, index=False)
                    logger.info(
                        f"Saved {len(individual_tile_changes)} individual tile changes \
                        to {csv_path}"
                    )
            else:
                logger.info("No class changes occurred after max voting.")
        else:
            logger.info(
                f"Didn't find any predictions with confidence lower than \
                {contextual_confidence_threshold}, so do not use context"
            )

    # Check the shape of the arrays before forwarding them to TestMetrics
    logger.info("Calling TestMetrics and initializing metrics")

    metrics = TestMetrics(
        x_test,
        y_test,
        y_test_labels,
        metrics_collector,
        filenames,
        results_dir,
        predicted_labels,
        true_labels,
        all_probs,
        rows,
        cols,
    )

    logger.info("Allocating metrics variables")
    metrics.get_perfile_metrics()
    logger.info("Generating confusion matrix")
    metrics.get_conf_matrix()
    logger.info("Generating images with incorrect tiles per image")
    metrics.get_pred_by_file()
    logger.info("Generating images with incorrect tiles per class")
    metrics.get_pred_by_class()
    logger.info("Generating number of tiles per confidence threshold")
    metrics.get_num_tiles_per_confidence()
    logger.info("Calculating test metrics per confidence threshold")
    metrics.get_metrics_with_threshold_comparison()

    logger.info("Saving metrics function.")
    AmfSave.save_metrics(metrics.metrics_collector, results_dir)

    # End timing
    end_time = time.time()
    execution_time = end_time - start_time
    logger.debug(f"Execution Time: {execution_time:.2f} seconds")


def run(input_images: list[str]) -> int:
    """
    Runs prediction on a bunch of images.

    :param input_images: input images to use for predictions.
    :param save: indicate whether results should be saved or returned.
    """

    model = AmfModel.load()

    # Assign correct device, depending on cpu or gpu
    device = AmfConfig.get("device")
    model = model.to(device)

    metrics_collector = MetricsCollector()
    parts = AmfLoad.categorise_path(input_images)

    test_images = parts["test"]
    if len(test_images) == 0:
        logger.error(
            "No images found. Test data must be separated into a 'test' subfolder"
        )
        # ERR_NO_DATA = 10
        sys.exit(10)

    # Create timestamped folder for results
    results_dir = AmfConfig.get("outdir")
    # print(f"Results Directory: {results_dir}")
    logger.info(f"Results Directory: {results_dir}")

    # Creating dataset
    dataset_loader = AmfLoad.TileFilesandData(test_images)
    x_test, y_test, filenames, rows, cols = dataset_loader.get_all_data()
    test_dataset = AmfLoad.CustomNormalisedDataset(x_test, y_test)
    bs = AmfConfig.get("batch_size")
    num_workers = AmfConfig.get("num_workers")
    test_loader = DataLoader(
        test_dataset, batch_size=bs, shuffle=False, num_workers=num_workers
    )

    get_test_results(
        x_test,
        y_test,
        filenames,
        results_dir,
        model,
        test_loader,
        metrics_collector,
        rows,
        cols,
    )

    return 200
