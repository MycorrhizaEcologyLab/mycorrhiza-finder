# AMFinder - predict.py
#
# MIT License
# Copyright (c) 2021 Edouard Evangelisti, Carl Turner
#               2024-2025 Royal Botanic Gardens, Kew
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.

"""
Predicts fungal colonisation.

Functions
------------

:function predict_level1: level 1 model predictions.
:function run: main prediction function.
"""

import os
import random
import time
from collections import Counter
from datetime import datetime
from typing import Any, Callable, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from PIL import Image

import amf.helper.config as AmfConfig
import amf.helper.log as AmfLog
import amf.helper.model as AmfModel
import amf.helper.save as AmfSave
import amf.helper.segmentation as AmfSegm

random.seed(42)


def table_header() -> list[str]:
    return ["row", "col"] + cast(list[str], AmfConfig.get("header"))


def process_batch(
    model: torch.nn.Module,
    batch_unnormalised: NDArray,
    temperature_factor: float,
) -> pd.DataFrame:
    """
    Predict colonisation (level 1 model) on a batch of tiles with PyTorch.
    :param model: level 1 model PyTorch model for predictions.
    :param image: Input image to extract tiles from.
    :param tiles_coords: List of (row, col) tuples for tiles in this batch.
    :param temperature_factor: Factor used to scale probabilities.
    :return: DataFrame of predictions.
    """
    # Make sure device is available.
    device = AmfConfig.get("device")

    # Extract tiles for this batch.
    # batch_unnormalised = [AmfSegm.tile(image, r, c) for r, c in tiles_coords]
    batch = AmfSegm.preprocess(batch_unnormalised)  # Normalize the tiles.

    # Convert to PyTorch tensor
    batch_tensor = torch.tensor(batch, dtype=torch.float32)
    batch_tensor = batch_tensor.to(device)  # Move to the same device as the model

    # Predict using the model
    with torch.no_grad():  # No gradient tracking for inference
        output = model(batch_tensor)  # Get predictions from the model

    # Temperature scaling: scale logits using the temperature factor
    scaled_output = output / temperature_factor

    # Apply softmax to convert logits to probabilities if needed
    output = F.softmax(
        scaled_output, dim=1
    )  # Assuming output is not already in probability form

    # Convert prediction tensor back to DataFrame
    return pd.DataFrame(output.cpu().numpy())


def apply_contextual_confidence(
    table: pd.DataFrame, image: Image.Image, model: torch.nn.Module, base: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply contextual confidence to improve predictions for low confidence tiles.

    :param table: DataFrame with initial predictions.
    :param image: The source image.
    :param temperature_factor: Temperature factor for scaling logits.
    :param model: The trained model for predictions.
    :param base: Base path for image.
    :return: Updated table with refined predictions.
    """

    contextual_confidence_threshold = AmfConfig.get("contextual_confidence_threshold")
    device = AmfConfig.get("device")
    edge = AmfConfig.get("tile_edge")
    header = AmfConfig.get("header")

    # Get the class indices for each prediction (highest probability class)
    predicted_classes = table.iloc[:, 2:-1].idxmax(axis=1)
    # Get the maximum probability for each prediction
    max_confidences = table.iloc[:, 2:-1].max(axis=1)

    # Find low confidence predictions
    low_confidence_indices = max_confidences[
        max_confidences <= contextual_confidence_threshold
    ].index

    if len(low_confidence_indices) == 0:
        # No low confidence predictions to refine
        AmfLog.info(
            "Found 0 predictions that are under the context threshold of "
            f"{contextual_confidence_threshold}, so don't make any changes."
        )
        return table, pd.DataFrame()

    AmfLog.info(
        f"Found {len(low_confidence_indices)} predictions that are under the context "
        f"threshold of {contextual_confidence_threshold}. "
        "Applying contextual refinement."
    )

    # Create a mapping from class names to indices
    class_names_to_idx = {name: i for i, name in enumerate(header)}

    # Dictionary to track class changes
    class_changes: dict[str, int] = {}
    # List to track individual tile changes
    individual_tile_changes = []

    AmfLog.progress_bar(0, len(low_confidence_indices), indent=1)

    # Process each low confidence tile
    for idx, confidence_idx in enumerate(low_confidence_indices):
        row, col = table.loc[confidence_idx, "row"], table.loc[confidence_idx, "col"]

        # Get surrounding tiles
        tiles = AmfSegm.get_contextual_tiles(image, row, col, edge)

        # Filter out None values (beyond image boundaries)
        valid_tiles_unnormalised = [t for t in tiles if t is not None]

        if not valid_tiles_unnormalised:
            continue

        # Preprocess tiles
        valid_tiles = AmfSegm.preprocess(valid_tiles_unnormalised)

        # Convert to PyTorch tensor
        tiles_tensor = torch.tensor(valid_tiles, dtype=torch.float32).to(device)

        # Make predictions
        with torch.no_grad():
            outputs = model(tiles_tensor)
            probs = F.softmax(outputs, dim=1)
            surroundings_pred_classes = torch.argmax(probs, dim=1).cpu().numpy()

        # Get the original prediction class (as index)
        orig_class_name = predicted_classes[confidence_idx]
        orig_class_idx = class_names_to_idx[orig_class_name]

        # Add the central tile's prediction to the list
        central_tile_with_neighborhood_preds = [
            orig_class_idx
        ] + surroundings_pred_classes.tolist()

        # Use majority voting to determine the final class
        final_class_idx = Counter(central_tile_with_neighborhood_preds).most_common(1)[
            0
        ][0]
        final_class_name = header[final_class_idx]

        # Update prediction if changed
        if final_class_name != orig_class_name:
            # Update the tracking dictionary
            change_key = f"{orig_class_name} -> {final_class_name}"
            class_changes[change_key] = class_changes.get(change_key, 0) + 1

            # Store individual tile change details
            individual_tile_changes.append(
                {
                    "file": base,
                    "row": row,
                    "col": col,
                    "from_class": orig_class_name,
                    "to_class": final_class_name,
                    "confidence": max_confidences[confidence_idx],
                }
            )

            table.at[confidence_idx, "ContextualLabel"] = final_class_name

        AmfLog.progress_bar(idx + 1, len(low_confidence_indices), indent=1)

    changes_df = pd.DataFrame()
    # Print summary of class changes
    if class_changes:
        AmfLog.info("\n=== Summary of Class Changes ===")
        total_changes = 0
        for change, count in sorted(class_changes.items()):
            AmfLog.info(f"{change}: {count} changes")
            total_changes += count
        AmfLog.info(f"Total changes: {total_changes}")
        AmfLog.info("===============================\n")

        # Save individual tile changes to CSV if there are any
        if individual_tile_changes:
            changes_df = pd.DataFrame(individual_tile_changes)
    else:
        AmfLog.info("No class changes occurred after contextual refinement.")

    return table, changes_df


def tile_entire_image(
    image: Image.Image, nrows: int, ncols: int
) -> list[NDArray[np.uint8]]:
    """
    Extract all tiles from an image along with their position.

    :param image: The source image.
    :param nrows: Number of rows of tiles.
    :param ncols: Number of columns of tiles.
    :return: List of tiles as NumPy arrays.
    """
    tiles = []
    for r in range(nrows):
        for c in range(ncols):
            tile = AmfSegm.tile(image, r, c)
            tiles.append([tile, r, c])
    return tiles


def threshold_empty_tiles(
    tiles: list[NDArray[np.uint8]],
    threshold: float,
) -> list[NDArray[np.uint8]]:
    """
    Function to assign tiles to either background or root category based on mean pixel
    intensity.

    :param tiles: List of tiles as numpy arrays with their row and column indices.
    :param threshold: Mean pixel intensity threshold to classify a tile as background.
    :return: Lists of background and root tiles as separate numpy arrays.
    """
    background_tiles = []
    root_tiles = []
    # print(tiles.shape)
    # print(tiles)
    # exit()
    for [tile, r, c] in tiles:
        # Calculate mean pixel intensity across tile
        mean_intensity = np.mean(tile) / 255.0  # Normalise to [0, 1]
        if mean_intensity >= threshold:
            background_tiles.append([tile, r, c])
        else:
            root_tiles.append([tile, r, c])

    return background_tiles, root_tiles


def visualise_tile_intensity_distribution(
    image: Image.Image,
    nrows: int,
    ncols: int,
    threshold: float,
    max_width: int = 8192,
) -> None:
    """
    Visualise the distribution of mean pixel intensities across all tiles.
    Creates three plots:
    1. Histogram of intensity distribution with statistics
    2. Spatial heatmap of tile intensities overlayed on the image
    3. Threshold classification overlay showing background vs root tiles

    :param image: The source image.
    :param nrows: Number of rows of tiles.
    :param ncols: Number of columns of tiles.
    :param threshold: Threshold for classifying background tiles (0-1).
    :param max_width: Maximum width for resized image (default 1024 pixels).
    :return: None (saves visualisation as .png file).
    """
    all_tiles = tile_entire_image(image, nrows, ncols)
    background_tiles, root_tiles = threshold_empty_tiles(all_tiles, threshold=threshold)
    print(f"Image width: {image.width}")

    # Resize image for visualization while preserving aspect ratio
    scale_factor = max_width / image.width if image.width > max_width else 1.0
    if scale_factor < 1.0:
        new_width = int(image.width * scale_factor)
        new_height = int(image.height * scale_factor)
        image_resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    else:
        image_resized = image
        scale_factor = 1.0

    # Calculate mean intensities for all tiles
    all_intensities = []
    intensity_grid = np.zeros((nrows, ncols))
    threshold_grid = np.zeros((nrows, ncols, 4))  # RGBA for transparency

    # Create set of background tile positions for quick lookup
    background_positions = {(r, c) for tile, r, c in background_tiles}

    for tile, r, c in all_tiles:
        mean_intensity = np.mean(tile) / 255.0
        all_intensities.append(mean_intensity)
        intensity_grid[r, c] = mean_intensity

        # Create threshold visualization grid
        if (r, c) in background_positions:
            # Background tile: solid color (gray), mostly opaque
            threshold_grid[r, c] = [0.5, 0.5, 0.5, 0.7]  # Gray with 70% opacity
        else:
            # Root tile: transparent
            threshold_grid[r, c] = [0, 0, 0, 0]  # Fully transparent

    # Get min and max intensity values for heatmap scaling
    min_intensity = np.min(all_intensities)
    max_intensity = np.max(all_intensities)

    edge = AmfConfig.get("tile_edge")
    edge_resized = int(edge * scale_factor)

    # Create visualization with three subplots
    fig = plt.figure(figsize=(180, 60))

    # First subplot: Histogram
    ax1 = plt.subplot(1, 3, 1)
    ax1.hist(all_intensities, bins=50, alpha=0.7, label="All tiles", edgecolor="black")
    ax1.axvline(
        threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Threshold ({threshold})",
    )
    ax1.set_xlabel("Mean Pixel Intensity (normalized to [0, 1])")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Distribution of Mean Pixel Intensities")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Second subplot: Spatial heatmap
    ax2 = plt.subplot(1, 3, 2)

    # Display the resized image
    ax2.imshow(image_resized, extent=[0, image_resized.width, image_resized.height, 0])

    # Create heatmap overlay of tile intensities
    im = ax2.imshow(
        intensity_grid,
        cmap="RdYlGn",
        alpha=0.4,
        extent=[0, image_resized.width, image_resized.height, 0],
        vmin=min_intensity,
        vmax=max_intensity,
    )

    # Draw tile grid lines
    for r in range(nrows + 1):
        ax2.axhline(y=r * edge_resized, color="gray", linewidth=0.5, alpha=0.5)
    for c in range(ncols + 1):
        ax2.axvline(x=c * edge_resized, color="gray", linewidth=0.5, alpha=0.5)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2, label="Mean Pixel Intensity")

    ax2.set_xlabel("X Coordinate (pixels)")
    ax2.set_ylabel("Y Coordinate (pixels)")
    ax2.set_title("Spatial Distribution of Tile Intensities")

    # Third subplot: Threshold classification overlay
    ax3 = plt.subplot(1, 3, 3)

    # Display the resized image
    ax3.imshow(image_resized, extent=[0, image_resized.width, image_resized.height, 0])

    # Overlay threshold classification (background tiles in gray, root tiles transparent)
    threshold_grid_resized = np.zeros((image_resized.height, image_resized.width, 4))
    for r in range(nrows):
        for c in range(ncols):
            y_start = int(r * edge_resized)
            y_end = int((r + 1) * edge_resized)
            x_start = int(c * edge_resized)
            x_end = int((c + 1) * edge_resized)
            # Only set alpha for background tiles; root tiles remain transparent (alpha=0)
            threshold_grid_resized[y_start:y_end, x_start:x_end] = threshold_grid[r, c]

    ax3.imshow(
        threshold_grid_resized, extent=[0, image_resized.width, image_resized.height, 0]
    )

    # Draw tile grid lines
    # for r in range(nrows + 1):
    #     ax3.axhline(y=r * edge_resized, color="gray", linewidth=0.5, alpha=0.5)
    # for c in range(ncols + 1):
    #     ax3.axvline(x=c * edge_resized, color="gray", linewidth=0.5, alpha=0.5)

    ax3.set_xlabel("X Coordinate (pixels)")
    ax3.set_ylabel("Y Coordinate (pixels)")
    ax3.set_title("Threshold Classification (Gray = Background)")

    # Add summary statistics as text
    stats_text = (
        f"Total tiles: {len(all_tiles)}\n"
        f"Background: {len(background_tiles)} ({100 * len(background_tiles) / len(all_tiles):.1f}%)\n"
        f"Root tiles: {len(root_tiles)} ({100 * len(root_tiles) / len(all_tiles):.1f}%)\n"
        f"Threshold: {threshold}"
    )
    ax3.text(
        0.02,
        0.98,
        stats_text,
        transform=ax3.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        family="monospace",
    )

    plt.tight_layout()
    plt.savefig(f"Background Vis {int(threshold * 1000)} v4.png")
    plt.close()


def predict_level1(
    image: Image.Image,
    nrows: int,
    ncols: int,
    model: torch.nn.Module,
    temperature_factor: float,
    base: str,
    threshold: float = 0.99,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identifies colonised root segments using PyTorch model.
    :param image: input image (to extract tiles).
    :param nrows: row count.
    :param ncols: column count.
    :param model: trained level 1 model used for predictions.
    :param temperature_factor: scaling factor for probabilities.
    :param base: image name.
    :param threshold: threshold for classifying background tiles (0-1)
    :return: Table with predictions and changes made by contextual predictions.
    """

    # Get batch size from config
    batch_size = AmfConfig.get("batch_size")

    # DEBUG: Enable to visualise which tiles are thresholded as background
    # visualise_tile_intensity_distribution(image, nrows, ncols, threshold=threshold)

    all_tiles = tile_entire_image(image, nrows, ncols)
    # Threshold out background tiles on mean pixel intensity, 0.985 is a good balance
    background_tiles, root_tiles = threshold_empty_tiles(all_tiles, threshold=threshold)

    total_tiles = len(root_tiles)

    # Initialize the progress bar.
    AmfLog.progress_bar(0, total_tiles, indent=1)

    # Process tiles in batches

    results = []  # List to hold results for each batch
    rows = []  # List to store row indices
    cols = []  # List to store column indices

    b_results = []  # List to hold background results if they
    b_rows = []
    b_cols = []

    # TODO: Replace with logger messages and delete prints
    # print(f"{len(background_tiles)} background tiles identified and autoclassified.")
    # print(f"Processing {total_tiles} tiles in batches of {batch_size}...")

    # DEBUG: Enable for tracking total execution time of inference over a root image
    # start_time = time.time()

    if len(background_tiles) > 0:
        for tile, r, c in background_tiles:
            background_pred = np.zeros(len(AmfConfig.get("header")))
            background_pred[2] = 1.0
            b_results.append(pd.DataFrame([background_pred]))
            b_rows.append(r)
            b_cols.append(c)

    for batch_start in range(0, total_tiles, batch_size):
        batch_end = min(batch_start + batch_size, total_tiles)
        batch = root_tiles[batch_start:batch_end]

        batch_tiles = []
        # Store coordinates for this batch
        for tile, r, c in batch:
            batch_tiles.append(tile)
            rows.append(r)
            cols.append(c)

        # Process this batch
        batch_result = process_batch(model, batch_tiles, temperature_factor)
        results.append(batch_result)

        # Update the progress bar
        AmfLog.progress_bar(batch_end, total_tiles, indent=1)

    # TODO: Replace with logger message if needed and delete print
    # print("--- %s seconds ---" % (time.time() - start_time))

    # Concatenate to a single Pandas dataframe
    table = pd.concat(results, ignore_index=True)

    table.insert(0, column="col", value=cols)
    table.insert(0, column="row", value=rows)
    table.columns = table_header()

    table["ContextualLabel"] = None
    use_contextual_confidence = AmfConfig.get("use_contextual_confidence")
    class_changes = pd.DataFrame()
    if use_contextual_confidence:
        AmfLog.info("Applying contextual refinement of predictions.")
        table, class_changes = apply_contextual_confidence(table, image, model, base)

    if len(b_results) > 0:
        b_table = pd.concat(b_results, ignore_index=True)
        b_table.insert(0, column="col", value=b_cols)
        b_table.insert(0, column="row", value=b_rows)
        b_table.columns = table_header()
        b_table["ContextualLabel"] = None

        table = pd.concat([table, b_table], ignore_index=True)

    return table, class_changes


def prepare_metrics(
    path: str, tile_results_table: pd.DataFrame, include_hybrid: bool = True
) -> dict[str, Any]:
    tile_results_table.drop("ContextualLabel", axis=1, inplace=True)
    """
    Generate a dictionary of summary metrics for an image,
    from the DataFrame of individual tile results

    Args:
        path (str): full image path
        tile_results_table (pd.DataFrame): DataFrame of individual tile results

    Returns:
        dict: summary metrics for this image
    """
    directory, filename = os.path.split(path)
    results_dict: dict[str, Any] = {
        "source": directory,
        "file": filename,
    }

    col_headers = AmfConfig.get("header")

    class_predictions = tile_results_table.loc[:, col_headers].idxmax(axis=1)
    class_totals = class_predictions.value_counts()
    for col in col_headers:
        if col in class_totals.keys():
            results_dict[col] = class_totals[col]
        else:
            results_dict[col] = 0

    # Calculate % colonised for different classes
    col_type = AmfConfig.get("colonisation_type")

    if col_type == "am":
        # Save number of root tilesclasses in results_dict
        # Takes the number of tiles for each class from results_dict and sums them up,
        # subtracts background images and unreadable
        # The first two values of results_dict are exclude, since they are defined as
        # strings
        total_root_tiles = (
            sum(value for _, value in list(results_dict.items())[2:])
            - results_dict[col_headers[2]]  # Remove class Background
            - results_dict[col_headers[3]]  # Remove class Unreadable
        )

        # Ensures that the Hybrid class is included in or excluded from the percentage
        # calculation depending on the include_hybrid parameter
        hybrid_addition = results_dict[col_headers[5]] if include_hybrid else 0

        # Calculate % AM colonised as a ratio to all root tiles
        results_dict["am_colonised_percentage"] = (
            100 * (results_dict[col_headers[0]] + hybrid_addition) / total_root_tiles
        )

        # Calculate % DSE colonised as a ratio to all root tiles
        results_dict["dse_colonised_percentage"] = (
            100 * (results_dict[col_headers[4]] + hybrid_addition) / total_root_tiles
        )

        # Calculate % total colonised as a ratio to all root tiles
        results_dict["total_colonised_percentage"] = (
            results_dict["am_colonised_percentage"]
            + results_dict["dse_colonised_percentage"]
            - 100 * (hybrid_addition / total_root_tiles)
        )

    else:
        total_root_tiles = (
            sum(value for _, value in list(results_dict.items())[2:])
            - results_dict[col_headers[4]]  # Remove class Background
            - results_dict[col_headers[5]]  # Remove class Main root
            - results_dict[col_headers[6]]  # Remove class Unreadable
        )

        # Ensures that the Hybrid class is included in or excluded from the percentage
        # calculation depending on the include_hybrid parameter
        hybriderm_addition = results_dict[col_headers[8]]
        hybriddse_addition = results_dict[col_headers[9]]

        # Calculate % blue coils colonised as a ratio to all root tiles
        results_dict["BlueCoils_colonised_percentage"] = (
            100 * (results_dict[col_headers[0]]) / total_root_tiles
        )

        # Calculate % brown coils colonised as a ratio to all root tiles
        results_dict["BrownCoils_colonised_percentage"] = (
            100 * (results_dict[col_headers[1]]) / total_root_tiles
        )

        # Calculate % type two colonised as a ratio to all root tiles
        results_dict["TypeTwo_colonised_percentage"] = (
            100 * (results_dict[col_headers[2]]) / total_root_tiles
        )

        # Calculate % DSE colonised as a ratio to all root tiles
        results_dict["dse_colonised_percentage"] = (
            100 * (results_dict[col_headers[7]] + hybriddse_addition) / total_root_tiles
        )

        # Calculate % total colonised as a ratio to all root tiles
        results_dict["total_colonised_percentage"] = (
            results_dict["BlueCoils_colonised_percentage"]
            + results_dict["BrownCoils_colonised_percentage"]
            + results_dict["TypeTwo_colonised_percentage"]
            + 100 * (hybriderm_addition / total_root_tiles)
        )

    # Calculate bootstrap distribution and confidence intervals
    AmfLog.info("Calculating bootstrap distribution")
    bootstr_dist = bootstrap_distribution(tile_results_table)
    AmfLog.info("Calculating confidence intervals")
    results_dict = add_conf_intervals(bootstr_dist, results_dict, include_hybrid)

    # Retrieve cumulative count distribution for max probabilities per tile
    AmfLog.info("Calculating number of tiles per confidence level")
    count_distribution_dict = get_num_tiles_per_confidence(tile_results_table)
    results_dict.update(count_distribution_dict)

    return results_dict


def write_metrics(
    collated_metrics: list[dict[str, Any]], timestamp_string: str, folder: str
) -> None:
    """
    Write metrics for all processed images to a metrics file

    Args:
        collated_metrics (list): list of metrics dicts (one per image)
        timestamp_string (list): timestamp formatted as "%Y%m%d_%H%M%S"
        folder (str): path to write file to
    """
    os.makedirs(folder, exist_ok=True)
    metrics_filepath = os.path.join(folder, timestamp_string + "_results.csv")
    AmfLog.info(f"Saving metrics as {metrics_filepath}... ", end="")

    try:
        output_df = pd.DataFrame.from_records(collated_metrics)
        col_type = AmfConfig.get("colonisation_type")

        if col_type == "am":
            column_order = [
                "source",
                "file",
                "am_colonised_percentage",
                "am_colonised_percentage_Mean",
                "am_colonised_percentage_LC",
                "am_colonised_percentage_UC",
                "dse_colonised_percentage",
                "dse_colonised_percentage_Mean",
                "dse_colonised_percentage_LC",
                "dse_colonised_percentage_UC",
                "total_colonised_percentage",
                "total_colonised_percentage_Mean",
                "total_colonised_percentage_LC",
                "total_colonised_percentage_UC",
                "AMColonised",
                "AMColonised_Mean",
                "AMColonised_LowerConfidence",
                "AMColonised_UpperConfidence",
                "Uncolonised",
                "Uncolonised_Mean",
                "Uncolonised_LowerConfidence",
                "Uncolonised_UpperConfidence",
                "Background",
                "Background_Mean",
                "Background_LowerConfidence",
                "Background_UpperConfidence",
                "Unreadable",
                "Unreadable_Mean",
                "Unreadable_LowerConfidence",
                "Unreadable_UpperConfidence",
                "DSE",
                "DSE_Mean",
                "DSE_LowerConfidence",
                "DSE_UpperConfidence",
                "Hybrid",
                "Hybrid_Mean",
                "Hybrid_LowerConfidence",
                "Hybrid_UpperConfidence",
                "Tiles with confidence <= 0.1",
                "Tiles with confidence <= 0.2",
                "Tiles with confidence <= 0.3",
                "Tiles with confidence <= 0.4",
                "Tiles with confidence <= 0.5",
                "Tiles with confidence <= 0.6",
                "Tiles with confidence <= 0.7",
                "Tiles with confidence <= 0.8",
                "Tiles with confidence <= 0.9",
                "Tiles with confidence <= 1.0",
            ]

        else:
            column_order = [
                "source",
                "file",
                "BlueCoils_colonised_percentage",
                "BlueCoils_colonised_percentage_Mean",
                "BlueCoils_colonised_percentage_LC",
                "BlueCoils_colonised_percentage_UC",
                "BrownCoils_colonised_percentage",
                "BrownCoils_colonised_percentage_Mean",
                "BrownCoils_colonised_percentage_LC",
                "BrownCoils_colonised_percentage_UC",
                "TypeTwo_colonised_percentage",
                "TypeTwo_colonised_percentage_Mean",
                "TypeTwo_colonised_percentage_LC",
                "TypeTwo_colonised_percentage_UC",
                "dse_colonised_percentage",
                "dse_colonised_percentage_Mean",
                "dse_colonised_percentage_LC",
                "dse_colonised_percentage_UC",
                "total_colonised_percentage",
                "total_colonised_percentage_Mean",
                "total_colonised_percentage_LC",
                "total_colonised_percentage_UC",
                "BlueCoils",
                "BlueCoils_Mean",
                "BlueCoils_LowerConfidence",
                "BlueCoils_UpperConfidence",
                "BrownCoils",
                "BrownCoils_Mean",
                "BrownCoils_LowerConfidence",
                "BrownCoils_UpperConfidence",
                "TypeTwo",
                "TypeTwo_Mean",
                "TypeTwo_LowerConfidence",
                "TypeTwo_UpperConfidence",
                "Uncolonised",
                "Uncolonised_Mean",
                "Uncolonised_LowerConfidence",
                "Uncolonised_UpperConfidence",
                "Background",
                "Background_Mean",
                "Background_LowerConfidence",
                "Background_UpperConfidence",
                "MainRoot",
                "MainRoot_Mean",
                "MainRoot_LowerConfidence",
                "MainRoot_UpperConfidence",
                "Unreadable",
                "Unreadable_Mean",
                "Unreadable_LowerConfidence",
                "Unreadable_UpperConfidence",
                "DSE",
                "DSE_Mean",
                "DSE_LowerConfidence",
                "DSE_UpperConfidence",
                "HybridErm",
                "HybridErm_Mean",
                "HybridErm_LowerConfidence",
                "HybridErm_UpperConfidence",
                "HybridDse",
                "HybridDse_Mean",
                "HybridDse_LowerConfidence",
                "HybridDse_UpperConfidence",
                "Tiles with confidence <= 0.1",
                "Tiles with confidence <= 0.2",
                "Tiles with confidence <= 0.3",
                "Tiles with confidence <= 0.4",
                "Tiles with confidence <= 0.5",
                "Tiles with confidence <= 0.6",
                "Tiles with confidence <= 0.7",
                "Tiles with confidence <= 0.8",
                "Tiles with confidence <= 0.9",
                "Tiles with confidence <= 1.0",
            ]

        output_df.to_csv(
            metrics_filepath,
            encoding="utf-8",
            index=False,
            lineterminator="\n",
            float_format="%g",
            columns=column_order,
        )
    except Exception as e:
        print(f"Errors occured writing the metrics file: {e}")


def run(
    input_images: list[str],
    postprocess: Callable[[Image.Image, pd.DataFrame, str], None] | None = None,
) -> int:
    """
    Runs prediction on a bunch of images.

    :param input_images: input images to use for predictions.
    :param save: indicate whether results should be saved or returned.
    """

    AmfLog.info(f"Number of test images: {len(input_images)}")

    model = AmfModel.load()

    # Assign correct device, depending on cpu or gpu
    device = AmfConfig.get("device")
    model = model.to(device)

    # Set model into evaluation mode
    model.eval()

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
            AmfLog.warning(
                f"Failed to read temperature factor from file {path}, setting to 1"
            )

    AmfLog.info(f"Using temperature factor of {temperature_factor}")

    collated_metrics = []  # Get results from each image together
    class_changes_total = []
    for path in input_images:
        base = os.path.basename(path)
        AmfLog.text(f"Image {base}")

        # Only updates tile edge if settings exist
        edge = AmfConfig.get("tile_edge")

        image = AmfSegm.load(path)

        width, height = image.size

        nrows = height // edge
        ncols = width // edge

        if nrows == 0 or ncols == 0:
            AmfLog.warning("Tile size ({edge} pixels) is too large")
            continue

        else:
            # run the model and make predictions on a single test image
            # and produce a table of predictions including the position of the
            # tile corresponding to those predictions.
            table, class_changes = predict_level1(
                image, nrows, ncols, model, temperature_factor, base
            )
            class_changes_total.append(class_changes)

            # Save results or use continuation for further processing.
            if postprocess is None:
                # None was cams, reuse for super-resolution.
                AmfSave.prediction_table(table, path)
                AmfLog.info("Preparing metrics for prediction output")
                these_metrics = prepare_metrics(path, table)
                collated_metrics.append(these_metrics)

            else:
                postprocess(image, table, path)

    timestamp_string = datetime.now().strftime("%Y%m%d_%H%M%S")
    use_contextual_confidence = AmfConfig.get("use_contextual_confidence")
    if use_contextual_confidence:
        csv_path = os.path.join(
            AmfConfig.get("outdir"), timestamp_string + "_tile_class_changes.csv"
        )
        final_class_changes_df = pd.concat(class_changes_total)
        final_class_changes_df.to_csv(csv_path, index=False)
        AmfLog.info(
            f"Saved {len(final_class_changes_df)} individual tile changes to {csv_path}"
        )

    # Output collated metrics to file
    write_metrics(collated_metrics, timestamp_string, folder=AmfConfig.get("outdir"))

    return 200


def sample_class(row: pd.Series) -> int:
    # First two entries are row and col
    probs = row.values[2:]
    # Make sure probabilities add to 1
    # NOTE: Currently a safety measure, shouldn't be needed in future versions
    probs = probs / sum(probs)
    return cast(
        int, np.random.choice(range(len(probs)), p=probs)
    )  # returns scalar when size=None


def bootstrap_distribution(
    df_cal_probs: pd.DataFrame, n_samples: int = 1000
) -> NDArray[np.float64]:
    # Retrieve number of classes
    n_classes = len(AmfConfig.get("header"))
    AmfLog.progress_bar(0, n_samples, indent=1)

    # Create matrix for storing Monte Carlo results
    # and the bootstrap distribution per class
    mc_samples = np.zeros((n_samples, len(df_cal_probs)), dtype=int)
    bootstrap_dist = np.zeros((n_samples, n_classes))

    # Monte Carlo sampling
    for i in range(n_samples):
        AmfLog.progress_bar(i, n_samples, indent=1)
        mc_samples[i] = df_cal_probs.apply(sample_class, axis=1)
        bootstrap_dist[i] = np.bincount(mc_samples[i], minlength=n_classes)

    return bootstrap_dist


def get_relative_conf_intervals(
    bootstrap_distribution: NDArray[np.float64], include_hybrid: bool, metric: str
) -> dict[str, float]:
    col_type = AmfConfig.get("colonisation_type")

    if col_type == "am":
        numerator = bootstrap_distribution[:, 5] * include_hybrid
        denominator = (
            bootstrap_distribution[:, 0]
            + bootstrap_distribution[:, 1]
            + bootstrap_distribution[:, 4]
            + bootstrap_distribution[:, 5]
        )

        if metric == "am_colonised_percentage":
            numerator += bootstrap_distribution[:, 0]
        elif metric == "dse_colonised_percentage":
            numerator += bootstrap_distribution[:, 4]
        elif metric == "total_colonised_percentage":
            numerator += bootstrap_distribution[:, 0] + bootstrap_distribution[:, 4]
        else:
            raise ValueError(
                "Not a possible metric. Choose either am_colonised_percentage, "
                "dse_colonised_percentage or total_colonised_percentage"
            )

    else:
        numerator = np.zeros(bootstrap_distribution.shape[0])
        denominator = (
            bootstrap_distribution[:, 0]
            + bootstrap_distribution[:, 1]
            + bootstrap_distribution[:, 2]
            + bootstrap_distribution[:, 3]
            + bootstrap_distribution[:, 7]
            + bootstrap_distribution[:, 8]
            + bootstrap_distribution[:, 9]
        )

        if metric == "BlueCoils_colonised_percentage":
            numerator += bootstrap_distribution[:, 0]
        elif metric == "BrownCoils_colonised_percentage":
            numerator += bootstrap_distribution[:, 1]
        elif metric == "TypeTwo_colonised_percentage":
            numerator += bootstrap_distribution[:, 2]
        elif metric == "dse_colonised_percentage":
            numerator += bootstrap_distribution[:, 7] + bootstrap_distribution[:, 9]
        elif metric == "total_colonised_percentage":
            numerator += (
                bootstrap_distribution[:, 0]
                + bootstrap_distribution[:, 1]
                + bootstrap_distribution[:, 2]
                + bootstrap_distribution[:, 8]
            )
        else:
            raise ValueError(
                "Not a possible metric. Choose either BlueCoils_colonised_percentage, "
                "BrownCoils_colonised_percentage, TypeTwo_colonised_percentage, "
                "dse_colonised_percentage or total_colonised_percentage"
            )

    percentages = np.where(denominator != 0, (numerator / denominator), np.nan)

    mean = np.nanmean(percentages)
    stddev = np.nanstd(percentages)

    res_dict = {}
    res_dict[metric + "_Mean"] = max(mean, 0) * 100
    res_dict[metric + "_LC"] = max(mean - 2 * stddev, 0) * 100
    res_dict[metric + "_UC"] = min(mean + 2 * stddev, 1) * 100

    return res_dict


def add_conf_intervals(
    bootstrap_distribution: NDArray[np.float64],
    metrics_dict: dict[str, Any],
    include_hybrid: bool,
) -> dict[str, Any]:
    # Calculate relevant metrics for creating confidence intervals
    mean = np.mean(bootstrap_distribution, axis=0)
    std_dev = np.std(bootstrap_distribution, axis=0)

    # Calculation of absolute lower and upper bound for each class
    headers = AmfConfig.get("header")
    dict_keys = metrics_dict.keys()
    res = {}

    for header in dict_keys:
        if header in headers:
            i = headers.index(header)
            res[header + "_Mean"] = max(mean[i], 0)
            res[header + "_LowerConfidence"] = max(mean[i] - 2 * std_dev[i], 0)
            res[header + "_UpperConfidence"] = max(mean[i] + 2 * std_dev[i], 0)

        elif "percentage" in header:
            temp = get_relative_conf_intervals(
                bootstrap_distribution, include_hybrid, header
            )
            res.update(temp)

        else:
            continue

    metrics_dict.update(res)

    return metrics_dict


def get_num_tiles_per_confidence(df_cal_probs: pd.DataFrame) -> dict[str, int]:
    max_probs = df_cal_probs.iloc[:, 2:].max(axis=1)
    thresholds = np.linspace(0.1, 1.0, 10)
    counts, _ = np.histogram(max_probs, bins=np.concatenate(([0], thresholds)))

    cumulative_count_dist = np.cumsum(counts)
    return_dict = {
        f"Tiles with confidence <= {threshold:.1f}": count
        for threshold, count in zip(thresholds, cumulative_count_dist)
    }

    return return_dict
