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
from collections import Counter
from datetime import datetime
from typing import Any, Callable, cast

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


def process_row_1(
    model: torch.nn.Module,
    image: Image.Image,
    nrows: int,
    ncols: int,
    r: int,
    temperature_factor: float,
) -> pd.DataFrame:
    """
    Predict colonisation (level 1 model) on a single tile row with PyTorch.
    :param model: level 1 model PyTorch model for predictions.
    :param image: Input image to extract tiles.
    :param nrows: Total number of rows in the image.
    :param ncols: Total number of columns in the image.
    :param r: Current row index.
    :param temperature_factor: Factor used to scale probabilities.
    :return: DataFrame of predictions.
    """
    # Make sure device is available.
    device = AmfConfig.get("device")

    # First, extract all tiles within a row.
    row_unnormalised = [AmfSegm.tile(image, r, c) for c in range(ncols)]
    row = AmfSegm.preprocess(row_unnormalised)  # Normalize the tiles.

    # Convert to PyTorch tensor
    row_tensor = torch.tensor(row, dtype=torch.float32)
    row_tensor = row_tensor.to(device)  # Move to the same device as the model

    # Predict using the model
    with torch.no_grad():  # No gradient tracking for inference
        output = model(row_tensor)  # Get predictions from the model

    # Temperature scaling: scale logits using the temperature factor
    scaled_output = output / temperature_factor

    # Apply softmax to convert logits to probabilities if needed
    output = F.softmax(
        scaled_output, dim=1
    )  # Assuming output is not already in probability form

    # Update the progress bar
    AmfLog.progress_bar(r + 1, nrows, indent=1)

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


def predict_level1(
    image: Image.Image,
    nrows: int,
    ncols: int,
    model: torch.nn.Module,
    temperature_factor: float,
    base: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identifies colonised root segments using PyTorch model.
    :param image: input image (to extract tiles).
    :param nrows: row count.
    :param ncols: column count.
    :param model: trained level 1 model used for predictions.
    :param temperature_factor: scaling factor for probabilities.
    :param base: image name.
    :return: Table with predictions and changes made by contextual predictions.
    """

    # Initialize the progress bar.
    AmfLog.progress_bar(0, nrows, indent=1)

    # Retrieve predictions within the image row by row.
    results = []  # List to hold results for each row
    for r in range(nrows):
        # Process each row
        result_df = process_row_1(model, image, nrows, ncols, r, temperature_factor)
        results.append(result_df)

    # Concatenate to a single Pandas dataframe
    table = pd.concat(results, ignore_index=True)

    col_values = list(range(ncols)) * nrows
    row_values = [x // ncols for x in range(nrows * ncols)]

    table.insert(0, column="col", value=col_values)
    table.insert(0, column="row", value=row_values)
    table.columns = table_header()

    table["ContextualLabel"] = None
    use_contextual_confidence = AmfConfig.get("use_contextual_confidence")
    class_changes = pd.DataFrame()
    if use_contextual_confidence:
        AmfLog.info("Applying contextual refinement of predictions.")
        table, class_changes = apply_contextual_confidence(table, image, model, base)

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
        # Takes the numer of tiles for each class from results_dict and sums them up,
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
