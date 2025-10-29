# AMFinder - amfinder_predict.py
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
Predicts fungal colonisation (level 1 model) and intraradical hyphal structures (CNN2).

Functions
------------

:function predict_level2: CNN2 predictions.
:function predict_level1: level 1 model predictions.
:function run: main prediction function.
"""

import os
import re
import pandas as pd
import numpy as np
from itertools import zip_longest
import random
from datetime import datetime
import torch.nn.functional as F
import torch
from collections import Counter

random.seed(42)

import random
import amfinder_log as AmfLog
import amfinder_save as AmfSave
import amfinder_model as AmfModel
import amfinder_config as AmfConfig
import amfinder_segmentation as AmfSegm


def table_header():

    return ["row", "col"] + AmfConfig.get("header")


def process_row_1(model, image, nrows, ncols, r, temperature_factor):
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
    row = [AmfSegm.tile(image, r, c) for c in range(ncols)]
    row = AmfSegm.preprocess(row)  # Normalize the tiles.

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


def predict_level2(path, image, nrows, ncols, model):
    """
    Identifies AM fungal structures in colonised root segments.

    :param path: path to the input image.
    :param image: input image (to extract tiles).
    :param nrows: row count.
    :param ncols: column count.
    :para model: CNN2 model used for predictions.
    """

    image_name = os.path.splitext(os.path.basename(path))[0]
    directory = os.path.dirname(path)
    files = os.listdir(directory)

    annotation_type = "cnn_1_annotations"
    regex_pattern = f"{image_name}_.+{annotation_type}"

    # Collect matching annotation files
    matching_annotations = [file for file in files if re.match(regex_pattern, file)]

    # Ensure exactly one matching annotation file exists
    if len(matching_annotations) != 1:
        AmfLog.error(
            f"The dir {directory} does not contain "
            "stage 1 annotations (fungal colonisation)",
            AmfLog.ERR_MISSING_ANNOTATIONS,
        )
        raise ValueError(
            f"Expected exactly one annotation file, found: {len(matching_annotations)}"
        )

    annotations = pd.read_csv(os.path.join(directory, matching_annotations[0]))

    # Retrieve tiles corresponding to arbuscular colonised root segments.
    colonised = annotations.loc[annotations["AMColonised"] == 1, ["row", "col"]]
    colonised = [x for x in colonised.values.tolist()]

    # Create tile batches.
    batches = zip_longest(*(iter(colonised),) * 25)
    nbatches = len(colonised) // 25 + int(len(colonised) % 25 != 0)

    def process_batch(batch, b):
        batch = [x for x in batch if x is not None]
        # First, extract all tiles from the batch.
        row = [AmfSegm.tile(image, x[0], x[1]) for x in batch]
        row = AmfSegm.preprocess(row)
        # Returns three prediction tables (one per class).
        prd = model.predict(row, batch_size=25, verbose=0)
        # Converts to a table of predictions.
        ap = prd[0].tolist()
        vp = prd[1].tolist()
        hp = prd[2].tolist()
        ip = prd[3].tolist()
        dat = [[a[0], v[0], h[0], i[0]] for a, v, h, i in zip(ap, vp, hp, ip)]
        # AmfMapping.generate(cams, model, row, batch)
        res = [[x[0], x[1], y[0], y[1], y[2], y[3]] for (x, y) in zip(batch, dat)]
        AmfLog.progress_bar(b, nbatches, indent=1)
        return pd.DataFrame(res)

    AmfLog.progress_bar(0, nbatches, indent=1)
    results = [process_batch(x, b) for x, b in zip(batches, range(1, nbatches + 1))]

    table = None
    if len(results) > 0:
        table = pd.concat(results, ignore_index=True)
        table.columns = table_header()

    return table


def apply_contextual_confidence(table, image, model, base):
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
            f"Found 0 predictions that are under the context threshold of {contextual_confidence_threshold}, so don't make any changes."
        )
        return table

    AmfLog.info(
        f"Found {len(low_confidence_indices)} predictions that are under the context threshold of {contextual_confidence_threshold}. Applying contextual refinement"
    )

    # Create a mapping from class names to indices
    class_names_to_idx = {name: i for i, name in enumerate(header)}

    # Dictionary to track class changes
    class_changes = {}
    # List to track individual tile changes
    individual_tile_changes = []

    AmfLog.progress_bar(0, len(low_confidence_indices), indent=1)

    # Process each low confidence tile
    for idx, confidence_idx in enumerate(low_confidence_indices):
        row, col = table.loc[confidence_idx, "row"], table.loc[confidence_idx, "col"]

        # Get surrounding tiles
        tiles = AmfSegm.get_contextual_tiles(image, row, col, edge)

        # Filter out None values (beyond image boundaries)
        valid_tiles = [t for t in tiles if t is not None]

        if not valid_tiles:
            continue

        # Preprocess tiles
        valid_tiles = AmfSegm.preprocess(valid_tiles)

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


def predict_level1(image, nrows, ncols, model, temperature_factor, base):
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


# TODO: Save conv2d_outputs currently disabled. Reintegration necessary.
# def save_conv2d_outputs(model, image, base):
#     """
#     Save outputs of each Conv2D layer.
#     Note: currently only works for a single tile.
#     """

#     cmap = plt.get_cmap(AmfConfig.get("colormap"))
#     submodels = AmfModel.get_feature_extractors(model)

#     zipf = "{}_layer_outputs.zip".format(os.path.splitext(base)[0])
#     zipf = os.path.join(AmfConfig.get("outdir"), zipf)

#     with zf.ZipFile(zipf, "w") as z:

#         for conv2d, submodel in submodels:

#             tiles = [AmfSegm.tile(image, 0, 0)]  # TODO: generalise!
#             batch = AmfSegm.preprocess(tiles)

#             predictions = submodel.predict(batch)

#             for i in range(predictions.shape[0]):

#                 im = predictions[i]

#                 for channel in range(im.shape[-1]):

#                     tmp = cmap(im[:, :, channel])
#                     tmp = Image.fromarray(np.uint8(tmp * 255))
#                     tmp = tmp.convert("RGB")
#                     bytes = io.BytesIO()
#                     tmp.save(bytes, "JPEG", quality=100)
#                     # Should add i in filename.
#                     filename = "{}/channel_{}.jpg".format(conv2d.name, channel)
#                     z.writestr(filename, bytes.getvalue())

# TODO: Functionality currently disabled. Reintegration necessary.
# def save_conv2d_kernels(model):
#     """
#     Save kernels for all convolutional layers.
#     """

#     cmap = plt.get_cmap(AmfConfig.get("colormap"))
#     base = os.path.basename(AmfConfig.get("model"))
#     zipf = "{}_kernels.zip".format(os.path.splitext(base)[0])
#     zipf = os.path.join(AmfConfig.get("outdir"), zipf)

#     with zf.ZipFile(zipf, "w") as z:

#         iterations = 30
#         learning_rate = 10.0

#         for conv2d, submodel in AmfModel.get_feature_extractors(model):

#             for filter_index in range(conv2d.output.shape[3]):

#                 loss, img = AmfCalc.visualize_filter(submodel, filter_index)

#                 tmp = Image.fromarray(np.uint8(img * 255))
#                 tmp = tmp.convert("RGB")
#                 bytes = io.BytesIO()
#                 tmp.save(bytes, "JPEG", quality=100)
#                 # Should add i in filename.
#                 filename = "{}/filter_{}.jpg".format(conv2d.name, filter_index)
#                 z.writestr(filename, bytes.getvalue())


def prepare_metrics(
    path: str, tile_results_table: pd.DataFrame, level: int, include_hybrid: bool = True
) -> dict:

    tile_results_table.drop("ContextualLabel", axis=1, inplace=True)
    """
    Generate a dictionary of summary metrics for an image,
    from the DataFrame of individual tile results

    Args:
        path (str): full image path
        tile_results_table (pd.DataFrame): DataFrame of individual tile results
        level: results level 1 or 2

    Returns:
        dict: summary metrics for this image
    """
    directory, filename = os.path.split(path)
    results_dict = {
        "source": directory,
        "file": filename,
    }

    if (level == 1) or (level == 2):
        col_headers = AmfConfig.get("header")
    else:
        raise ValueError("Unknown level")

    class_predictions = tile_results_table.loc[:, col_headers].idxmax(axis=1)
    class_totals = class_predictions.value_counts()
    for col in col_headers:
        if col in class_totals.keys():
            results_dict[col] = class_totals[col]
        else:
            results_dict[col] = 0

    # Calculate % colonised for different classes
    if level == 1:

        col_type = AmfConfig.get("colonisation_type")

        if col_type == "am":
            # Save number of root tilesclasses in results_dict
            # Takes the numer of tiles for each class from results_dict and sums them up, substracts background images and unreadable
            # The first two values of results_dict are exclude, since they are defined as strings
            total_root_tiles = (
                sum(value for _, value in list(results_dict.items())[2:])
                - results_dict[col_headers[2]]  # Remove class Background
                - results_dict[col_headers[3]]  # Remove class Unreadable
            )

            # Ensures that the Hybrid class is included in or excluded from the percentage calculation depending on the include_hybrid parameter
            hybrid_addition = results_dict[col_headers[5]] if include_hybrid else 0

            # Calculate % AM colonised as a ratio to all root tiles
            results_dict["am_colonised_percentage"] = (
                100
                * (results_dict[col_headers[0]] + hybrid_addition)
                / total_root_tiles
            )

            # Calculate % DSE colonised as a ratio to all root tiles
            results_dict["dse_colonised_percentage"] = (
                100
                * (results_dict[col_headers[4]] + hybrid_addition)
                / total_root_tiles
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

            # Ensures that the Hybrid class is included in or excluded from the percentage calculation depending on the include_hybrid parameter
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
                100
                * (results_dict[col_headers[7]] + hybriddse_addition)
                / total_root_tiles
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


def write_metrics(collated_metrics: list, timestamp_string: str, folder: str):
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


def run(input_images, postprocess=None):
    """
    Runs prediction on a bunch of images.

    :param input_images: input images to use for predictions.
    :param save: indicate whether results should be saved or returned.
    """

    if AmfConfig.get("level") == 2 and AmfConfig.get("colonisation_type") == "erm":
        AmfLog.error(
            "There is no CNN2 for ErM colonisation, so fail.",
            AmfLog.ERR_INVALID_ANNOTATION_LEVEL,
        )

        return 500

    if AmfConfig.get("level") == 2:
        AmfLog.error(
            "Currently CNN2 predictions are not supported",
            AmfLog.ERR_INVALID_MODEL,
        )

        return 500

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
    temperature_factor = 1
    if temperature_factor_file is not None:
        path = os.path.join(
            AmfConfig.get_appdir(), "trained_networks", temperature_factor_file
        )

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
        # edge = AmfConfig.update_tile_edge(path)
        edge = AmfConfig.get("tile_edge")

        image = AmfSegm.load(path)

        width, height = image.size

        nrows = height // edge
        ncols = width // edge

        if nrows == 0 or ncols == 0:

            AmfLog.warning("Tile size ({edge} pixels) is too large")
            continue

        else:

            if AmfConfig.get("level") == 1:
                # run the model and make predictions on a single test image
                # and produce a table of predictions including the position of the
                # tile corresponding to those predictions.
                table, class_changes = predict_level1(
                    image, nrows, ncols, model, temperature_factor, base
                )
                class_changes_total.append(class_changes)

            else:
                AmfLog.error(
                    "Level 2 predictions are currently not supported.",
                    exit_code=AmfLog.ERR_INVALID_ANNOTATION_LEVEL,
                )
                # table = predict_level2(path, image, nrows, ncols, model)

            # Save results or use continuation for further processing.
            if postprocess is None:

                # None was cams, reuse for super-resolution.
                AmfSave.prediction_table(table, path)
                AmfLog.info("Preparing metrics for prediction output")
                these_metrics = prepare_metrics(
                    path, table, level=AmfConfig.get("level")
                )
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


def sample_class(row):
    # First two entries are row and col
    probs = row.values[2:]
    # Make sure probabilities add to 1
    # NOTE: Currently a safety measure, shouldn't be needed in future versions
    probs = probs / sum(probs)
    return np.random.choice(range(len(probs)), p=probs)


def bootstrap_distribution(df_cal_probs, n_samples=1000):
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


def get_relative_conf_intervals(bootstrap_distribution, include_hybrid, metric):

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
                "Not a possible metric. Choose either am_colonised_percentage, dse_colonised_percentage or total_colonised_percentage"
            )

    else:
        numerator = 0
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
                "Not a possible metric. Choose either BlueCoils_colonised_percentage, BrownCoils_colonised_percentage, TypeTwo_colonised_percentage, dse_colonised_percentage or total_colonised_percentage"
            )

    percentages = np.where(denominator != 0, (numerator / denominator), np.nan)

    mean = np.nanmean(percentages)
    stddev = np.nanstd(percentages)

    res_dict = {}
    res_dict[metric + "_Mean"] = max(mean, 0) * 100
    res_dict[metric + "_LC"] = max(mean - 2 * stddev, 0) * 100
    res_dict[metric + "_UC"] = min(mean + 2 * stddev, 1) * 100

    return res_dict


def add_conf_intervals(bootstrap_distribution, metrics_dict, include_hybrid):

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


def get_num_tiles_per_confidence(df_cal_probs):

    max_probs = df_cal_probs.iloc[:, 2:].max(axis=1)
    thresholds = np.linspace(0.1, 1.0, 10)
    counts, _ = np.histogram(max_probs, bins=np.concatenate(([0], thresholds)))

    cumulative_count_dist = np.cumsum(counts)
    return_dict = {
        f"Tiles with confidence <= {threshold:.1f}": count
        for threshold, count in zip(thresholds, cumulative_count_dist)
    }

    return return_dict
