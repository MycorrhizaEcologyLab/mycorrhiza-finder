"""`bald` mode for active learning acquisition.

This can be used to select the optimal samples for labelling using BALD or BatchBALD.
"""

from collections import defaultdict

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
from amf.helper.acquisition_functions import get_batchbald_batch


def bald_acquisition(
    model: torch.nn.Module,
    device: torch.device,
    x_unlabelled: list[NDArray[np.uint8]],
    num_samples_for_labelling: int,
    num_classes: int,
    mc_samples: int = 50,
    batch_size: int = 32,
) -> list[int]:
    """Return the indices of the most uncertain samples using BALD.

    Implements BALD acquisition function using MC Dropout for uncertainty estimation.

    Args:
        model: Neural network model.
        device: The device (CPU/GPU) to run the model on.
        x_unlabelled: Unlabelled data points for selection.
        num_samples_for_labelling: Number of most uncertain samples to return.
        num_classes: Number of output classes.
        mc_samples: Number of stochastic forward passes.
        batch_size: Batch size for efficient memory management.

    Returns:
        List of indices of the num_samples_for_labelling most uncertain samples.
    """
    model.train()  # Set model to evaluation mode
    model.to(device)
    num_samples_for_labelling = min(num_samples_for_labelling, len(x_unlabelled))

    # Convert X to a DataLoader for efficient memory management
    dataset = AmfLoad.CustomNormalisedDataset(
        x_unlabelled, np.zeros((len(x_unlabelled), num_classes))
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_probs = []  # Store all MC Dropout softmax outputs

    with torch.no_grad():
        for i in tqdm(range(mc_samples)):
            batch_probs = []
            for batch_x, _ in dataloader:
                batch_x = batch_x.to(device)
                logits = model(batch_x)  # Forward pass
                probs = (
                    F.softmax(logits, dim=1).cpu().numpy()
                )  # Convert to probabilities & move to CPU
                batch_probs.append(probs)

            all_probs.append(
                np.vstack(batch_probs)
            )  # Store each MC iteration's probabilities

    all_probs = np.stack(all_probs, axis=0)  # Shape: (mc_samples, N, num_classes)

    # Compute the predictive entropy: H[y | x, D]
    mean_probs = np.mean(all_probs, axis=0)  # Average over MC samples
    entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-8), axis=1)

    # Compute the expected entropy: E_p(theta)[H[y | x, theta]]
    conditional_entropy = -np.mean(
        np.sum(all_probs * np.log(all_probs + 1e-8), axis=2), axis=0
    )

    # Compute BALD score: Mutual Information
    bald_scores = entropy - conditional_entropy

    # Get indices of the num_samples_for_labelling most uncertain samples
    top_indices = np.argsort(-bald_scores)[:num_samples_for_labelling]
    return list(top_indices)


def batch_bald_acquisition(
    model: torch.nn.Module,
    device: torch.device,
    x_unlabelled: list[NDArray[np.uint8]],
    num_samples_for_labelling: int,
    num_classes: int,
    mc_samples: int = 10,
    batch_size: int = 32,
) -> list[int]:
    """Return the indices of the most uncertain samples using BatchBALD.

    Implements BatchBALD acquisition function using MC Dropout for uncertainty
    estimation.

    Args:
        model: Neural network model, which must contain random dropout layers.
        device: The device (CPU/GPU) to run the model on.
        x_unlabelled: Unlabelled data points for selection.
        num_samples_for_labelling: Number of most uncertain samples to return.
        num_classes: Number of output classes.
        mc_samples: Number of stochastic forward passes.
        batch_size: Batch size for efficient memory management.

    Returns:
        List of indices of the num_samples_for_labelling most uncertain samples.
    """
    model.train()  # Set model to train mode to activate dropout
    model.to(device)
    num_samples_for_labelling = min(num_samples_for_labelling, len(x_unlabelled))

    # Convert X to a DataLoader for efficient memory management
    dataset = AmfLoad.CustomNormalisedDataset(
        x_unlabelled, np.zeros((len(x_unlabelled), num_classes))
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_probs = []  # Store all MC Dropout softmax outputs

    with torch.no_grad():
        for i in tqdm(range(mc_samples)):
            batch_probs = []
            for batch_x, _ in dataloader:
                batch_x = batch_x.to(device)
                logits = model(batch_x)  # Forward pass
                probs = (
                    F.softmax(logits, dim=1).cpu().numpy()
                )  # Convert to probabilities & move to CPU
                batch_probs.append(probs)

            all_probs.append(
                np.vstack(batch_probs)
            )  # Store each MC iteration's probabilities

    all_probs = np.stack(all_probs, axis=1)  # Shape: (N, mc_samples, num_classes)
    all_probs = torch.tensor(np.log(all_probs + 1e-8))
    return get_batchbald_batch(all_probs, num_samples_for_labelling, mc_samples).indices


def run(input_files: list[str]) -> int:
    """Run active learning acquisition to select optimal tiles for labelling.

    Args:
        input_files: List of input file paths containing unlabelled data.

    Returns: Exit code.
    """
    model = AmfModel.load()

    device = AmfConfig.get("device")
    num_classes = len(AmfConfig.get("class_names")[AmfConfig.get("colonisation_type")])
    method = AmfConfig.get("active_learning_method")
    num_samples_for_labelling = AmfConfig.get("num_samples_for_labelling")
    mc_samples = AmfConfig.get("mc_samples")
    batch_size = AmfConfig.get("batch_size")
    output_dir = AmfConfig.get("outdir")

    logger.info(
        f"Running active learning to procure optimal tiles for retraining using "
        f"{method}."
    )

    parts = AmfLoad.categorise_path(input_files)
    bald_images = parts["BALD"]

    dataset_loader_bald = AmfLoad.TileFilesandData(bald_images, is_bald_folder=True)
    x_unlabelled, file_names, rows, cols = dataset_loader_bald.get_all_unlabelled_data()

    # Initialize dictionary to store grouped data
    grouped_data: dict[str, tuple[list[NDArray[np.uint8]], list[int], list[int]]] = (
        defaultdict(lambda: ([], [], []))
    )

    # Iterate over the dataset and group by file_name
    for file_name, file_x, file_rows, file_cols in zip(
        file_names, x_unlabelled, rows, cols, strict=True
    ):
        # Append the values to their respective lists in the tuple
        grouped_data[file_name][0].append(file_x)  # x_unlabelled
        grouped_data[file_name][1].append(file_rows)  # rows
        grouped_data[file_name][2].append(file_cols)  # cols

    output_dict: dict[str, list[list[int]]] = defaultdict(lambda: [[], []])

    for file_name in grouped_data:
        logger.info(f"Acquiring samples for {file_name}")
        x, rows, cols = grouped_data[file_name]

        if method == "bald":
            indices_to_label = bald_acquisition(
                model,
                device,
                x,
                num_samples_for_labelling,
                num_classes,
                mc_samples,
                batch_size,
            )
        elif method == "batchbald":
            indices_to_label = batch_bald_acquisition(
                model,
                device,
                x,
                num_samples_for_labelling,
                num_classes,
                mc_samples,
                batch_size,
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        rows_chosen = [rows[i] for i in indices_to_label]
        cols_chosen = [cols[i] for i in indices_to_label]

        output_dict[file_name][0] = rows_chosen
        output_dict[file_name][1] = cols_chosen

    # Convert output_dict into a list of dictionaries for DataFrame
    output_data = []

    for file_name, (rows_chosen, cols_chosen) in output_dict.items():
        for r, c in zip(rows_chosen, cols_chosen, strict=True):
            output_data.append({"file_name": file_name, "row": r, "col": c})

    # Create a pandas DataFrame
    df = pd.DataFrame(output_data)

    # Define the output path and save as CSV
    output_file = f"{output_dir}/output_data.csv"
    df.to_csv(output_file, index=False)

    logger.info(f"CSV saved at {output_file}")

    return 500
