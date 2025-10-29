import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import amfinder_load as AmfLoad
import amfinder_config as AmfConfig
import amfinder_model as AmfModel
from torch.utils.data import DataLoader
from collections import defaultdict

from copy import deepcopy
from acquisition_functions import get_batchbald_batch
import pandas as pd
import amfinder_log as AmfLog


def get_example_model_to_check_dropout_addition():
    pt_flag = AmfConfig.get("pre_trained")
    num_classes = len(AmfConfig.get("class_names"))
    if AmfConfig.get("model_type") == "cnn1":
        if pt_flag:
            AmfLog.error(
                "Pre-trained weights are unanavailable for CNN1. Please proceed with pre_trained=False",
                exit_code=AmfLog.ERR_NO_PRETRAINED_MODEL,
            )
        else:
            model = AmfModel.create_cnn1()

    elif AmfConfig.get("model_type") == "resnet":
        model = AmfModel.create_resnet50(num_classes=num_classes, pre_trained=pt_flag)

    elif AmfConfig.get("model_type") == "resnext":
        model = AmfModel.create_resnext50(num_classes=num_classes, pre_trained=pt_flag)

    elif AmfConfig.get("model_type") == "efficientnet":
        model = AmfModel.create_efficientnetb5(
            num_classes=num_classes, pre_trained=pt_flag
        )

    elif AmfConfig.get("model_type") == "efficientnetv2":
        model = AmfModel.create_efficientnet_v2_m(
            num_classes=num_classes, pre_trained=pt_flag
        )
    else:
        AmfLog.error(
            "Invalid model type. Please choose one of the following model types: 'cnn1', 'resnet', 'resnext', 'efficientnet' or 'efficientnetv2'.",
            exit_code=AmfLog.ERR_INVALID_MODEL,
        )

    model_name = AmfConfig.get("model_type")
    AmfLog.text(
        f"The model you are utilising is {model_name}. The Pre-Trained Flag is: {pt_flag}"
    )
    return model


def is_dropout_applied(model):
    """Check if two PyTorch models have identical architectures."""

    base_model = get_example_model_to_check_dropout_addition()
    base_model_with_dropout = add_dropout_layers(base_model, 0.25)
    # Compare string representations
    if str(model) != str(base_model_with_dropout):
        return False

    # Compare parameter structure
    params1 = {n: p.shape for n, p in model.named_parameters()}
    params2 = {n: p.shape for n, p in base_model_with_dropout.named_parameters()}

    return params1 == params2


def add_dropout_layers(model: nn.Module, p: float) -> nn.Module:
    model = deepcopy(model)

    # Find last layer
    last_layer = ""
    for name, m in model.named_modules():
        if isinstance(m, (nn.Linear)):
            last_layer = name

    # Add dropout after appropriate layers
    for name, m in model.named_modules():
        if (
            isinstance(m, (nn.Linear))
            and name != last_layer
            and not isinstance(m, nn.Dropout)
        ):
            parent = (
                model
                if "." not in name
                else dict([*model.named_modules()])[name.rsplit(".", 1)[0]]
            )
            setattr(parent, name.split(".")[-1], nn.Sequential(m, nn.Dropout(p)))

    return model


# The 3 functions above are not currently used as CNN1 has sufficient amount of dropout. Will need this or something similiar for the advanced model architectures like efficientnet, resnet, resnext etc


def bald_acquisition(
    model,
    device,
    x_unlabelled,
    num_samples_for_labelling,
    num_classes,
    mc_samples=50,
    batch_size=32,
):
    """
    Implements BALD acquisition function using MC Dropout for uncertainty estimation.

    Parameters:
        model (torch.nn.Module): The Neural Network (BNN) with dropout.
        device (torch.device): The device (CPU/GPU) to run the model on.
        X (list or np.array): The unlabeled data points for selection.
        num_samples_for_labelling (int): Number of most uncertain samples to return.
        mc_samples (int): Number of stochastic forward passes.
        batch_size (int): Batch size for efficient memory management.

    Returns:
        np.array: Indices of the num_samples_for_labelling most uncertain samples.
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

    AmfLog.progress_bar(0, mc_samples, indent=1)

    with torch.no_grad():
        for i in range(mc_samples):
            batch_probs = []
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(device)
                logits = model(batch_x)  # Forward pass
                probs = (
                    F.softmax(logits, dim=1).cpu().numpy()
                )  # Convert to probabilities & move to CPU
                batch_probs.append(probs)

            all_probs.append(
                np.vstack(batch_probs)
            )  # Store each MC iteration's probabilities

            AmfLog.progress_bar(i, mc_samples, indent=1)

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
    return top_indices.tolist()


def batch_bald_acquisition(
    model,
    device,
    x_unlabelled,
    num_samples_for_labelling,
    num_classes,
    mc_samples=10,
    batch_size=32,
):
    model.train()  # Set model to evaluation mode
    model.to(device)
    num_samples_for_labelling = min(num_samples_for_labelling, len(x_unlabelled))

    # Convert X to a DataLoader for efficient memory management
    dataset = AmfLoad.CustomNormalisedDataset(
        x_unlabelled, np.zeros((len(x_unlabelled), num_classes))
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_probs = []  # Store all MC Dropout softmax outputs

    AmfLog.progress_bar(0, mc_samples, indent=1)

    with torch.no_grad():
        for i in range(mc_samples):
            batch_probs = []
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(device)
                logits = model(batch_x)  # Forward pass
                probs = (
                    F.softmax(logits, dim=1).cpu().numpy()
                )  # Convert to probabilities & move to CPU
                batch_probs.append(probs)

            all_probs.append(
                np.vstack(batch_probs)
            )  # Store each MC iteration's probabilities

            AmfLog.progress_bar(i, mc_samples, indent=1)

    all_probs = np.stack(all_probs, axis=1)  # Shape: (N, mc_samples, num_classes)
    all_probs = torch.tensor(np.log(all_probs + 1e-8))
    bald_batch_indices = get_batchbald_batch(
        all_probs, num_samples_for_labelling, mc_samples
    ).indices
    return bald_batch_indices


def run(input_files):

    model = AmfModel.load()

    device = AmfConfig.get("device")
    num_classes = len(AmfConfig.get("class_names")[AmfConfig.get("colonisation_type")])
    method = AmfConfig.get("active_learning_method")
    num_samples_for_labelling = AmfConfig.get("num_samples_for_labelling")
    mc_samples = AmfConfig.get("mc_samples")
    batch_size = AmfConfig.get("batch_size")
    output_dir = AmfConfig.get("outdir")

    AmfLog.info(
        f"Running active learning to procure optimal tiles for retraining using {method}."
    )

    parts = AmfLoad.categorise_path(input_files)
    bald_images = parts["BALD"]

    dataset_loader_bald = AmfLoad.TileFilesandData(bald_images, is_bald_folder=True)
    x_unlabelled, file_names, rows, cols = dataset_loader_bald.get_all_unlabelled_data()

    # Initialize dictionary to store grouped data
    grouped_data = defaultdict(lambda: ([], [], []))

    # Iterate over the dataset and group by file_name
    for file_name, x, r, c in zip(file_names, x_unlabelled, rows, cols):
        # Append the values to their respective lists in the tuple
        grouped_data[file_name][0].append(x)  # x_unlabelled
        grouped_data[file_name][1].append(r)  # rows
        grouped_data[file_name][2].append(c)  # cols

    output_dict = defaultdict(lambda: [[], []])

    for file_name in grouped_data:
        AmfLog.info(f"Acquiring samples for {file_name}")
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
        for r, c in zip(rows_chosen, cols_chosen):
            output_data.append({"file_name": file_name, "row": r, "col": c})

    # Create a pandas DataFrame
    df = pd.DataFrame(output_data)

    # Define the output path and save as CSV
    output_file = f"{output_dir}/output_data.csv"
    df.to_csv(output_file, index=False)

    print(f"CSV saved at {output_file}")

    return 500
