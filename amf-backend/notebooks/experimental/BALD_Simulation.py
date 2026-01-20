# mypy: ignore-errors

import argparse
import copy
import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader

# from torchsummary import summary
from torchinfo import summary
from tqdm import tqdm

import amf.helper.config as AmfConfig
import amf.helper.load as AmfLoad
import amf.helper.model as AmfModel
from amf.helper.acquisition_functions import get_batchbald_batch
from amf.mode.train import class_weights
from amf.helper.log import logger

def add_dropout_layers(model: nn.Module, p: float = 0.25) -> nn.Module:
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


def bald_acquisition(
    model: torch.nn.Module,
    device: torch.device,
    x_unlabelled: list[NDArray[np.uint8]],
    num_samples_for_labelling: int,
    num_classes: int,
    mc_samples: int = 10,
    batch_size: int = 32,
) -> list[int]:
    """
    Implements BALD acquisition function using MC Dropout for uncertainty estimation.

    Parameters:
        model (torch.nn.Module): The Bayesian Neural Network (BNN) with dropout.
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

    # Convert X to a DataLoader for efficient memory management
    dataset = AmfLoad.CustomNormalisedDataset(
        x_unlabelled, np.zeros((len(x_unlabelled), num_classes))
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_probs = []  # Store all MC Dropout softmax outputs

    with torch.no_grad():
        for _ in range(mc_samples):
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
    model.train()  # Set model to evaluation mode
    model.to(device)

    # Convert X to a DataLoader for efficient memory management
    dataset = AmfLoad.CustomNormalisedDataset(
        x_unlabelled, np.zeros((len(x_unlabelled), num_classes))
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_probs = []  # Store all MC Dropout softmax outputs

    with torch.no_grad():
        for _ in range(mc_samples):
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

    all_probs = np.stack(all_probs, axis=1)  # Shape: (N, mc_samples, num_classes)
    all_probs = torch.tensor(np.log(all_probs + 1e-8))
    bald_batch_indices = get_batchbald_batch(
        all_probs, num_samples_for_labelling, mc_samples
    ).indices
    return bald_batch_indices


def balance_dataset(
    x: list[NDArray[np.uint8]], y: list[NDArray[np.uint8]], factor: float = 1.0
) -> tuple[
    list[NDArray[np.uint8]], list[NDArray[np.uint8]]
]:  # factor is a hyperparameter
    # Count the number of samples in class 0 (assumes one-hot encoding)
    class_0_samples = sum(array[0] for array in y)

    # Initialize a list to keep track of indices to retain
    indices_to_keep: list[int] = []

    # Convert the one-hot encoded labels to class indices
    y_indices = np.argmax(y, axis=1)

    # Retrieve the number of classes based on configuration
    num_classes = len(AmfConfig.get("class_names")[AmfConfig.get("colonisation_type")])

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

            # Clip samples to the minimum of `factor * class_0_samples`
            # or the class size
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
                # Retain all samples if the number to select exceeds
                # or equals the class size
                indices_to_keep.extend(idx_class_samples)

    # Update the input data (x) and labels (y) to retain only the selected indices
    indices_to_keep = sorted(indices_to_keep)  # Ensure indices are in order
    x = [x[i] for i in indices_to_keep]
    y = [y[i] for i in indices_to_keep]

    return x, y


def train_model(
    model: torch.nn.Module,
    device: torch.device,
    x_train: list[NDArray[np.uint8]],
    y_train: list[NDArray[np.uint8]],
    x_val: list[NDArray[np.uint8]],
    y_val: list[NDArray[np.uint8]],
    num_epochs: int = 5,
    batch_size: int = 32,
    optimiser: torch.optim.Optimizer | None = None,
) -> torch.nn.Module:
    model.train()
    # Set a default optimizer if none is provided
    if optimiser is None:
        optimiser = optim.Adam(model.parameters(), lr=0.0001)

    # Define the loss function (categorical cross entropy)
    y_train_np = np.array(y_train)
    y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32)
    weights = class_weights(y_train_tensor, weight_type="effective_num")
    class_weights_tensor = weights[1].to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # Create a Dataset and DataLoader
    train_dataset = AmfLoad.CustomNormalisedDataset(x_train, y_train)
    val_dataset = AmfLoad.CustomNormalisedDataset(x_val, y_val)
    num_workers = AmfConfig.get("num_workers")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Training loop
    history: dict[str, list[float]] = {"loss": [], "val_loss": []}

    best_val_loss = float("inf")

    best_model_weights = model.state_dict()

    for epoch in range(num_epochs):
        model.train()  # Setting training mode
        running_loss = 0.0  # Initialising running loss

        for batch_x, batch_y in tqdm(
            train_loader,
            desc=f"Training Epoch {epoch + 1}/{num_epochs}",
            leave=False,
        ):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimiser.zero_grad()
            output = model(batch_x)
            batch_y = torch.argmax(batch_y, dim=1).to(
                device
            )  # Conversion to class indices suitable for Pytorch
            loss = criterion(output, batch_y)  #
            loss.backward()
            optimiser.step()

            running_loss += loss.item()  # Tracking running loss

        # Average loss per epoch
        avg_loss = running_loss / len(train_loader)
        history["loss"].append(avg_loss)

        # Validation run
        model.eval()  # Setting evaluation mode
        val_running_loss = 0.0

        with torch.no_grad():  # No gradients during val
            for batch_x, batch_y in tqdm(
                val_loader,
                desc=f"Validating Epoch {epoch + 1}/{num_epochs}",
                leave=False,
            ):
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_x)
                batch_y = torch.argmax(batch_y, dim=1).to(
                    device
                )  # Conversion to class indices suitable for Pytorch
                val_loss = criterion(outputs, batch_y)
                val_running_loss += val_loss.item()

        avg_val_loss = val_running_loss / len(val_loader)
        history["val_loss"].append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            # print("Average validation loss improved, updating saved model weights.")
            logger.info("Average validation loss improved, updating saved model weights.")
            best_val_loss = avg_val_loss
            best_model_weights = (
                model.state_dict()
            )  # Always store best performing model weights.

    model.load_state_dict(
        best_model_weights
    )  # Load the weights that achieved lowest loss.

    return model


def evaluate_model(
    x_test: list[NDArray[np.uint8]],
    y_test: list[NDArray[np.uint8]],
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 32,
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    model.eval()
    num_classes = y_test[0].shape[0]
    # Ensure the model is in evaluation mode
    model.eval()

    # Create a DataLoader for batching
    test_dataset = AmfLoad.CustomNormalisedDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    all_y_true = []
    all_y_pred = []

    with torch.no_grad():  # Turn off gradients for evaluation
        for x_batch, y_batch in test_loader:
            # Move the batch to the specified device
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Forward pass: Compute predicted outputs by passing input data to the model
            outputs = model(x_batch)

            # Convert y_batch to the expected shape (batch_size,)
            _, y_true_batch = torch.max(y_batch, 1)

            # Accumulate the cross-entropy loss
            loss = criterion(outputs, y_true_batch).item()
            total_loss += loss * x_batch.size(0)

            # Get the predicted class from the model's output
            _, y_pred_batch = torch.max(outputs, 1)

            # Collect predictions
            all_y_true.append(y_true_batch.cpu())
            all_y_pred.append(y_pred_batch.cpu())

    # Concatenate results across all batches
    all_y_true = torch.cat(all_y_true)
    all_y_pred = torch.cat(all_y_pred)

    # Convert tensors to numpy arrays for scikit-learn evaluation
    y_true_np = all_y_true.numpy()
    y_pred_np = all_y_pred.numpy()

    # Calculate F1 score for each class
    class_labels = range(num_classes)
    f1_scores = f1_score(
        y_true_np, y_pred_np, average=None, labels=class_labels, zero_division=0
    )

    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_true_np, y_pred_np, labels=range(num_classes))

    # Compute average loss
    average_loss = total_loss / len(test_dataset)

    return average_loss, f1_scores, conf_matrix


def simulate_active_learning(
    model: torch.nn.Module,
    device: torch.device,
    n_iterations: int,
    num_samples_to_be_labelled_per_iteration: int,
    x_train: list[NDArray[np.uint8]],
    y_train: list[NDArray[np.uint8]],
    x_val: list[NDArray[np.uint8]],
    y_val: list[NDArray[np.uint8]],
    x_unlabelled: list[NDArray[np.uint8]],
    y_unlabelled: list[NDArray[np.uint8]],
    x_test: list[NDArray[np.uint8]],
    y_test: list[NDArray[np.uint8]],
    method: str = "bald",  # TODO enum?
) -> tuple[
    list[float],
    list[NDArray[np.float64]],
    list[list[int]],
    list[NDArray[np.float64]],
    list[dict[str, int]],
]:
    # print(f"Method is {method}.")
    logger.info(f"Method is {method}.")
    loss_array = []
    f1_scores_array = []
    bald_indices_array = []  # To store the indices of selected samples
    conf_matrix_array = []
    num_classes = len(AmfConfig.get("class_names")[AmfConfig.get("colonisation_type")])
    class_dict_in_selected_labels_array = []

    for i in range(n_iterations):
        indices_to_label = []

        if method == "bald":
            indices_to_label = bald_acquisition(
                model,
                device,
                x_unlabelled,
                num_samples_to_be_labelled_per_iteration,
                num_classes,
            )

        elif method == "batch_bald":
            indices_to_label = batch_bald_acquisition(
                model,
                device,
                x_unlabelled,
                num_samples_to_be_labelled_per_iteration,
                num_classes,
            )

        else:
            num_unlabelled = len(x_unlabelled)
            indices_to_label = random.sample(
                range(num_unlabelled), num_samples_to_be_labelled_per_iteration
            )

        # Print the selected indices for the current iteration
        # print(f"Iteration {i + 1} - Selected indices for labeling: {indices_to_label}")
        logger.debug(f"Iteration {i + 1} - Selected indices for labeling: {indices_to_label}")
        bald_indices_array.append(indices_to_label)

        class_names = ["AM+", "M−", "Background", "Unreadable", "DSE", "Hybrid"]
        class_dict_in_selected_labels: dict[str, int] = {}
        for idx in indices_to_label:
            class_dict_in_selected_labels[class_names[np.argmax(y_unlabelled[idx])]] = (
                class_dict_in_selected_labels.get(
                    class_names[np.argmax(y_unlabelled[idx])], 0
                )
                + 1
            )
        class_dict_in_selected_labels_array.append(class_dict_in_selected_labels)

        # print(f"Evaluation on test for {method} acquisition method. Iteration {i}")
        loss, f1_scores, conf_matrix = evaluate_model(x_test, y_test, model, device)
        loss_array.append(loss)
        f1_scores_array.append(f1_scores)
        conf_matrix_array.append(conf_matrix)

        x_train.extend([x_unlabelled[i] for i in indices_to_label])
        y_train.extend([y_unlabelled[i] for i in indices_to_label])
        x_unlabelled = [
            item for idx, item in enumerate(x_unlabelled) if idx not in indices_to_label
        ]
        y_unlabelled = [
            item for idx, item in enumerate(y_unlabelled) if idx not in indices_to_label
        ]

        model = train_model(
            model,
            device,
            x_train,
            y_train,
            x_val,
            y_val,
            num_epochs=1,
            batch_size=32,
            optimiser=None,
        )

    return (
        loss_array,
        f1_scores_array,
        bald_indices_array,
        conf_matrix_array,
        class_dict_in_selected_labels_array,
    )


def main(data_directory_name: str) -> None:
    model = AmfModel.load()
    model = add_dropout_layers(model, p=0.25)
    summary(model)
    device = AmfConfig.get("device")

    input_files = AmfConfig.find_files_in_directory(data_directory_name)
    parts = AmfLoad.categorise_path(input_files)
    train_images = parts["train"]
    unlabelled_images = parts["BALD"]
    test_images = parts["test"]

    dataset_loader_train = AmfLoad.TileFilesandData(train_images, is_bald_folder=True)
    x_train, y_train, fn, r, c = dataset_loader_train.get_all_data()
    x_train, y_train = balance_dataset(x_train, y_train)
    splitter = AmfLoad.StratifiedDatasetSplitter(
        x_train, y_train, val_size=0.2, random_state=42
    )
    x_train, y_train = splitter._get_dataset("val")
    # x_val, y_val = splitter._get_dataset("val")

    splitter = AmfLoad.StratifiedDatasetSplitter(
        x_train, y_train, val_size=0.2, random_state=42
    )
    x_train = []
    y_train = []
    x_val, y_val = splitter._get_dataset("val")

    dataset_loader_unlabelled = AmfLoad.TileFilesandData(
        unlabelled_images, is_bald_folder=True
    )
    x_unlabelled, y_unlabelled, _, _, _ = dataset_loader_unlabelled.get_all_data()

    dataset_loader_test = AmfLoad.TileFilesandData(test_images, is_bald_folder=True)
    x_test, y_test, _, _, _ = dataset_loader_test.get_all_data()
    x_test, y_test = balance_dataset(x_test, y_test)

    model_bald = copy.deepcopy(model)
    model_bald = model_bald.to(device)

    model_rand = copy.deepcopy(model)
    model_rand = model_rand.to(device)

    model_batch_bald = copy.deepcopy(model)
    model_batch_bald = model_batch_bald.to(device)

    # print(f"Length of x_train: {len(x_train)}")
    # print(f"Length of y_train: {len(y_train)}")
    # print(f"Length of x_val: {len(x_val)}")
    # print(f"Length of y_val: {len(y_val)}")
    # print(f"Length of x_unlabelled: {len(x_unlabelled)}")
    # print(f"Length of y_unlabelled: {len(y_unlabelled)}")
    # print(f"Length of x_test: {len(x_test)}")
    # print(f"Length of y_test: {len(y_test)}")
    logger.info(f"Length of x_train: {len(x_train)}")
    logger.info(f"Length of y_train: {len(y_train)}")
    logger.info(f"Length of x_val: {len(x_val)}")
    logger.info(f"Length of y_val: {len(y_val)}")
    logger.info(f"Length of x_unlabelled: {len(x_unlabelled)}")
    logger.info(f"Length of y_unlabelled: {len(y_unlabelled)}")
    logger.info(f"Length of x_test: {len(x_test)}")
    logger.info(f"Length of y_test: {len(y_test)}")

    num_iterations = 2
    num_samples_to_select_per_iteration = 2
    # Simulate active learning with the BALD acquisition method
    (
        loss_array_bald,
        f1_scores_array_bald,
        bald_indices_array_bald,
        conf_matrix_array_bald,
        class_dict_in_selected_labels_array_bald,
    ) = simulate_active_learning(
        model_bald,
        device,
        num_iterations,
        num_samples_to_select_per_iteration,
        x_train,
        y_train,
        x_val,
        y_val,
        x_unlabelled,
        y_unlabelled,
        x_test,
        y_test,
        method="bald",
    )

    # Simulate active learning with the random acquisition method
    (
        loss_array_random,
        f1_scores_array_random,
        bald_indices_array_random,
        conf_matrix_array_random,
        class_dict_in_selected_labels_array_random,
    ) = simulate_active_learning(
        model_rand,
        device,
        num_iterations,
        num_samples_to_select_per_iteration,
        x_train,
        y_train,
        x_val,
        y_val,
        x_unlabelled,
        y_unlabelled,
        x_test,
        y_test,
        method="random",
    )

    # Simulate active learning with the batchBALD acquisition method
    (
        loss_array_batch_bald,
        f1_scores_array_batch_bald,
        bald_indices_array_batch_bald,
        conf_matrix_array_batch_bald,
        class_dict_in_selected_labels_array_batch_bald,
    ) = simulate_active_learning(
        model_batch_bald,
        device,
        num_iterations,
        num_samples_to_select_per_iteration,
        x_train,
        y_train,
        x_val,
        y_val,
        x_unlabelled,
        y_unlabelled,
        x_test,
        y_test,
        method="batch_bald",
    )

    num_iterations = len(loss_array_bald)
    # print("Comparison of BALD and Random Methods:")
    logger.debug("Comparison of BALD and Random Methods:")
    for i in range(num_iterations):
        # print(f"\nIteration {i + 1}:")
        logger.debug(f"\nIteration {i + 1}:")

        # Output for BALD method
        loss_bald = loss_array_bald[i]
        f1_scores_bald = f1_scores_array_bald[i]
        f1_bald_formatted = [f"{score:.2f}" for score in f1_scores_bald]
        # print(
        #     f"  BALD - Loss: {loss_bald:.2f}, "
        #     f"F1 Scores (per class): {f1_bald_formatted}, "
        #     f"Macro F1: {f1_scores_bald.mean():.2f}, "
        #     f"Confusion Matrix: {conf_matrix_array_bald[i]}, "
        #     f"Class Dictionary: {class_dict_in_selected_labels_array_bald[i]}"
        # )
        logger.debug(
            f"  BALD - Loss: {loss_bald:.2f}, "
            f"F1 Scores (per class): {f1_bald_formatted}, "
            f"Macro F1: {f1_scores_bald.mean():.2f}, "
            f"Confusion Matrix: {conf_matrix_array_bald[i]}, "
            f"Class Dictionary: {class_dict_in_selected_labels_array_bald[i]}"
        )

        # Output for Random method
        loss_random = loss_array_random[i]
        f1_scores_random = f1_scores_array_random[i]
        f1_random_formatted = [f"{score:.2f}" for score in f1_scores_random]
        # print(
        #     f"  Random - Loss: {loss_random:.2f}, "
        #     f"F1 Scores (per class): {f1_random_formatted}, "
        #     f"Macro F1: {f1_scores_random.mean():.2f}, "
        #     f"Confusion Matrix: {conf_matrix_array_random[i]}, "
        #     f"Class Dictionary: {class_dict_in_selected_labels_array_random[i]}"
        # )
        logger.debug(
            f"  Random - Loss: {loss_random:.2f}, "
            f"F1 Scores (per class): {f1_random_formatted}, "
            f"Macro F1: {f1_scores_random.mean():.2f}, "
            f"Confusion Matrix: {conf_matrix_array_random[i]}, "
            f"Class Dictionary: {class_dict_in_selected_labels_array_random[i]}"
        )

        # Output for BatchBALD method
        loss_batch_bald = loss_array_batch_bald[i]
        f1_scores_batch_bald = f1_scores_array_batch_bald[i]
        f1_batch_bald_formatted = [f"{score:.2f}" for score in f1_scores_batch_bald]
        # print(
        #     f"  BatchBALD - Loss: {loss_batch_bald:.2f}, "
        #     f"F1 Scores (per class): {f1_batch_bald_formatted}, "
        #     f"Macro F1: {f1_scores_batch_bald.mean():.2f}, "
        #     f"Confusion Matrix: {conf_matrix_array_batch_bald[i]}, "
        #     f"Class Dictionary: {class_dict_in_selected_labels_array_batch_bald[i]}"
        # )
        logger.debug(
            f"  BatchBALD - Loss: {loss_batch_bald:.2f}, "
            f"F1 Scores (per class): {f1_batch_bald_formatted}, "
            f"Macro F1: {f1_scores_batch_bald.mean():.2f}, "
            f"Confusion Matrix: {conf_matrix_array_batch_bald[i]}, "
            f"Class Dictionary: {class_dict_in_selected_labels_array_batch_bald[i]}"
        )

    # print("Macro f1 for bald: ", [i.mean() for i in f1_scores_array_bald])
    # print("Macro f1 for random: ", [i.mean() for i in f1_scores_array_random])
    # print("Macro f1 for batch bald: ", [i.mean() for i in f1_scores_array_batch_bald])
    logger.info(f"Macro f1 for bald: {[i.mean() for i in f1_scores_array_bald]}")
    logger.info(f"Macro f1 for random: {[i.mean() for i in f1_scores_array_random]}")
    logger.info(f"Macro f1 for batch bald: {[i.mean() for i in f1_scores_array_batch_bald]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Active Learning Simulation")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the directory containing the dataset.",
    )
    args = parser.parse_args()
    main(args.data_dir)
