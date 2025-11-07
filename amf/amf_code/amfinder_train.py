# AMFinder - amfinder_train.py
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
Neural network training.

Trains a convolutional neural network with a set of ink-stained root
images associated with tile annotations which label colonised root
sections.
Annotations are stored in an auxiliary ZIP archive.

Functions
------------
:function import_settings: Imports image settings from a ZIP archive.
:function import_annotations: Imports tile annotations from a ZIP archive.
:function load_dataset: Loads training dataset.
:function class_weights: Computes class weights
:function get_callbacks: Configures Keras callbacks.
:function save_model_architecture: Saves neural network architecture.
:function run: Runs a training session.
"""

# New total imports log for train_flx
# Original AMFinder imports.
import random

# from torch.utils.data.distributed import DistributedSampler
# from torch.utils.tensorboard import SummaryWriter
import mlflow
import mlflow.pytorch
import numpy as np

# Semi-Supervised Imports (pruned with respect to earlier imports).
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import amfinder_config as AmfConfig
import amfinder_load as AmfLoad
import amfinder_log as AmfLog
import amfinder_model as AmfModel
import amfinder_save as AmfSave

# logger = tf.get_logger()
# logger.setLevel(logging.ERROR)
random.seed(42)


def class_weights(y, weight_type="inverse_freq", beta=0.9999, epsilon=1e-6):
    """
    Computes weights to counteract class imbalance and
    display statistics.

    :param one_hot_labels: Hot labels encoding tile annotations.
    :return: Dictionary of class weights.
    :rtype: dict

    Args:
    - y (torch.Tensor): One-hot encoded label tensor with shape (N, C), where C is the number of classes.
    - weight_type (str): Method for calculating class weights. Options are 'inverse_freq', 'effective_num',
    - beta (float): Smoothing parameter for effective number of samples (default 0.99).
    - epsilon (float): Small constant to avoid division by zero (default 1e-6).
    """

    print(f"[{AmfConfig.invite()}] Class weights")

    # Sum along axis 0 to count class occurrences (shape: (C,))
    class_counts = torch.sum(y, axis=0)
    total_samples = y.shape[0]
    final_class_weights = []
    if weight_type == "inverse_freq":
        # Inverse frequency weighting (handle zero counts)
        class_weights = total_samples / (class_counts.float() + epsilon)
        class_weights[class_counts == 0] = (
            0.0  # Assign 0 weight to classes with no samples
        )
        final_class_weights = class_weights

    elif weight_type == "effective_num":
        # Effective number of samples weighting (handle zero counts)
        effective_num = (1 - beta) / (1 - beta ** (class_counts.float() + epsilon))
        effective_num[class_counts == 0] = (
            0.0  # Assign 0 weight to classes with no samples
        )
        final_class_weights = effective_num

    else:
        raise ValueError(
            "Invalid weight_type. Choose from ['inverse_freq', 'effective_num']."
        )

    return [
        dict(enumerate(final_class_weights)),
        final_class_weights,
    ]  # class wegiths as a dict and array respectively.


# PyTorch implementation of earlier Keras functionality for early stopping.


class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta  # Tolerance margin
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_weights = None  # To store best weights so far.

    def check_early_stop(self, model, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_weights = model.state_dict()  # Save best weights

        elif score < self.best_score + self.delta:
            self.counter += 1

            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")

            if (
                self.counter >= self.patience
            ):  # Early Stopping reaches above patience level.
                self.early_stop = True

        else:
            self.best_score = score
            self.best_weights = model.state_dict()  # Save best weights
            self.counter = 0


class ReduceLROnPlateau:
    def __init__(self, optimiser, factor=0.2, patience=2, min_lr=1e-6, verbose=False):
        self.optimiser = optimiser
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None

    def step(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss

        elif val_loss > self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.counter = 0
                for param_group in self.optimiser.param_groups:
                    old_lr = param_group["lr"]
                    new_lr = max(old_lr * self.factor, self.min_lr)
                    param_group["lr"] = new_lr
                    if self.verbose:
                        print(f"Reducing lerning rate from {old_lr} to {new_lr}")

        else:
            self.best_loss = val_loss
            self.counter = 0


def run(input_files, flag, train_active_learning=False):
    """
    Creates or loads a convolutional neural network, and trains it
    with the annotated tiles extracted from input images.

    :param input_files: List of input images to train with.
    """
    if flag:
        mlflow.start_run()

    # Input model (either new or pre-trained).
    model = AmfModel.load()

    # Assign correct device, depending on cpu or gpu
    device = AmfConfig.get("device")
    model = model.to(device)

    # categorise path
    all_files = AmfLoad.categorise_path(input_files)
    full_list = all_files["train"]

    if train_active_learning:
        full_list = full_list + all_files["BALD"]

    # Validate input folder structure
    if not full_list:
        AmfLog.error("There is no train subfolder", AmfLog.ERR_NO_DATA)
        return 500

    full_dataset = AmfLoad.TileDatasetLoader(
        full_list,
        use_augmentation=AmfConfig.get("data_augm"),
        balance_factor=AmfConfig.get("balance_factor"),
        calculate_distribution=True,
    )

    # Extracting tiles and labels using list logic, to omit getitem() method.
    x = [x for x, y in full_dataset.dataset]
    y = [y for x, y in full_dataset.dataset]

    # x and y should be numpy arrays before passing them to the class
    val_size = AmfConfig.get("vfrac")
    splitter = AmfLoad.StratifiedDatasetSplitter(
        x, y, val_size=val_size, random_state=42
    )

    # Create datasets as NumPy arrays
    x_train, y_train = splitter._get_dataset("train")
    x_val, y_val = splitter._get_dataset("val")

    # Convert to CustomDataset
    train_dataset = AmfLoad.CustomNormalisedDataset(x_train, y_train)
    val_dataset = AmfLoad.CustomNormalisedDataset(x_val, y_val)

    # Initialise variables to prepare training and validation datasets
    labels = np.stack([full_dataset.labels[i] for i in range(len(full_dataset.labels))])
    labels = torch.from_numpy(labels)
    labels = labels.type(torch.float32)
    weights = class_weights(labels, "effective_num")
    class_weights_tensor = torch.tensor(weights[1], dtype=torch.float32).to(device)
    AmfLog.text(f"Weights: {weights}")
    loss_func = nn.CrossEntropyLoss(weight=class_weights_tensor)
    # Root segmentation (colonized vs non-colonized vs background).
    # ConvNet I has a standard, single input/single output architecture,
    # and can use ImageDataGenerator.
    # train_dataset = AmfLoad.CustomDataset(x_train, y_train)

    # Initialising relevant variables
    bs = AmfConfig.get("batch_size")
    num_workers = AmfConfig.get("num_workers")
    AmfLog.info(f"Running on {num_workers} separate workers")
    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    num_ep = (
        AmfConfig.get("epochs")
        if not train_active_learning
        else AmfConfig.get("epochs_active_learning")
    )
    optim = torch.optim.Adam(
        model.parameters(),
        lr=(
            AmfConfig.get("learning_rate")
            if not train_active_learning
            else AmfConfig.get("learning_rate_active_learning")
        ),
        betas=(AmfConfig.get("adam_beta1"), AmfConfig.get("adam_beta2")),
    )
    save_directory = AmfConfig.get("outdir")
    # Optional
    e = EarlyStopping(patience=AmfConfig.get("patience_e"), verbose=True)
    r = ReduceLROnPlateau(
        optimiser=optim,
        factor=0.5,
        patience=AmfConfig.get("patience_r"),
        min_lr=1e-9,
        verbose=True,
    )

    # Defining training loop
    # The training loop saves model history and the model itself.
    def train(
        model,
        train_loader,
        val_loader,
        criterion,
        optimiser,
        num_epochs,
        save_path,
        early_stopping=None,
        reduce_lr=None,
    ):
        history = {"loss": [], "val_loss": []}

        best_val_loss = float("inf")

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

            # Log losses
            AmfLog.text(
                f"Epoch {epoch + 1}/{num_epochs}, Average validation loss: {avg_val_loss:.4f}"
            )

            if flag:
                # Logging the metrics after each epoch
                mlflow.log_metric("Training loss", avg_loss, step=epoch)
                mlflow.log_metric("Validation loss", avg_val_loss, step=epoch)
                # mlflow.log_metric("Learning Rate", r.optimiser.param_groups["lr"], step=epoch)

            # Check for early stopping
            if (
                early_stopping != None
            ):  # TODO: This needs to be tied to the config file!
                AmfLog.text("Early stopping mechanism active")
                early_stopping.check_early_stop(model, avg_val_loss)
                AmfConfig.set("early_stopping", early_stopping)
                if early_stopping.early_stop:
                    AmfLog.text("Early stopping triggered.")
                    model.load_state_dict(
                        early_stopping.best_weights
                    )  # Save best weights
                    AmfConfig.set("early_break_epoch", epoch)
                    AmfSave.save_training_data(history, model, save_path)
                    break  # Exit the training loop if early stopping condition is met

            if avg_val_loss < best_val_loss:
                AmfLog.text(
                    "Average validation loss improved, updating saved model weights."
                )
                best_val_loss = avg_val_loss
                best_model_weights = (
                    model.state_dict()
                )  # Always store best performing model weights.

            # Step the learning rate scheduler
            if reduce_lr != None:
                AmfLog.text("Reduce LR mechanism active")
                reduce_lr.step(avg_val_loss)
                AmfConfig.set("reduce_lr_on_plateau", reduce_lr)

        model.load_state_dict(
            best_model_weights
        )  # Load the weights that achieved lowest loss.

        if flag:
            mlflow.pytorch.log_model(model, "model")
            mlflow.end_run()

        # Saving training data
        AmfSave.save_training_data(history, model, save_path)

        return 200

    # Call the training function
    output = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=loss_func,
        optimiser=optim,
        num_epochs=num_ep,
        save_path=save_directory,
        early_stopping=None,
        reduce_lr=r,
    )

    # Save model information (layers and graph) upon user request.
    if AmfConfig.get("summary"):
        AmfSave.save_model_architecture(model, device, AmfConfig.get("tile_edge"))

    return output
