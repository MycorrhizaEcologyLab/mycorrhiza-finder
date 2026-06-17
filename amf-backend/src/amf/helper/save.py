# AMFinder - save.py
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
Model and prediction saving.

Functions
------------

:function now: Returns the current date/time.
:function training_data: Saves training weights, history and plots.
:function save_settings: Saves image settings.
:function prediction_table: Saves or append predictions to an archive.
"""

import csv
import datetime
import json
import os
import pickle
from contextlib import redirect_stdout
from zipfile import ZipFile

import h5py
import numpy as np
import pandas as pd
import torch
from loguru import logger
from torchinfo import summary
from torchview import draw_graph

import amf.helper.config as AmfConfig
import amf.helper.plot as AmfPlot
from amf.helper.api_objects import PredictionValues
from amf.helper.api_utils import save_predictions_to_db
from amf.helper.db_config import connect
from amf.helper.metrics_collector import MetricsCollector


def now() -> str:
    """
    Returns the current date/time in a format suitable
    for use as file name.
    """

    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def save_history(history: dict[str, list[float]], output_dir: str) -> None:
    """
    Saves training history to a file.
    """
    filepath = os.path.join(output_dir, "train_history.csv")

    rows = list(
        zip(range(1, len(history["loss"]) + 1), history["loss"], history["val_loss"])
    )

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "val_loss"])
        writer.writerows(rows)

        # Summary row
        best_epoch = min(
            range(len(history["val_loss"])), key=lambda i: history["val_loss"][i]
        )
        writer.writerow([])
        writer.writerow(["best_epoch", best_epoch + 1, history["val_loss"][best_epoch]])

    return history["val_loss"][best_epoch]  # return best val loss for reuse


def save_training_data(
    history: dict[str, list[float]], model: torch.nn.Module, save_path: str
) -> None:
    """
    Saves training history, model weights, and other relevant information.

    :param history: Training history dictionary.
    :param model: The trained model.
    :param save_path: The directory to save the training data.
    """
    ## TODO: An additional model load class (loading the state dict seperately to
    # the architecture) would be required in the case of HDF5

    best_val_loss = save_history(
        history, save_path
    )  # Save history and get best val loss
    # Save best val loss to a text file for easy access
    with open(os.path.join(save_path, "val_loss.txt"), "w") as f:
        f.write(f"{best_val_loss:.6f}\n")

    # Save the model state_dict as pth
    model_path = os.path.join(save_path, "model.pth")
    torch.save(model, model_path)

    # Optionally, save additional plots or metrics here
    # In the original AMFinder version, the accuracy was tracked throughout the
    # training process.
    # As this metric is not used any longer, it is not included in the PyTorch
    # version of the code.´

    early = AmfConfig.get("early_stopping")
    if early is not None and early.early_stop:
        epochs = AmfConfig.get("early_break_epoch")
    else:
        train_active_learning = AmfConfig.get("train_active_learning")
        epochs = (
            AmfConfig.get("epochs")
            if not train_active_learning
            else AmfConfig.get("epochs_active_learning")
        )

    x_range = np.arange(0, epochs)

    AmfPlot.initialize()
    AmfPlot.draw(
        history,
        epochs,
        "Loss",
        x_range,
        "loss",
        "val_loss",
        fname="loss.png",
        dir=save_path,
    )

    logger.info(f"Saved model output to {save_path}")


# Function to get a summary of the model architecture as txt file and graph.
def save_model_architecture(
    model: torch.nn.Module, device: torch.device, tile_size: int
) -> None:
    """
    Saves neural network architecture, parameters count, etc.

    :param model: Model to save.
    """

    path_summary = os.path.join(AmfConfig.get("outdir"), "CNN1_summary.txt")
    path_graph = os.path.join(AmfConfig.get("outdir"), "CNN1_graph")
    input_tensor = torch.randn(1, 3, tile_size, tile_size).to(device)

    with open(path_summary, "w") as sf:
        with redirect_stdout(sf):
            # Model summary in a txt file
            summary(model, input_data=input_tensor)

    draw_graph(model, input_data=input_tensor, filename=path_graph, save_graph=True)
    # TODO Currently this creates three files: "CNN1_summary.txt", "CNN1_graph.png" and
    # "CNN1_graph". The third file is not required in theory.


def save_settings(path: str) -> None:
    """
    Saves image settings (currently, only tile size).

    :param z: ZIP archive.
    """

    directory = os.path.dirname(path)
    image_name = os.path.splitext(os.path.basename(path))[0]
    json_file_path = os.path.join(directory, f"{image_name}_settings.json")

    settings = {"tile_edge": AmfConfig.get("tile_edge")}

    # Write settings to the JSON file
    with open(json_file_path, "w") as file:
        json.dump(settings, file)


def save_metrics(metrics_collector: MetricsCollector, path: str) -> None:
    """
    Saves predictions, class metrics, file metrics, and generic metrics to an archive.
    :param path: path to the ZIP archive.
    :param metrics_collector: Instance of MetricsCollector containing metrics to save.
    """
    uniq = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    zipf = os.path.join(path, f"results_{uniq}.zip")
    logger.debug(f"    - saving as {zipf}... ", end="")

    with ZipFile(zipf, "w") as z:
        # Handle class metrics
        if metrics_collector.per_class_metrics:
            class_metrics_df = metrics_collector.convert_to_dataframe(
                metrics_type="class"
            )
            class_metrics_csv = class_metrics_df.to_csv(index=False)
            class_metrics_file_name = "class_metrics.csv"
            z.writestr(class_metrics_file_name, class_metrics_csv)

        # Handle file-specific metrics
        if "file_metrics" in metrics_collector.metrics_dataframes:
            file_metrics_df = metrics_collector.convert_to_dataframe(
                metrics_type="file_metrics"
            )
            file_metrics_csv = file_metrics_df.to_csv(index=False)
            file_metrics_file_name = "file_metrics.csv"
            z.writestr(file_metrics_file_name, file_metrics_csv)

        # Handle generic metrics
        if metrics_collector.generic_metrics:
            generic_metrics_df = metrics_collector.convert_to_dataframe(
                metrics_type="generic"
            )
            generic_metrics_csv = generic_metrics_df.to_csv(index=False)
            generic_metrics_file_name = "generic_metrics.csv"
            generic_metrics_df.to_csv(
                os.path.join(path, "generic_metrics.csv"), index=False
            )
            z.writestr(generic_metrics_file_name, generic_metrics_csv)

        for image_name, image in metrics_collector.images.items():
            with open(image, "rb") as img_file:
                z.writestr(image_name, img_file.read())


def prediction_table(results: pd.DataFrame, path: str) -> None:
    """
    Saves or append predictions to an archive.

    :param results: annotation table to save.
    :param path: path to the ZIP archive.
    """

    if results is not None:
        image_name = os.path.splitext(os.path.basename(path))[0]

        if AmfConfig.get("use_db"):
            conn = connect("amf")
            with conn, conn.cursor() as crsr:
                cnn1results = results.values.tolist()
                values = PredictionValues(
                    fileName=image_name,
                    colonisationType=AmfConfig.get("colonisation_type"),
                    tileEdge=AmfConfig.get("tile_edge"),
                    cnnOneValues=cnn1results,
                )
                save_predictions_to_db(crsr, values)

            logger.info("Saved results to DB")

        else:
            directory = os.path.dirname(path)
            uniq = now()
            results.to_csv(
                os.path.join(directory, f"{image_name}_{uniq}_cnn_1_predictions.csv"),
                sep=",",
                encoding="utf-8",
                index=False,
                mode="w",
            )

            save_settings(path)

            logger.info(f"Saved results to CSV at {path}")
