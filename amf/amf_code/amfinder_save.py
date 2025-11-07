# AMFinder - amfinder_save.py
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

import datetime
import json
import os
import pickle
from contextlib import redirect_stdout

import h5py
import numpy as np
import torch
from torchinfo import summary
from torchview import draw_graph

import amfinder_config as AmfConfig
import amfinder_log as AmfLog
import amfinder_plot as AmfPlot
import amfinder_zipfile as zf
from api_objects import PredictionValues
from api_utils import save_predictions_to_db
from db_config import connect


def now():
    """
    Returns the current date/time in a format suitable
    for use as file name.
    """

    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def save_training_data(history, model, save_path):
    """
    Saves training history, model weights, and other relevant information.

    :param history: Training history dictionary.
    :param model: The trained model.
    :param save_path: The directory to save the training data.
    """
    zipf = now() + "_training.zip"
    zipf = os.path.join(AmfConfig.get("outdir"), zipf)

    with zf.ZipFile(zipf, "w") as z:
        # Save the training history
        data = pickle.dumps(history, protocol=pickle.HIGHEST_PROTOCOL)
        z.writestr("history.bin", data)

        # Option 1: Save the model state_dict as h5
        model_path = os.path.join(save_path, "model.h5")
        with h5py.File(model_path, "w") as h5file:
            # Create group to store model information
            group = h5file.create_group("model")

            # Save parameter (weights and biases)
            for name, param in model.named_parameters():
                group.create_dataset(
                    name, data=param.data.cpu().numpy()
                )  # Numpy array conversion

        z.writestr("model.h5", open(model_path, "rb").read())
        ## TODO: An additional model load class (loading the state dict seperately to the architecture) would be required in the case of HDF5

        # Option 2: Save the model state_dict as pth
        model_path = os.path.join(save_path, "model.pth")
        torch.save(model, model_path)
        z.writestr("model.pth", open(model_path, "rb").read())

        # Optionally, save additional plots or metrics here
        # In the original AMFinder version, the accuracy was tracked throughout the training process.
        # As this metric is not used any longer, it is not included in the PyTorch version of the code.´

        # TODO: Integrate save mechanism when Early Stopping is triggered.
        # With early stopping implemented
        early = AmfConfig.get("early_stopping")
        if early is not None and early.early_stop:
            break_epoch = AmfConfig.get("early_break_epoch")
            x_range = np.arange(0, break_epoch)

        else:
            train_active_learning = AmfConfig.get("train_active_learning")
            epochs = (
                AmfConfig.get("epochs")
                if not train_active_learning
                else AmfConfig.get("epochs_active_learning")
            )
            x_range = np.arange(0, epochs)

        AmfPlot.initialize()
        data = AmfPlot.draw(history, epochs, "Loss", x_range, "loss", "val_loss")
        z.writestr("loss.png", data.getvalue())

    print(f"Saved model output to {save_path}")


# Function to get a summary of the model architecture as txt file and graph.
def save_model_architecture(model, device, tile_size):
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
    # TODO Currently this creates three files: "CNN1_summary.txt", "CNN1_graph.png" and "CNN1_graph". The third file is not required in theory.


def save_settings(path):
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


def save_metrics(metrics_collector, path):
    """
    Saves predictions, class metrics, file metrics, and generic metrics to an archive.
    :param path: path to the ZIP archive.
    :param metrics_collector: Instance of MetricsCollector containing metrics to save.
    """
    uniq = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    zipf = os.path.join(path, f"results_{uniq}.zip")
    print(f"    - saving as {zipf}... ", end="")

    with zf.ZipFile(zipf, "w") as z:
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
            z.writestr(generic_metrics_file_name, generic_metrics_csv)

        for image_name, image in metrics_collector.images.items():
            if isinstance(image, str):
                with open(image, "rb") as img_file:
                    z.writestr(image_name, img_file.read())
            else:
                z.writestr(image_name, image)


def prediction_table(results, path):
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

            AmfLog.info("Saved results to DB")

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

            AmfLog.info(f"Saved results to CSV at {path}")

            # print(f"Saved results to {os.path.join(directory, image_name + uniq + "cnn_1_predictions.csv")} file")
