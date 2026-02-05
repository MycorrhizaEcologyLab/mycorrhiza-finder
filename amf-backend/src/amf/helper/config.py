# AMFinder - config.py
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
AMFinder configuration module.
Read command-line arguments and store user settings.

Variables
------------
:HEADERS: Table headers for the different models.
:PAR: User settings.

Functions
------------
:function human_redable_header: Human-readable annotation class labels.
:function get: Retrieve the value associated with the given parameter ID.
:function set: Assign a new value to the given parameter ID.
:function training_subparser: Define the command-line parser used in training mode.
:function prediction_subparser: Define the command-line parser used in prediction mode.
:function test_subparser: Define the command-line parser used in test mode.
:function build_arg_parser: Build the full command-line parser.
:function import_settings: Read tile size from `settings.json`.
:function get_input_files: Return the list of vaid input images (based on MIME type).
:function initialize: Read command-line arguments and store user-defined values.
"""

import datetime
import glob
import mimetypes
import os
from argparse import ArgumentParser, RawTextHelpFormatter
from typing import Any, cast

import torch
from loguru import logger

from amf.helper.api_objects import (
    BaseConfig,
    CalibrateConfig,
    ConvertConfig,
    PredictionConfig,
    TestConfig,
    TifConversionConfig,
    TrainConfig,
)

USER_HOME_LOC = "~"
RESULTS_FOLDER_NAME = "amfinder"


def get_default_output_dir() -> str | None:
    # Returns the path to the user home directory, Documents if it exists,
    # based on what exists and is writeable
    home = os.path.expanduser(USER_HOME_LOC)
    docs_path = os.path.join(home, "Documents")
    if os.path.exists(docs_path) and os.access(docs_path, os.W_OK | os.X_OK):
        output_dir = docs_path
    elif os.path.exists(home) and os.access(home, os.W_OK | os.X_OK):
        output_dir = home
    else:
        output_dir = None
    return output_dir


def create_results_dir(label: str = "") -> str:
    # Create timestamped folder for results
    default_output_loc = get_default_output_dir()
    if default_output_loc is None:
        raise Exception(
            "Error: output directory not specified and the default location is not "
            "writeable. Cannot continue."
        )
    results_dir_path = os.path.join(
        default_output_loc,
        RESULTS_FOLDER_NAME,
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    if len(label):
        results_dir_path += "_" + label
    os.makedirs(results_dir_path)
    logger.info(f"Output directory created: {results_dir_path}")
    return results_dir_path


HEADERS = {
    "am": ["AMColonised", "Uncolonised", "Background", "Unreadable", "DSE", "Hybrid"],
    "erm": [
        "BlueCoils",
        "BrownCoils",
        "TypeTwo",
        "Uncolonised",
        "Background",
        "MainRoot",
        "Unreadable",
        "DSE",
        "HybridErm",
        "HybridDse",
    ],
}

HUMAN_HEADERS = {
    "am": ["AM+", "N-", "Background", "Unreadable", "DSE", "Hybrid"],
    "erm": [
        "BlueCoils",
        "BrownCoils",
        "TypeTwo",
        "Uncolonised",
        "Background",
        "MainRoot",
        "Unreadable",
        "DSE",
        "HybridErm",
        "HybridDse",
    ],
}
DESCRIPTIONS = {
    "am": ["amcolonised", "non-colonised", "background", "unreadable", "dse", "hybrid"],
    "erm": [
        "BlueCoils",
        "BrownCoils",
        "TypeTwo",
        "Uncolonised",
        "Background",
        "MainRoot",
        "Unreadable",
        "DSE",
        "HybridErm",
        "HybridDse",
    ],
}

PAR = {
    # Use local DB
    "use_db": True,
    # Set main run mode
    "run_mode": None,
    # Set device to CPU or GPU
    "device": "automatic",
    # Pick network (AM)
    "model": "am_252_efficientnet.pth",
    # Pick network (ErM)
    "model_erm": "erm_126_efficientnet.pth",
    # Type of model - cnn1, resnet, resnext, efficientnet and efficientnetv2 supported
    "model_type": "efficientnet",
    # Define whether to use pre-trained model or not
    "pre_trained": True,
    # Enable BALD/BatchBALD training mode
    "train_active_learning": False,
    # Use BALD to provide labels
    "get_tiles_for_labelling_using_active_learning": False,
    # Specify tile edge to use
    "tile_edge": 252,
    # Define path to input files
    "input_files": None,
    # Turn on MLFlow tracking
    "mlflow_flag": False,
    # Set training params
    "batch_size": 32,
    "learning_rate": 0.000004218361045,
    # learning rate after active learning
    "learning_rate_active_learning": 0.0000004,
    "adam_beta1": 0.906450740008503,
    "adam_beta2": 0.986390310777448,
    # Balancing factor for balancing dataset
    "balance_factor": 1.24895925434138,
    # Num of epochs for training
    "epochs": 50,
    # NUm epochs after active learning
    "epochs_active_learning": 5,
    # Proportion of training set to use for validation
    "vfrac": 0.2,
    # Only convert preds to annots if above this threshold
    "threshold": 0.5,
    # Bool for carrying out data augmentation
    "data_augm": False,
    # Save model architecture after training
    "summary": False,
    # Early stopping patience
    "patience_e": 10,
    # Learning rate reduction patience
    "patience_r": 5,
    "outdir": None,
    # Specify classes
    "header": HEADERS["am"],
    # Several monitoring functions
    "monitors": {
        "early_stopping": None,
        "reduce_lr_on_plateau": None,
    },
    # Whether to carry out Arbusucular or Ericoid processes
    "colonisation_type": "am",
    "class_names": {
        "am": HEADERS["am"],
        "erm": HEADERS["erm"],
    },
    # Dict to store tiles in when tiling an image for the front end
    "tiles": {},
    # Argument to aggregate 126 to 252 tiles
    "aggregate_tiles": False,
    # Image type to convert tifs to
    "convert_image_file_type": "jpg",
    # Only output colonisation percentages when running test
    "active_learning_method": "bald",  # Active learning method (bald, batchbald)
    "num_samples_for_labelling": 10,  # Samples to select per file for labeling
    "mc_samples": 50,  # Number of MC dropout samples for uncertainty
    "dropout_rate": 0.25,  # Dropout probability
    # This value points towards the temperature value for the model (AM)
    "temperature_factor_path": "am_252_efficientnet_temperature_value.txt",
    # This value points towards the temperature value for the model (ErM)
    "temperature_factor_path_erm": "erm_126_efficientnet_temperature_value.txt",
    "num_workers": 0,
    "use_contextual_confidence": True,  # Enable contextual confidence refinement
    "contextual_confidence_threshold": 1,  # Threshold for applying max voting
}


MODEL_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "trained_networks"
)


def get_model_dir() -> str:
    """Returns the trained models directory."""

    return MODEL_DIR


def invite() -> str:
    """
    Command-line invite
    """
    return datetime.datetime.now().strftime("%H:%M:%S")


def human_readable_header() -> list[str]:
    """
    Return the human-readable header of the current model.
    """
    colonisation_type = cast(str, PAR["colonisation_type"])
    return HUMAN_HEADERS[colonisation_type]


def get(id_: str) -> Any:
    """
    Retrieve application settings.

    :param id: Unique identifier.
    """

    id_ = id_.lower()

    if id_ in PAR:
        # Special case, look into a specific folder.
        if id_ in ["model", "model_erm"] and PAR[id_] is not None:
            # Check if model exists in trained networks, else use absolute path
            model_path = cast(str, PAR[id_])
            model_name = os.path.basename(model_path)
            path = os.path.join(get_model_dir(), model_name)

            if not os.path.isfile(path):
                path = cast(str, PAR[id_])

            return path

        else:
            return PAR[id_]

    elif id_ in PAR["monitors"]:  # type: ignore[operator] # TODO typed parameters
        monitors: dict[str, Any] = PAR["monitors"]  # type: ignore[assignment]
        return monitors[id_]

    else:
        logger.warning(f"Unknown parameter {id_}")
        return None


def find_files_in_directory(directory: str, convert_tiff: bool = False) -> list[str]:
    # Search for .jpg, .jpeg & .png files
    search_patterns = [
        os.path.join(directory, "**", "*.jpg"),
        os.path.join(directory, "**", "*.jpeg"),
        os.path.join(directory, "**", "*.png"),
    ]

    if convert_tiff:
        logger.info("Also searching for tif files")
        search_patterns.append(os.path.join(directory, "**", "*.tif"))
        search_patterns.append(os.path.join(directory, "**", "*.tiff"))

    img_files = []
    for pattern in search_patterns:
        img_files.extend(glob.glob(pattern, recursive=True))

    if len(img_files) == 0:
        raise FileNotFoundError(f"No image files found in {directory}")

    return img_files


def set_(id_: str, value: Any, create: bool = False, use_none: bool = False) -> None:
    """
    Updates application settings.

    :param id: unique identifier.
    :param value: value to store.
    :param create: create id if it does not exist (optional).
    """

    if value is None and not use_none:
        return

    else:
        id_ = id_.lower()

        if id_ in PAR:
            PAR[id_] = value

            if id_ == "colonisation_type":
                PAR["header"] = HEADERS[value]

        elif id_ in PAR["monitors"]:  # type: ignore[operator] # TODO typed parameters
            PAR["monitors"][id_] = value  # type: ignore[index, call-overload]

        elif create:
            PAR[id_] = value

        else:
            logger.warning(f"Unknown parameter {id_}")


def add_training_subparser(subparsers) -> None:  # type: ignore[no-untyped-def]
    """
    Defines arguments used in training mode.

    :param subparsers: subparser generator.
    """

    parser = subparsers.add_parser(
        "train",
        help="learns how to identify AMF structures.",
        formatter_class=RawTextHelpFormatter,
    )

    parser.add_argument(
        "-l",
        "--use-csvs",
        action="store_const",
        dest="use_db",
        const=False,
        help="Use CSVs instead of DB.",
    )

    parser.add_argument(
        "-i",
        "--images",
        type=str,
        action="store",
        required=True,
        help="Directory of images to process.",
    )

    x = PAR["train_active_learning"]
    parser.add_argument(
        "-tal",
        "--train-active-learning",
        action="store_const",
        dest="train_active_learning",
        const=True,
        help=(
            "Enables training with different default learning rate and num epochs "
            "after active learning samples have been acquired."
        ),
    )

    x = PAR["get_tiles_for_labelling_using_active_learning"]
    parser.add_argument(
        "-gtfl",
        "--get-tiles-for-labelling",
        action="store_const",
        dest="get_tiles_for_labelling_using_active_learning",
        const=True,
        help="Run active learning to get tiles for labelling.",
    )

    x = PAR["active_learning_method"]
    parser.add_argument(
        "-alm",
        "--active-learning-method",
        action="store",
        dest="active_learning_method",
        type=str,
        default=x,
        help="Active learning method to use (bald, batchbald)."
        "\ndefault value: {}".format(x),
    )

    x = PAR["num_samples_for_labelling"]
    parser.add_argument(
        "-sfl",
        "--samples-for-labelling",
        action="store",
        dest="num_samples_for_labelling",
        type=int,
        default=x,
        help="Number of samples to select per file for labeling."
        "\ndefault value: {}".format(x),
    )

    x = PAR["mc_samples"]
    parser.add_argument(
        "-mc",
        "--mc-samples",
        action="store",
        dest="mc_samples",
        type=int,
        default=x,
        help="Number of Monte Carlo samples for uncertainty estimation."
        "\ndefault value: {}".format(x),
    )

    x = PAR["dropout_rate"]
    parser.add_argument(
        "-dr",
        "--dropout-rate",
        action="store",
        dest="dropout_rate",
        type=float,
        default=x,
        help="Dropout rate to use in the model for uncertainty-based acquisition."
        "\ndefault value: {}".format(x),
    )

    x = PAR["mlflow_flag"]
    parser.add_argument(
        "-mfl",
        "--mlflow",
        action="store_const",
        dest="mlflow_flag",
        const=True,
        help="Enables mlflow tracking in training.",
    )

    x = PAR["batch_size"]
    parser.add_argument(
        "-b",
        "--batch_size",
        action="store",
        dest="batch_size",
        metavar="NUM",
        type=int,
        default=x,
        help="training batch size.\ndefault value: {}".format(x),
    )

    x = PAR["data_augm"]
    parser.add_argument(
        "-a",
        "--data_augmentation",
        action="store_true",
        dest="data_augm",
        default=x,
        help="apply data augmentation (hue, chroma, saturation, etc.)"
        "\nby default, data augmentation is not used.",
    )

    x = PAR["summary"]
    parser.add_argument(
        "-s",
        "--summary",
        action="store_true",
        dest="summary",
        default=x,
        help="save CNN architecture (CNN graph and model summary)"
        "\nby default, does not save any information.",
    )

    x = PAR["outdir"]
    parser.add_argument(
        "-o",
        "--outdir",
        action="store",
        dest="outdir",
        default=x,
        help="folder where to save trained model and CNN architecture."
        "\ndefault: {}".format(x),
    )

    x = PAR["epochs"]
    parser.add_argument(
        "-e",
        "--epochs",
        action="store",
        dest="epochs",
        metavar="NUM",
        type=int,
        default=x,
        help="number of epochs to run.\ndefault value: {}".format(x),
    )

    x = PAR["epochs_active_learning"]
    parser.add_argument(
        "-eal",
        "--epochs-active-learning",
        action="store",
        dest="epochs_active_learning",
        metavar="NUM",
        type=int,
        default=x,
        help="number of epochs to run after active learning.\ndefault value: {}".format(
            x
        ),
    )

    x = PAR["patience_e"]
    parser.add_argument(
        "-pe",
        "--patience_e",
        action="store",
        dest="patience_e",
        metavar="NUM",
        type=int,
        default=x,
        help="number of epochs to wait before early stopping is triggered."
        "\ndefault value: {}".format(x),
    )

    x = PAR["patience_r"]
    parser.add_argument(
        "-pr",
        "--patience_r",
        action="store",
        dest="patience_r",
        metavar="NUM",
        type=int,
        default=x,
        help="number of epochs to wait before learning rate reduction is triggered."
        "\ndefault value: {}".format(x),
    )

    x = PAR["learning_rate"]
    parser.add_argument(
        "-lr",
        "--learning_rate",
        action="store",
        dest="learning_rate",
        metavar="NUM",
        type=float,
        default=x,
        help="learning rate used by the Adam optimizer.\ndefault value: {}".format(x),
    )

    x = PAR["learning_rate_active_learning"]
    parser.add_argument(
        "-lral",
        "--learning_rate-active-learning",
        action="store",
        dest="learning_rate_active_learning",
        metavar="NUM",
        type=float,
        default=x,
        help="learning rate used by the Adam optimizer after active learning."
        "\ndefault value: {}".format(x),
    )

    x = PAR["adam_beta1"]
    parser.add_argument(
        "-ab1",
        "--adam_beta1",
        action="store",
        dest="adam_beta1",
        metavar="NUM",
        type=float,
        default=x,
        help="Beta 1 Hyperparameter for Adam optimiser\ndefault value: {}".format(x),
    )

    x = PAR["adam_beta2"]
    parser.add_argument(
        "-ab2",
        "--adam_beta2",
        action="store",
        dest="adam_beta2",
        metavar="NUM",
        type=float,
        default=x,
        help="Beta 2 Hyperparameter for Adam optimiser\ndefault value: {}".format(x),
    )

    x = PAR["balance_factor"]
    parser.add_argument(
        "-bf",
        "--balance_factor",
        action="store",
        dest="balance_factor",
        metavar="NUM",
        type=float,
        default=x,
        help=(
            "multiplier for balancing datasets based on class sizes and oriented on "
            "Colonised classes.\ndefault value: {}".format(x)
        ),
    )

    x = PAR["vfrac"]
    parser.add_argument(
        "-vf",
        "--validation_fraction",
        action="store",
        dest="vfrac",
        metavar="N",
        type=int,
        default=x,
        help="Proportion of tiles used for validation.\ndefault value: {}%%".format(x),
    )

    x = None  # by default, do not fine-tune
    parser.add_argument(
        "-net",
        "--network",
        action="store",
        dest="model",
        metavar="pth",
        type=str,
        default=x,
        help="name of the pre-trained network to use as a basis for training for AM."
        "\ndefault value: {}".format(x),
    )

    x = None  # by default, do not fine-tune
    parser.add_argument(
        "-neterm",
        "--network_erm",
        action="store",
        dest="model_erm",
        metavar="pth",
        type=str,
        default=x,
        help="name of the pre-trained network to use as a basis for training for ErM."
        "\ndefault value: {}".format(x),
    )

    x = PAR["model_type"]
    parser.add_argument(
        "-mt",
        "--model_type",
        action="store",
        dest="model_type",
        metavar="pth",
        type=str,
        default=x,
        help=(
            "Choice for new model intialisation: cnn1, resnet, resnext, efficientnet, "
            "efficientnetv2.\ndefault value: {}".format(x)
        ),
    )

    parser.add_argument(
        "-pretr",
        "--pretrain",
        action="store_true",
        dest="pre_trained",
        help=(
            "Loads ImageNet weights if ResNet, ResNeXt or EfficentNet is selected for "
            "Model type."
        ),
    )

    parser.add_argument(
        "-nopretr",
        "--no-pretrain",
        action="store_false",
        dest="pre_trained",
        help=("Do not load ImageNet weights."),
    )

    x = PAR["colonisation_type"]
    parser.add_argument(
        "-ct",
        "--colonisation_type",
        action="store",
        dest="colonisation_type",
        type=str,
        default=x,
        help="Choosing between Abuscular and Ericoid colonisation.",
    )

    x = PAR["tile_edge"]
    parser.add_argument(
        "-size",
        "--tile_size",
        action="store",
        dest="edge",
        type=int,
        default=x,
        help="Tile size (in pixels) used for image segmentation."
        "\ndefault value: {} pixels".format(x),
    )

    return


def add_test_subparser(subparsers) -> None:  # type: ignore[no-untyped-def]
    """
    Defines arguments used in test  mode.

    :param subparsers: subparser generator.
    """

    parser = subparsers.add_parser(
        "test",
        help="Runs AMFinder in test mode.",
        formatter_class=RawTextHelpFormatter,
    )

    parser.add_argument(
        "-l",
        "--use-csvs",
        action="store_const",
        dest="use_db",
        const=False,
        help="Use CSVs instead of DB.",
    )

    parser.add_argument(
        "-i",
        "--images",
        type=str,
        action="store",
        required=True,
        help="Directory of images to process.",
    )

    x = PAR["model"]
    parser.add_argument(
        "-net",
        "--network",
        action="store",
        dest="model",
        metavar="H5",
        type=str,
        default=x,
        help="name of the pre-trained model to use for predictions for AM."
        "\ndefault value: {}".format(x),
    )

    x = PAR["model_erm"]
    parser.add_argument(
        "-neterm",
        "--network_erm",
        action="store",
        dest="model_erm",
        metavar="pth",
        type=str,
        default=x,
        help="name of the pre-trained model to use for predictions for ErM."
        "\ndefault value: {}".format(x),
    )

    x = PAR["outdir"]
    parser.add_argument(
        "-o",
        "--outdir",
        action="store",
        dest="outdir",
        default=x,
        help="folder where to save trained model and CNN architecture."
        "\ndefault: {}".format(x),
    )

    x = PAR["colonisation_type"]
    parser.add_argument(
        "-ct",
        "--colonisation_type",
        action="store",
        dest="colonisation_type",
        type=str,
        default=x,
        help="Choosing between Abuscular and Ericoid colonisation.",
    )

    x = PAR["tile_edge"]
    parser.add_argument(
        "-size",
        "--tile_size",
        action="store",
        dest="edge",
        type=int,
        default=x,
        help="Tile size (in pixels) used for image segmentation."
        "\ndefault value: {} pixels".format(x),
    )

    x = PAR["temperature_factor_path"]
    parser.add_argument(
        "-t",
        "--temperature-factor-path",
        action="store",
        dest="temperature_factor_path",
        type=str,
        default=x,
        help="name of the file that contains the temperature factor for the AM model."
        "\ndefault value: {}".format(x),
    )

    x = PAR["use_contextual_confidence"]
    parser.add_argument(
        "-ucc",
        "--use-contextual-confidence",
        action="store_const",
        dest="use_contextual_confidence",
        const=True,
        help="Enable contextual confidence refinement for low confidence predictions",
    )

    x = PAR["contextual_confidence_threshold"]
    parser.add_argument(
        "-cct",
        "--contextual-confidence-threshold",
        action="store",
        dest="contextual_confidence_threshold",
        metavar="N",
        type=float,
        default=x,
        help="Threshold below which max voting with surrounding tiles will be applied."
        "\ndefault value: {}".format(x),
    )

    x = PAR["temperature_factor_path_erm"]
    parser.add_argument(
        "-term",
        "--temperature-factor-path-erm",
        action="store",
        dest="temperature_factor_path_erm",
        type=str,
        default=x,
        help="name of the file that contains the temperature factor for the ErM model."
        "\ndefault value: {}".format(x),
    )

    return


def add_colonisation_subparser(subparsers) -> None:  # type: ignore[no-untyped-def]
    """
    Defines arguments used in colonisation mode.

    :param subparsers: subparser generator.
    """

    parser = subparsers.add_parser(
        "colonisation",
        help="Runs AMFinder in colonisation mode.",
        formatter_class=RawTextHelpFormatter,
    )

    parser.add_argument(
        "-l",
        "--use-csvs",
        action="store_const",
        dest="use_db",
        const=False,
        help="Use CSVs instead of DB.",
    )

    parser.add_argument(
        "-i",
        "--images",
        type=str,
        action="store",
        required=True,
        help="Directory of images to process.",
    )

    x = PAR["model"]
    parser.add_argument(
        "-net",
        "--network",
        action="store",
        dest="model",
        metavar="pth",
        type=str,
        default=x,
        help="name of the pre-trained model to use for predictions for AM."
        "\ndefault value: {}".format(x),
    )

    x = PAR["model_erm"]
    parser.add_argument(
        "-neterm",
        "--networkerm",
        action="store",
        dest="model_erm",
        metavar="pth",
        type=str,
        default=x,
        help="name of the pre-trained model to use for predictions for ErM."
        "\ndefault value: {}".format(x),
    )

    x = PAR["outdir"]
    parser.add_argument(
        "-o",
        "--outdir",
        action="store",
        dest="outdir",
        default=x,
        help="folder where to save trained model and CNN architecture."
        "\ndefault: {}".format(x),
    )

    x = PAR["colonisation_type"]
    parser.add_argument(
        "-ct",
        "--colonisation_type",
        action="store",
        dest="colonisation_type",
        type=str,
        default=x,
        help="Choosing between Abuscular and Ericoid colonisation.",
    )

    x = PAR["tile_edge"]
    parser.add_argument(
        "-size",
        "--tile_size",
        action="store",
        dest="edge",
        type=int,
        default=x,
        help="Tile size (in pixels) used for image segmentation."
        "\ndefault value: {} pixels".format(x),
    )

    return


def add_prediction_subparser(subparsers) -> None:  # type: ignore[no-untyped-def]
    """
    Defines arguments used in prediction mode.

    :param subparsers: subparser generator.
    """

    parser = subparsers.add_parser(
        "predict",
        help="Runs AMFinder in prediction mode.",
        formatter_class=RawTextHelpFormatter,
    )

    parser.add_argument(
        "-l",
        "--use-csvs",
        action="store_const",
        dest="use_db",
        const=False,
        help="Use CSVs instead of DB.",
    )

    parser.add_argument(
        "-i",
        "--images",
        type=str,
        action="store",
        required=True,
        help="Directory of images to process.",
    )

    x = PAR["tile_edge"]
    parser.add_argument(
        "-size",
        "--tile_size",
        action="store",
        dest="edge",
        type=int,
        default=x,
        help="Tile size (in pixels) used for image segmentation."
        "\ndefault value: {} pixels".format(x),
    )

    x = PAR["model"]
    parser.add_argument(
        "-net",
        "--network",
        action="store",
        dest="model",
        metavar="pth",
        type=str,
        default=x,
        help="name of the pre-trained model to use for predictions for AM."
        "\ndefault value: {}".format(x),
    )

    x = PAR["model_erm"]
    parser.add_argument(
        "-neterm",
        "--network_erm",
        action="store",
        dest="model_erm",
        metavar="pth",
        type=str,
        default=x,
        help="name of the pre-trained model to use for predictions for ErM."
        "\ndefault value: {}".format(x),
    )

    parser.add_argument(
        "-o",
        "--outdir",
        action="store",
        dest="outdir",
        default=None,
        help="where to store results metrics files."
        "\ndefault value: same as input images.",
    )

    x = PAR["colonisation_type"]
    parser.add_argument(
        "-ct",
        "--colonisation_type",
        action="store",
        dest="colonisation_type",
        type=str,
        default=x,
        help="Choosing between Abuscular and Ericoid colonisation.",
    )

    x = PAR["temperature_factor_path"]
    parser.add_argument(
        "-t",
        "--temperature-factor-path",
        action="store",
        dest="temperature_factor_path",
        type=str,
        default=x,
        help="name of the file that contains the temperature factor for the AM model."
        "\ndefault value: {}".format(x),
    )

    x = PAR["temperature_factor_path_erm"]
    parser.add_argument(
        "-term",
        "--temperature-factor-path-erm",
        action="store",
        dest="temperature_factor_path_erm",
        type=str,
        default=x,
        help="name of the file that contains the temperature factor for the ErM model."
        "\ndefault value: {}".format(x),
    )

    x = PAR["use_contextual_confidence"]
    parser.add_argument(
        "-ucc",
        "--use-contextual-confidence",
        action="store_const",
        dest="use_contextual_confidence",
        const=True,
        help="Enable contextual confidence refinement for low confidence predictions",
    )

    x = PAR["contextual_confidence_threshold"]
    parser.add_argument(
        "-cct",
        "--contextual-confidence-threshold",
        action="store",
        dest="contextual_confidence_threshold",
        metavar="N",
        type=float,
        default=x,
        help="Threshold below which max voting with surrounding tiles will be applied."
        "\ndefault value: {}".format(x),
    )

    return


def add_conversion_subparser(subparsers) -> None:  # type: ignore[no-untyped-def]
    parser = subparsers.add_parser(
        "convert",
        help="Runs AMFinder in conversion mode.",
        formatter_class=RawTextHelpFormatter,
    )

    parser.add_argument(
        "-l",
        "--use-csvs",
        action="store_const",
        dest="use_db",
        const=False,
        help="Use CSVs instead of DB.",
    )

    x = PAR["threshold"]
    parser.add_argument(
        "-th",
        "--threshold",
        action="store",
        dest="threshold",
        metavar="N",
        type=float,
        default=x,
        help="threshold for conversion: {}".format(x),
    )

    x = PAR["input_files"]
    parser.add_argument(
        "-i",
        "--images",
        type=str,
        action="store",
        default=x,
        help="plant root image to process.\ndefault value: {}".format(x),
    )

    x = PAR["colonisation_type"]
    parser.add_argument(
        "-ct",
        "--colonisation_type",
        action="store",
        dest="colonisation_type",
        type=str,
        default=x,
        help="Choosing between Abuscular and Ericoid colonisation.",
    )

    parser.add_argument(
        "-agg",
        "--aggregate",
        action="store_const",
        dest="aggregate_tiles",
        const=True,
        help="Argument to trigger tile upscaling by factor of 2",
    )

    x = PAR["tile_edge"]
    parser.add_argument(
        "-size",
        "--tile_size",
        action="store",
        dest="edge",
        type=int,
        default=x,
        help="Tile size (in pixels) used for image segmentation."
        "\ndefault value: {} pixels".format(x),
    )

    x = PAR["use_contextual_confidence"]
    parser.add_argument(
        "-ucc",
        "--use-contextual-confidence",
        action="store_const",
        dest="use_contextual_confidence",
        const=True,
        help="Enable contextual confidence refinement for low confidence predictions",
    )

    return


def add_tif_conversion_subparser(subparsers) -> None:  # type: ignore[no-untyped-def]
    parser = subparsers.add_parser(
        "tifconversion",
        help="Runs AMFinder in TIF conversion mode.",
        formatter_class=RawTextHelpFormatter,
    )

    parser.add_argument(
        "-l",
        "--use-csvs",
        action="store_const",
        dest="use_db",
        const=False,
        help="Use CSVs instead of DB.",
    )

    x = PAR["input_files"]
    parser.add_argument(
        "-i",
        "--images",
        type=str,
        action="store",
        default=x,
        help="plant root image to process.\ndefault value: {}".format(x),
    )

    x = PAR["colonisation_type"]
    parser.add_argument(
        "-ct",
        "--colonisation_type",
        action="store",
        dest="colonisation_type",
        type=str,
        default=x,
        help="Choosing between Abuscular and Ericoid colonisation.",
    )

    x = PAR["convert_image_file_type"]
    parser.add_argument(
        "-it",
        "--image-type",
        action="store",
        dest="convert_image_file_type",
        type=str,
        default=x,
        help="Choosing between jpg and png for file type to convert tifs to.",
    )

    x = PAR["tile_edge"]
    parser.add_argument(
        "-size",
        "--tile_size",
        action="store",
        dest="edge",
        type=int,
        default=x,
        help="Tile size (in pixels) used for image segmentation."
        "\ndefault value: {} pixels".format(x),
    )

    return


def add_calibrate_subparser(subparsers) -> None:  # type: ignore[no-untyped-def]
    """
    Defines arguments used in calibration mode.

    :param subparsers: subparser generator.
    """

    parser = subparsers.add_parser(
        "calibrate",
        help="Runs calibration for AMFinder",
        formatter_class=RawTextHelpFormatter,
    )

    parser.add_argument(
        "-l",
        "--use-csvs",
        action="store_const",
        dest="use_db",
        const=False,
        help="Use CSVs instead of DB.",
    )

    x = PAR["model"]
    parser.add_argument(
        "-net",
        "--network",
        action="store",
        dest="model",
        metavar="H5",
        type=str,
        default=x,
        help="name of the model which shall be calibrated for AM."
        "\ndefault value: {}".format(x),
    )

    x = PAR["model_erm"]
    parser.add_argument(
        "-neterm",
        "--network_erm",
        action="store",
        dest="model_erm",
        metavar="pth",
        type=str,
        default=x,
        help="name of the model which shall be calibrated for ErM."
        "\ndefault value: {}".format(x),
    )

    x = PAR["input_files"]
    parser.add_argument(
        "-i",
        "--images",
        type=str,
        action="store",
        required=True,
        help="Directory of images to calibrate on.",
    )

    x = PAR["outdir"]
    parser.add_argument(
        "-o",
        "--outdir",
        action="store",
        dest="outdir",
        default=x,
        help="folder where to save trained model and CNN architecture."
        "\ndefault: {}".format(x),
    )

    x = PAR["colonisation_type"]
    parser.add_argument(
        "-ct",
        "--colonisation_type",
        action="store",
        dest="colonisation_type",
        type=str,
        default=x,
        help="Choosing between Abuscular and Ericoid colonisation.",
    )

    x = PAR["tile_edge"]
    parser.add_argument(
        "-size",
        "--tile_size",
        action="store",
        dest="edge",
        type=int,
        default=x,
        help="Tile size (in pixels) used for image segmentation."
        "\ndefault value: {} pixels".format(x),
    )

    return


def build_arg_parser() -> ArgumentParser:
    """
    Builds AMFinder command-line parser.
    """

    main = ArgumentParser(
        description="AMFinder command-line arguments.",
        allow_abbrev=False,
        formatter_class=RawTextHelpFormatter,
    )

    subparsers = main.add_subparsers(
        dest="run_mode", required=True, help="action to be performed."
    )

    add_training_subparser(subparsers)
    add_prediction_subparser(subparsers)
    add_test_subparser(subparsers)
    add_conversion_subparser(subparsers)
    add_calibrate_subparser(subparsers)
    add_colonisation_subparser(subparsers)
    add_tif_conversion_subparser(subparsers)

    return main


def abspath(files: list[str]) -> list[str]:
    """
    Returns absolute paths to input files.

    :param files: Raw list of input file names (can contain wildcards).
    """
    files = sum([glob.glob(x) for x in files], [])
    return [os.path.abspath(x) for x in files]


def get_input_files() -> list[str]:
    """
    Filter input file list and keep valid JPEG or TIFF images.
    """

    raw_list = abspath(get("input_files"))

    valid_types = ["image/jpeg", "image/tiff", "image/png"]
    images = [x for x in raw_list if mimetypes.guess_type(x)[0] in valid_types]
    logger.info(f"Input images: {len(images)}")
    return images


def clean_path(path: str) -> str:
    """
    Remove wildcards etc from the path and just return
    the directory

    Args:
        path (str): path to clean

    Returns:
        str: clean path to directory
    """
    if os.path.isdir(path):
        return path
    else:
        return os.path.dirname(path)


def set_train_config(trainConfig: TrainConfig) -> None:
    """
    Set configuration for the training run.

    Args:
        trainConfig (TrainConfig): Configuration object containing training parameters.

    Sets various parameters such as input files, model details, batch size,
    number of epochs, and output directory for the training process.
    """
    set_("run_mode", "train")
    set_device(cast(str, trainConfig.device))

    set_("train_active_learning", trainConfig.trainActiveLearning)
    set_(
        "get_tiles_for_labelling_using_active_learning",
        trainConfig.getTilesForLabellingUsingActiveLearning,
    )
    set_("active_learning_method", trainConfig.activeLearningMethod)
    set_("num_samples_for_labelling", trainConfig.numSamplesForLabelling)
    set_("mc_samples", trainConfig.mcSamples)
    set_("dropout_rate", trainConfig.dropoutRate)
    files = find_files_in_directory(trainConfig.inputFiles)
    set_("input_files", files)

    set_("use_db", trainConfig.useDb)
    set_("mlflow_flag", trainConfig.mlflowFlag)
    set_("colonisation_type", trainConfig.colonisationType)
    set_("batch_size", trainConfig.batchSize)
    set_("adam_beta1", trainConfig.adamBeta1)
    set_("adam_beta2", trainConfig.adamBeta2)
    set_("balance_factor", trainConfig.balanceFactor)
    # Override learning rate and epochs if train_active_learning is enabled
    if trainConfig.trainActiveLearning:
        set_("learning_rate_active_learning", trainConfig.learningRateActiveLearning)
        set_("epochs_active_learning", trainConfig.epochsActiveLearning)
        logger.info("Training mode after active learning")
    else:
        set_("learning_rate", trainConfig.learningRate)
        set_("epochs", trainConfig.epochs)
    # Model paths can be set to None by default for fresh training
    set_("model", trainConfig.model, use_none=True)
    set_("model_erm", trainConfig.modelErm, use_none=True)
    set_("model_type", trainConfig.modelType)
    set_("pre_trained", trainConfig.preTrained)
    set_("vfrac", trainConfig.vfrac)
    set_("data_augm", trainConfig.dataAugm)
    set_("summary", trainConfig.summary)
    set_("patience_e", trainConfig.patienceE)
    set_("patience_r", trainConfig.patienceR)
    set_("tile_edge", trainConfig.tileEdge)

    if trainConfig.outdir is None or trainConfig.outdir == "":
        results_dir = create_results_dir("train")
        set_("outdir", results_dir)
    else:
        set_("outdir", trainConfig.outdir)


def set_predict_config(predictionConfig: PredictionConfig) -> None:
    """
    Set configuration for the prediction run.

    Args:
        predictionConfig (PredictionConfig): Configuration object containing prediction
            parameters.

    This function sets parameters required for making predictions, such as
    input files, model, and output directory.
    """
    set_("run_mode", "predict")
    set_device(cast(str, predictionConfig.device))
    files = find_files_in_directory(predictionConfig.inputFiles)
    set_("input_files", files)
    set_("use_db", predictionConfig.useDb)
    set_("colonisation_type", predictionConfig.colonisationType)
    set_("tile_edge", predictionConfig.tileEdge)
    set_("model", predictionConfig.model)
    set_("model_erm", predictionConfig.modelErm)
    set_("temperature_factor_path", predictionConfig.temperatureFactorPath)
    set_("temperature_factor_path_erm", predictionConfig.temperatureFactorPathErm)

    if predictionConfig.outdir is None or predictionConfig.outdir == "":
        set_("outdir", clean_path(predictionConfig.inputFiles))
    else:
        set_("outdir", predictionConfig.outdir)

    set_("use_contextual_confidence", predictionConfig.useContextualConfidence)
    set_(
        "contextual_confidence_threshold",
        predictionConfig.contextualConfidenceThreshold,
    )


def set_test_config(testConfig: TestConfig) -> None:
    """
    Set configuration for the testing run.

    Args:
        testConfig (TestConfig): Configuration object containing testing parameters.

    This function configures the settings needed for the testing procedure,
    including input files, model, and output directory for results.
    """
    set_("run_mode", "test")
    set_device(cast(str, testConfig.device))
    files = find_files_in_directory(testConfig.inputFiles)
    set_("input_files", files)
    set_("use_db", testConfig.useDb)
    set_("colonisation_type", testConfig.colonisationType)
    set_("model", testConfig.model)
    set_("model_erm", testConfig.modelErm)
    set_("tile_edge", testConfig.tileEdge)
    set_("temperature_factor_path", testConfig.temperatureFactorPath)
    set_("temperature_factor_path_erm", testConfig.temperatureFactorPathErm)

    if testConfig.outdir is None or testConfig.outdir == "":
        results_dir = create_results_dir("test")
        set_("outdir", results_dir)
    else:
        set_("outdir", testConfig.outdir)

    set_("use_contextual_confidence", testConfig.useContextualConfidence)
    set_("contextual_confidence_threshold", testConfig.contextualConfidenceThreshold)


def set_colonisation_config(colonisationConfig: BaseConfig) -> None:
    """
    Set configuration for the testing run.

    Args:
        testConfig (TestConfig): Configuration object containing testing parameters.

    This function configures the settings needed for the testing procedure,
    including input files, model, and output directory for results.
    """
    set_("run_mode", "colonisation")
    set_device(cast(str, colonisationConfig.device))
    files = find_files_in_directory(colonisationConfig.inputFiles)
    set_("input_files", files)

    set_("use_db", colonisationConfig.useDb)
    set_("colonisation_type", colonisationConfig.colonisationType)
    set_("tile_edge", colonisationConfig.tileEdge)

    if colonisationConfig.outdir is None or colonisationConfig.outdir == "":
        results_dir = create_results_dir("colonisation_results")
        set_("outdir", results_dir)
    else:
        set_("outdir", colonisationConfig.outdir)


def set_convert_config(convertConfig: ConvertConfig) -> None:
    """
    Set configuration for the conversion run.

    Args:
        convertConfig (ConvertConfig): Configuration object containing conversion
            parameters.

    This function defines the necessary settings for annotations conversion,
    including input files, colonisation type, threshold, and output directory.
    """
    set_("run_mode", "convert")
    set_device(cast(str, convertConfig.device))
    files = find_files_in_directory(convertConfig.inputFiles)
    set_("input_files", files)
    set_("use_db", convertConfig.useDb)
    set_("aggregate_tiles", convertConfig.aggregateTiles)
    set_("colonisation_type", convertConfig.colonisationType)
    set_("threshold", convertConfig.threshold)
    set_("tile_edge", convertConfig.tileEdge)
    set_("use_contextual_confidence", convertConfig.useContextualConfidence)

    if convertConfig.outdir is None or convertConfig.outdir == "":
        set_("outdir", clean_path(convertConfig.inputFiles))
    else:
        set_("outdir", convertConfig.outdir)


def set_tif_conversion_config(tifConversionConfig: TifConversionConfig) -> None:
    """
    Set configuration for the tif conversion.

    Args:
        tifConversionConfig (TifConversionConfig): Configuration object containing TIF
            conversion parameters.

    This function defines the necessary settings for tif file conversion.
    """
    set_("run_mode", "tifconversion")
    set_device(cast(str, tifConversionConfig.device))
    files = find_files_in_directory(tifConversionConfig.inputFiles, convert_tiff=True)
    set_("input_files", files)
    set_("use_db", tifConversionConfig.useDb)
    set_("colonisation_type", tifConversionConfig.colonisationType)
    set_("convert_image_file_type", tifConversionConfig.convertImageFileType)
    set_("tile_edge", tifConversionConfig.tileEdge)

    if tifConversionConfig.outdir is None or tifConversionConfig.outdir == "":
        set_("outdir", clean_path(tifConversionConfig.inputFiles))
    else:
        set_("outdir", tifConversionConfig.outdir)


def set_calibrate_config(calibrateConfig: CalibrateConfig) -> None:
    """
    Set configuration for the calibration.

    Args:
        calibrateConfig (CalibrateConfig): Configuration object containing calibration
        parameters.

    This function sets parameters necessary for calibration models, including
    input files, model, and output directory for results.
    """
    set_("run_mode", "calibration")
    set_device(cast(str, calibrateConfig.device))
    files = find_files_in_directory(calibrateConfig.inputFiles)
    set_("input_files", files)
    set_("use_db", calibrateConfig.useDb)
    set_("colonisation_type", calibrateConfig.colonisationType)
    set_("model", calibrateConfig.model)
    set_("model_erm", calibrateConfig.modelErm)
    set_("tile_edge", calibrateConfig.tileEdge)

    if calibrateConfig.outdir is None or calibrateConfig.outdir == "":
        results_dir = create_results_dir("calibrate")
        set_("outdir", results_dir)
    else:
        set_("outdir", calibrateConfig.outdir)


def set_device(device: str) -> None:
    # Get the device setting
    if device == "automatic":
        if torch.cuda.is_available():
            set_("device", "cuda:0")
            logger.debug("Device set on automatic mode. Running via gpu")

        else:
            set_("num_workers", 0)
            set_("device", "cpu")
            logger.debug(
                "Device set on automatic mode. CUDA is not available. Running via cpu"
            )

    elif device == "cpu":
        set_("num_workers", 0)
        logger.debug("Device manually set to cpu")

    elif device == "cuda:0" and torch.cuda.is_available():
        logger.debug("Device manually set to gpu")

    elif device == "cuda:0" and not torch.cuda.is_available():
        raise ValueError("Device manually set to gpu. CUDA not available")

    elif device not in ["cpu", "cuda:0", "automatic"]:
        raise ValueError('Invalid device. Set device to "cpu", "cuda:0", "automatic"')


def initialize() -> None:
    """
    Read command line and store user settings.
    """

    parser = build_arg_parser()
    par = parser.parse_known_args()[0]

    # Main arguments.
    set_("run_mode", par.run_mode)
    set_("colonisation_type", par.colonisation_type)
    convert_tiff = False
    if par.run_mode == "tifconversion":
        convert_tiff = True

    files = find_files_in_directory(par.images, convert_tiff)
    set_("input_files", files)

    set_("use_db", par.use_db)

    if par.run_mode == "train":
        if par.train_active_learning:
            set_("learning_rate_active_learning", par.learning_rate_active_learning)
            set_("epochs_active_learning", par.epochs_active_learning)
        else:
            set_("learning_rate", par.learning_rate)
            set_("epochs", par.epochs)

        set_("mlflow_flag", par.mlflow_flag)
        set_("batch_size", par.batch_size)
        set_("adam_beta1", par.adam_beta1)
        set_("adam_beta2", par.adam_beta2)
        set_("balance_factor", par.balance_factor)
        set_("model", par.model, use_none=True)
        set_("model_erm", par.model_erm, use_none=True)
        set_("model_type", par.model_type)
        set_("pre_trained", par.pre_trained)
        set_("vfrac", par.vfrac)
        set_("data_augm", par.data_augm)
        set_("summary", par.summary)
        set_("outdir", par.outdir)
        set_("patience_e", par.patience_e)
        set_("patience_r", par.patience_r)

        # if an outdir has not been specified, create one in the default loc
        if par.outdir is None:
            results_dir = create_results_dir("train")
            set_("outdir", results_dir)

        set_(
            "get_tiles_for_labelling_using_active_learning",
            par.get_tiles_for_labelling_using_active_learning,
        )
        set_("active_learning_method", par.active_learning_method)
        set_("num_samples_for_labelling", par.num_samples_for_labelling)
        set_("mc_samples", par.mc_samples)
        set_("dropout_rate", par.dropout_rate)
        set_("tile_edge", par.edge)

    elif par.run_mode == "predict":
        set_("tile_edge", par.edge)
        set_("model", par.model)
        set_("model_erm", par.model_erm)
        set_("temperature_factor_path", par.temperature_factor_path)
        set_("temperature_factor_path_erm", par.temperature_factor_path_erm)
        set_("use_contextual_confidence", par.use_contextual_confidence)
        set_("contextual_confidence_threshold", par.contextual_confidence_threshold)
        set_("outdir", par.outdir)

        if par.outdir is None:
            set_("outdir", clean_path(par.images))

    elif par.run_mode == "test":
        set_("model", par.model)
        set_("model_erm", par.model_erm)
        set_("outdir", par.outdir)
        set_("tile_edge", par.edge)
        set_("temperature_factor_path", par.temperature_factor_path)
        set_("temperature_factor_path_erm", par.temperature_factor_path_erm)
        set_("use_contextual_confidence", par.use_contextual_confidence)
        set_("contextual_confidence_threshold", par.contextual_confidence_threshold)

        # if an outdir has not been specified, create one in the default loc
        if par.outdir is None:
            results_dir = create_results_dir("test")
            set_("outdir", results_dir)

    elif par.run_mode == "colonisation":
        set_("outdir", par.outdir)
        set_("tile_edge", par.edge)

        # if an outdir has not been specified, create one in the default loc
        if par.outdir is None:
            results_dir = create_results_dir("colonisation_results")
            set_("outdir", results_dir)

    elif par.run_mode == "convert":
        set_("threshold", par.threshold)
        set_("aggregate_tiles", par.aggregate_tiles)
        set_("tile_edge", par.edge)
        set_("use_contextual_confidence", par.use_contextual_confidence)

    elif par.run_mode == "tifconversion":
        set_("convert_image_file_type", par.convert_image_file_type)
        set_("tile_edge", par.edge)

    elif par.run_mode == "calibrate":
        set_("model", par.model)
        set_("model_erm", par.model_erm)
        set_("tile_edge", par.edge)
        set_("outdir", par.outdir)

        if par.outdir is None:
            results_dir = create_results_dir("calibrate")
            set_("outdir", results_dir)

    else:
        pass
