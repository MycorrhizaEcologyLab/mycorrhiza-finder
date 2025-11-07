# AMFinder - amfinder_config.py
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

import torch

import amfinder_log as AmfLog
from api_objects import (
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


def get_default_output_dir():
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


def create_results_dir(label: str = ""):
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
    print(f"Output directory created: {results_dir_path}")
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
    # Carry out semi-sup training
    "semi_supervised": False,
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
    # Path to use for semi-supervised training
    "root_path": "",
    "fixmatch_results_directory": "",
    "num_workers": 0,
    "use_contextual_confidence": True,  # Enable contextual confidence refinement
    "contextual_confidence_threshold": 1,  # Threshold for applying max voting
    # Config to collapse classes
    "collapse": False,
}


APP_PATH = os.path.dirname(os.path.realpath(__file__))


def get_appdir():
    """Returns the application directory."""

    return APP_PATH


def invite():
    """
    Command-line invite
    """
    return datetime.datetime.now().strftime("%H:%M:%S")


def human_readable_header():
    """
    Return the human-readable header of the current model.
    """

    return HUMAN_HEADERS[PAR["colonisation_type"]]


def get(id):
    """
    Retrieve application settings.

    :param id: Unique identifier.
    """

    id = id.lower()

    if id in PAR:
        # Special case, look into a specific folder.
        if id in ["model", "model_erm"] and PAR[id] is not None:
            # Check if model exists in trained networks, else use absolute path
            model_name = os.path.basename(PAR[id])
            path = os.path.join(get_appdir(), "trained_networks", model_name)

            if not os.path.isfile(path):
                path = PAR[id]

            return path

        else:
            return PAR[id]

    elif id in PAR["monitors"]:
        return PAR["monitors"][id]

    else:
        AmfLog.warning(f"Unknown parameter {id}")
        return None


def find_files_in_directory(directory, convert_tiff=False):
    # Search for .jpg, .jpeg & .png files
    search_patterns = [
        os.path.join(directory, "**", "*.jpg"),
        os.path.join(directory, "**", "*.jpeg"),
        os.path.join(directory, "**", "*.png"),
    ]

    if convert_tiff:
        AmfLog.info("Also searching for tif files")
        search_patterns.append(os.path.join(directory, "**", "*.tif"))
        search_patterns.append(os.path.join(directory, "**", "*.tiff"))

    img_files = []
    for pattern in search_patterns:
        img_files.extend(glob.glob(pattern, recursive=True))

    if len(img_files) == 0:
        raise FileNotFoundError(f"No image files found in {directory}")

    return img_files


def set(id, value, create=False):
    """
    Updates application settings.

    :param id: unique identifier.
    :param value: value to store.
    :param create: create id if it does not exist (optional).
    """

    if value is None:
        return

    else:
        id = id.lower()

        if id in PAR:
            PAR[id] = value

            if id == "colonisation_type":
                PAR["header"] = HEADERS[value]

            elif id == "collapse":
                if value:
                    if PAR["colonisation_type"] == "am":
                        PAR["header"] = [
                            "AMColonised",
                            "Uncolonised",
                            "Background",
                            "Unreadable",
                            "DSE",
                        ]
                    else:
                        PAR["header"] = [
                            "ErMColonised",
                            "Uncolonised",
                            "Background",
                            "MainRoot",
                            "Unreadable",
                            "DSE",
                            "HybridDse",
                        ]

                    PAR["class_names"] = {
                        "am": [
                            "AMColonised",
                            "Uncolonised",
                            "Background",
                            "Unreadable",
                            "DSE",
                        ],
                        "erm": [
                            "ErMColonised",
                            "Uncolonised",
                            "Background",
                            "MainRoot",
                            "Unreadable",
                            "DSE",
                            "HybridDse",
                        ],
                    }

        elif id in PAR["monitors"]:
            PAR["monitors"][id] = value

        elif create:
            PAR[id] = value

        else:
            AmfLog.warning(f"Unknown parameter {id}")


def add_training_subparser(subparsers):
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

    x = PAR["semi_supervised"]
    parser.add_argument(
        "-smi",
        "--semi_supervised",
        action="store_const",
        dest="semi_supervised",
        const=True,
        help="Enables semi-supervised training.",
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

    x = PAR["pre_trained"]
    parser.add_argument(
        "-pretr",
        "--pretrain",
        action="store_const",
        dest="pre_trained",
        const=True,
        help=(
            "Loads ImageNet weights if ResNet, ResNeXt or EfficentNet is selected for "
            "Model type."
        ),
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

    parser.add_argument(
        "-c",
        "--collapse",
        action="store_const",
        dest="collapse",
        const=True,
        help="Argument to collapse extra classes down",
    )

    return


def add_test_subparser(subparsers):
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

    x = PAR["semi_supervised"]
    parser.add_argument(
        "-smi",
        "--semi_supervised",
        action="store_const",
        dest="semi_supervised",
        const=True,
        help="Enables semi-supervised training.",
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

    x = PAR["fixmatch_results_directory"]
    parser.add_argument(
        "-f",
        "--fixmatch_results_directory",
        action="store",
        dest="fixmatch_results_directory",
        default=x,
        help="folder where fixmatch results are stored.\ndefault: {}".format(x),
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

    parser.add_argument(
        "-c",
        "--collapse",
        action="store_const",
        dest="collapse",
        const=True,
        help="Argument to collapse extra classes down",
    )

    return


def add_colonisation_subparser(subparsers):
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


def add_prediction_subparser(subparsers):
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


def add_conversion_subparser(subparsers):
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

    parser.add_argument(
        "-c",
        "--collapse",
        action="store_const",
        dest="collapse",
        const=True,
        help="Argument to collapse extra classes down",
    )

    return


def add_tif_conversion_subparser(subparsers):
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


def add_calibrate_subparser(subparsers):
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


def build_arg_parser():
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


def abspath(files):
    """
    Returns absolute paths to input files.

    :param files: Raw list of input file names (can contain wildcards).
    """
    files = sum([glob.glob(x) for x in files], [])
    return [os.path.abspath(x) for x in files]


def get_input_files():
    """
    Filter input file list and keep valid JPEG or TIFF images.
    """

    raw_list = abspath(get("input_files"))

    valid_types = ["image/jpeg", "image/tiff", "image/png"]
    images = [x for x in raw_list if mimetypes.guess_type(x)[0] in valid_types]
    AmfLog.text(f"Input images: {len(images)}")
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


def set_train_config(trainConfig: TrainConfig):
    """
    Set configuration for the training run.

    Args:
        trainConfig (TrainConfig): Configuration object containing training parameters.

    Sets various parameters such as input files, model details, batch size,
    number of epochs, and output directory for the training process.
    """
    set("run_mode", "train")
    set_device(trainConfig.device)

    set("semi_supervised", trainConfig.semiSupervised)
    set("train_active_learning", trainConfig.trainActiveLearning)
    set(
        "get_tiles_for_labelling_using_active_learning",
        trainConfig.getTilesForLabellingUsingActiveLearning,
    )
    set("active_learning_method", trainConfig.activeLearningMethod)
    set("num_samples_for_labelling", trainConfig.numSamplesForLabelling)
    set("mc_samples", trainConfig.mcSamples)
    set("dropout_rate", trainConfig.dropoutRate)
    if trainConfig.semiSupervised:
        set("root_path", trainConfig.inputFiles)
    else:
        files = find_files_in_directory(trainConfig.inputFiles)
        set("input_files", files)

    set("use_db", trainConfig.useDb)
    set("mlflow_flag", trainConfig.mlflowFlag)
    set("colonisation_type", trainConfig.colonisationType)
    set("batch_size", trainConfig.batchSize)
    set("adam_beta1", trainConfig.adamBeta1)
    set("adam_beta2", trainConfig.adamBeta2)
    set("balance_factor", trainConfig.balanceFactor)
    # Override learning rate and epochs if train_active_learning is enabled
    if trainConfig.trainActiveLearning:
        set("learning_rate_active_learning", trainConfig.learningRateActiveLearning)
        set("epochs_active_learning", trainConfig.epochsActiveLearning)
        AmfLog.info("Training mode after active learning")
    else:
        set("learning_rate", trainConfig.learningRate)
        set("epochs", trainConfig.epochs)
    set("model", trainConfig.model)
    set("model_erm", trainConfig.modelErm)
    set("model_type", trainConfig.modelType)
    set("pre_trained", trainConfig.preTrained)
    set("vfrac", trainConfig.vfrac)
    set("data_augm", trainConfig.dataAugm)
    set("summary", trainConfig.summary)
    set("patience_e", trainConfig.patienceE)
    set("patience_r", trainConfig.patienceR)
    set("tile_edge", trainConfig.tileEdge)

    if trainConfig.outdir is None or trainConfig.outdir == "":
        results_dir = create_results_dir("train")
        set("outdir", results_dir)
    else:
        set("outdir", trainConfig.outdir)


def set_predict_config(predictionConfig: PredictionConfig):
    """
    Set configuration for the prediction run.

    Args:
        predictionConfig (PredictionConfig): Configuration object containing prediction
            parameters.

    This function sets parameters required for making predictions, such as
    input files, model, and output directory.
    """
    set("run_mode", "predict")
    set_device(predictionConfig.device)
    files = find_files_in_directory(predictionConfig.inputFiles)
    set("input_files", files)
    set("use_db", predictionConfig.useDb)
    set("colonisation_type", predictionConfig.colonisationType)
    set("tile_edge", predictionConfig.tileEdge)
    set("model", predictionConfig.model)
    set("model_erm", predictionConfig.modelErm)
    set("temperature_factor_path", predictionConfig.temperatureFactorPath)
    set("temperature_factor_path_erm", predictionConfig.temperatureFactorPathErm)

    if predictionConfig.outdir is None or predictionConfig.outdir == "":
        set("outdir", clean_path(predictionConfig.inputFiles))
    else:
        set("outdir", predictionConfig.outdir)

    set("use_contextual_confidence", predictionConfig.useContextualConfidence)
    set(
        "contextual_confidence_threshold",
        predictionConfig.contextualConfidenceThreshold,
    )


def set_test_config(testConfig: TestConfig):
    """
    Set configuration for the testing run.

    Args:
        testConfig (TestConfig): Configuration object containing testing parameters.

    This function configures the settings needed for the testing procedure,
    including input files, model, and output directory for results.
    """
    set("run_mode", "test")
    set_device(testConfig.device)
    set("semi_supervised", testConfig.semiSupervised)
    if testConfig.semiSupervised:
        set("root_path", testConfig.inputFiles)
    else:
        files = find_files_in_directory(testConfig.inputFiles)
        set("input_files", files)

    set("use_db", testConfig.useDb)
    set("colonisation_type", testConfig.colonisationType)
    set("model", testConfig.model)
    set("model_erm", testConfig.modelErm)
    set("tile_edge", testConfig.tileEdge)
    set("fixmatch_results_directory", testConfig.fixmatchResultsDirectory)
    set("temperature_factor_path", testConfig.temperatureFactorPath)
    set("temperature_factor_path_erm", testConfig.temperatureFactorPathErm)

    if (
        testConfig.outdir is None or testConfig.outdir == ""
    ) and not testConfig.semiSupervised:
        results_dir = create_results_dir("test")
        set("outdir", results_dir)
    else:
        set("outdir", testConfig.outdir)

    set("use_contextual_confidence", testConfig.useContextualConfidence)
    set("contextual_confidence_threshold", testConfig.contextualConfidenceThreshold)


def set_colonisation_config(colonisationConfig: BaseConfig):
    """
    Set configuration for the testing run.

    Args:
        testConfig (TestConfig): Configuration object containing testing parameters.

    This function configures the settings needed for the testing procedure,
    including input files, model, and output directory for results.
    """
    set("run_mode", "colonisation")
    set_device(colonisationConfig.device)
    files = find_files_in_directory(colonisationConfig.inputFiles)
    set("input_files", files)

    set("use_db", colonisationConfig.useDb)
    set("colonisation_type", colonisationConfig.colonisationType)
    set("tile_edge", colonisationConfig.tileEdge)

    if colonisationConfig.outdir is None or colonisationConfig.outdir == "":
        results_dir = create_results_dir("colonisation_results")
        set("outdir", results_dir)
    else:
        set("outdir", colonisationConfig.outdir)


def set_convert_config(convertConfig: ConvertConfig):
    """
    Set configuration for the conversion run.

    Args:
        convertConfig (ConvertConfig): Configuration object containing conversion
            parameters.

    This function defines the necessary settings for annotations conversion,
    including input files, colonisation type, threshold, and output directory.
    """
    set("run_mode", "convert")
    set_device(convertConfig.device)
    files = find_files_in_directory(convertConfig.inputFiles)
    set("input_files", files)
    set("use_db", convertConfig.useDb)
    set("aggregate_tiles", convertConfig.aggregateTiles)
    set("colonisation_type", convertConfig.colonisationType)
    set("threshold", convertConfig.threshold)
    set("tile_edge", convertConfig.tileEdge)
    set("use_contextual_confidence", convertConfig.useContextualConfidence)

    if convertConfig.outdir is None or convertConfig.outdir == "":
        set("outdir", clean_path(convertConfig.inputFiles))
    else:
        set("outdir", convertConfig.outdir)


def set_tif_conversion_config(tifConversionConfig: TifConversionConfig):
    """
    Set configuration for the tif conversion.

    Args:
        tifConversionConfig (TifConversionConfig): Configuration object containing TIF
            conversion parameters.

    This function defines the necessary settings for tif file conversion.
    """
    set("run_mode", "tifconversion")
    set_device(tifConversionConfig.device)
    files = find_files_in_directory(tifConversionConfig.inputFiles, convert_tiff=True)
    set("input_files", files)
    set("use_db", tifConversionConfig.useDb)
    set("colonisation_type", tifConversionConfig.colonisationType)
    set("convert_image_file_type", tifConversionConfig.convertImageFileType)
    set("tile_edge", tifConversionConfig.tileEdge)

    if tifConversionConfig.outdir is None or tifConversionConfig.outdir == "":
        set("outdir", clean_path(tifConversionConfig.inputFiles))
    else:
        set("outdir", tifConversionConfig.outdir)


def set_calibrate_config(calibrateConfig: CalibrateConfig):
    """
    Set configuration for the calibration.

    Args:
        calibrateConfig (CalibrateConfig): Configuration object containing calibration
        parameters.

    This function sets parameters necessary for calibration models, including
    input files, model, and output directory for results.
    """
    set("run_mode", "calibration")
    set_device(calibrateConfig.device)
    files = find_files_in_directory(calibrateConfig.inputFiles)
    set("input_files", files)
    set("use_db", calibrateConfig.useDb)
    set("colonisation_type", calibrateConfig.colonisationType)
    set("model", calibrateConfig.model)
    set("model_erm", calibrateConfig.modelErm)
    set("tile_edge", calibrateConfig.tileEdge)

    if calibrateConfig.outdir is None or calibrateConfig.outdir == "":
        results_dir = create_results_dir("calibrate")
        set("outdir", results_dir)
    else:
        set("outdir", calibrateConfig.outdir)


def set_device(device):
    # Get the device setting
    if device == "automatic":
        if torch.cuda.is_available():
            set("device", "cuda:0")
            AmfLog.text("Device set on automatic mode. Running via gpu")

        else:
            set("num_workers", 0)
            set("device", "cpu")
            AmfLog.text(
                "Device set on automatic mode. CUDA is not available. Running via cpu"
            )

    elif device == "cpu":
        set("num_workers", 0)
        AmfLog.text("Device manually set to cpu")

    elif device == "cuda:0" and torch.cuda.is_available():
        AmfLog.text("Device manually set to gpu")

    elif device == "cuda:0" and not torch.cuda.is_available():
        raise ValueError("Device manually set to gpu. CUDA not available")

    elif device not in ["cpu", "cuda:0", "automatic"]:
        raise ValueError('Invalid device. Set device to "cpu", "cuda:0", "automatic"')


def initialize():
    """
    Read command line and store user settings.
    """

    parser = build_arg_parser()
    par = parser.parse_known_args()[0]

    # Main arguments.
    set("run_mode", par.run_mode)
    set("colonisation_type", par.colonisation_type)
    convert_tiff = False
    if par.run_mode == "tifconversion":
        convert_tiff = True

    semi_supervised = False
    if par.run_mode == "test" or par.run_mode == "train":
        semi_supervised = par.semi_supervised
        set("semi_supervised", semi_supervised)
        if semi_supervised:
            set("root_path", par.images)

    if not semi_supervised:
        files = find_files_in_directory(par.images, convert_tiff)
        set("input_files", files)

    set("use_db", par.use_db)

    if par.run_mode == "train":
        if par.train_active_learning:
            set("learning_rate_active_learning", par.learning_rate_active_learning)
            set("epochs_active_learning", par.epochs_active_learning)
        else:
            set("learning_rate", par.learning_rate)
            set("epochs", par.epochs)

        set("mlflow_flag", par.mlflow_flag)
        set("batch_size", par.batch_size)
        set("adam_beta1", par.adam_beta1)
        set("adam_beta2", par.adam_beta2)
        set("balance_factor", par.balance_factor)
        set("model", par.model)
        set("model_erm", par.model_erm)
        set("model_type", par.model_type)
        set("pre_trained", par.pre_trained)
        set("vfrac", par.vfrac)
        set("data_augm", par.data_augm)
        set("summary", par.summary)
        set("outdir", par.outdir)
        set("patience_e", par.patience_e)
        set("patience_r", par.patience_r)

        # if an outdir has not been specified, create one in the default loc
        if par.outdir is None:
            results_dir = create_results_dir("train")
            set("outdir", results_dir)

        set(
            "get_tiles_for_labelling_using_active_learning",
            par.get_tiles_for_labelling_using_active_learning,
        )
        set("active_learning_method", par.active_learning_method)
        set("num_samples_for_labelling", par.num_samples_for_labelling)
        set("mc_samples", par.mc_samples)
        set("dropout_rate", par.dropout_rate)
        set("tile_edge", par.edge)

        set("collapse", par.collapse)

    elif par.run_mode == "predict":
        set("tile_edge", par.edge)
        set("model", par.model)
        set("model_erm", par.model_erm)
        set("temperature_factor_path", par.temperature_factor_path)
        set("temperature_factor_path_erm", par.temperature_factor_path_erm)
        set("use_contextual_confidence", par.use_contextual_confidence)
        set("contextual_confidence_threshold", par.contextual_confidence_threshold)
        set("outdir", par.outdir)

        set("collapse", par.collapse)

        if par.outdir is None:
            set("outdir", clean_path(par.images))

    elif par.run_mode == "test":
        set("model", par.model)
        set("model_erm", par.model_erm)
        set("outdir", par.outdir)
        set("tile_edge", par.edge)
        set("fixmatch_results_directory", par.fixmatch_results_directory)
        set("temperature_factor_path", par.temperature_factor_path)
        set("temperature_factor_path_erm", par.temperature_factor_path_erm)
        set("use_contextual_confidence", par.use_contextual_confidence)
        set("contextual_confidence_threshold", par.contextual_confidence_threshold)

        set("collapse", par.collapse)

        # if an outdir has not been specified, create one in the default loc
        if par.outdir is None and not semi_supervised:
            results_dir = create_results_dir("test")
            set("outdir", results_dir)

    elif par.run_mode == "colonisation":
        set("outdir", par.outdir)
        set("tile_edge", par.edge)

        # if an outdir has not been specified, create one in the default loc
        if par.outdir is None:
            results_dir = create_results_dir("colonisation_results")
            set("outdir", results_dir)

    elif par.run_mode == "convert":
        set("threshold", par.threshold)
        set("aggregate_tiles", par.aggregate_tiles)
        set("tile_edge", par.edge)
        set("use_contextual_confidence", par.use_contextual_confidence)
        set("collapse", par.collapse)

    elif par.run_mode == "tifconversion":
        set("convert_image_file_type", par.convert_image_file_type)
        set("tile_edge", par.edge)

    elif par.run_mode == "calibrate":
        set("model", par.model)
        set("model_erm", par.model_erm)
        set("tile_edge", par.edge)
        set("outdir", par.outdir)

        if par.outdir is None:
            results_dir = create_results_dir("calibrate")
            set("outdir", results_dir)

    else:
        pass
