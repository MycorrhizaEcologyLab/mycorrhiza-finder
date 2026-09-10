# AMFinder - settings_schema.py
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
Canonical registry of every user-editable AMFinder setting.

This is the single source of truth the frontend Settings page renders from
(via GET /settings-schema) and that the request-body builders for each
AmfTool action filter against. Each entry's `default` is read live from
`amf.helper.config.PAR`, so schema defaults can never drift from the
argparse/PAR defaults in config.py.

Session/derived state that isn't a persisted user preference is deliberately
excluded: run_mode, header, class_names, monitors, tiles, input_files,
colonisation_type and tile_edge (the latter two are driven by the currently
loaded image via GlobalContextProvider, not this schema).
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import amf.helper.config as AmfConfig
from amf.helper.config import PAR

RunMode = str  # "train" | "predict" | "test" | "convert" | "tifconversion" | "calibrate" | "colonisation"


@dataclass(frozen=True)
class SettingSpec:
    key: str  # camelCase, matches the API/DB field name, e.g. "ciMethod"
    parKey: str  # snake_case PAR/argparse dest, e.g. "ci_method"
    label: str
    help: str
    type: str  # "string" | "integer" | "float" | "boolean" | "select"
    group: str
    advanced: bool
    modes: list[RunMode]
    choices: Optional[list[dict[str, Any]]] = field(default=None)

    @property
    def default(self) -> Any:
        return PAR[self.parKey]

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "help": self.help,
            "type": self.type,
            "choices": self.choices,
            "default": self.default,
            "group": self.group,
            "advanced": self.advanced,
            "modes": self.modes,
        }


_ALL_MODES: list[RunMode] = [
    "train",
    "predict",
    "test",
    "convert",
    "tifconversion",
    "calibrate",
    "colonisation",
]

SETTINGS_SCHEMA: list[SettingSpec] = [
    # General
    SettingSpec(
        "outdir",
        "outdir",
        "Output Directory",
        "Folder where results, trained models and CNN architecture summaries are saved. Defaults to the input images' folder if left blank.",
        "string",
        "general",
        False,
        _ALL_MODES,
    ),
    SettingSpec(
        "useDb",
        "use_db",
        "Use Local Database",
        "Store and read annotations/predictions in the local database instead of CSV files.",
        "boolean",
        "general",
        False,
        _ALL_MODES,
    ),
    SettingSpec(
        "imageDirectory",
        "image_directory",
        "Image Directory",
        "Folder containing the images you browse/annotate. When Use Local Database is off, this (together with Output Directory) is searched for existing local prediction/annotation CSVs for the image you select in the Browser tab.",
        "string",
        "general",
        False,
        [],  # not consumed by any predict/train/etc action body
    ),
    SettingSpec(
        "device",
        "device",
        "Device",
        "Hardware used to run the model.",
        "select",
        "general",
        False,
        _ALL_MODES,
        choices=[
            {"value": "automatic", "label": "Automatic"},
            {"value": "cpu", "label": "CPU"},
            {"value": "cuda:0", "label": "CUDA (GPU)"},
        ],
    ),
    SettingSpec(
        "numWorkers",
        "num_workers",
        "Number of Workers",
        "Number of worker processes used for data loading. Forced to 0 automatically when running on CPU.",
        "integer",
        "general",
        True,
        _ALL_MODES,
    ),
    SettingSpec(
        "model",
        "model",
        "Model for AM",
        "Name of the pre-trained model to use for Arbuscular Mycorrhizal (AM) predictions. Lists .pth files found in the trained networks folder.",
        "modelSelect",
        "general",
        False,
        ["train", "predict", "test", "calibrate", "colonisation"],
    ),
    SettingSpec(
        "modelErm",
        "model_erm",
        "Model for ErM",
        "Name of the pre-trained model to use for Ericoid Mycorrhizal (ErM) predictions. Lists .pth files found in the trained networks folder.",
        "modelSelect",
        "general",
        False,
        ["train", "predict", "test", "calibrate", "colonisation"],
    ),
    SettingSpec(
        "modelPath",
        "model_path",
        "Model Path Override",
        "Explicit path to a model file stored outside the trained networks folder (e.g. an absolute path elsewhere on disk). If not set, falls back to the Model for AM/ErM setting above.",
        "string",
        "general",
        True,
        ["predict", "test", "calibrate"],
    ),
    SettingSpec(
        "resizeDim",
        "resize_dim",
        "Resize Dimension",
        "Dimension to resize input images to for model compatibility (e.g. for transformer-based models). Leave blank to disable.",
        "integer",
        "general",
        True,
        ["train", "predict", "test", "calibrate", "colonisation"],
    ),
    SettingSpec(
        "temperatureFactorPath",
        "temperature_factor_path",
        "Temperature Factor File (AM)",
        "Name of the file containing the calibrated temperature factor for the AM model.",
        "string",
        "general",
        True,
        ["predict", "test"],
    ),
    SettingSpec(
        "temperatureFactorPathErm",
        "temperature_factor_path_erm",
        "Temperature Factor File (ErM)",
        "Name of the file containing the calibrated temperature factor for the ErM model.",
        "string",
        "general",
        True,
        ["predict", "test"],
    ),
    SettingSpec(
        "useContextualConfidence",
        "use_contextual_confidence",
        "Use Contextual Confidence",
        "Enable contextual confidence refinement (max voting with surrounding tiles) for low-confidence predictions.",
        "boolean",
        "general",
        False,
        ["predict", "test", "convert"],
    ),
    SettingSpec(
        "contextualConfidenceThreshold",
        "contextual_confidence_threshold",
        "Contextual Confidence Threshold",
        "Confidence threshold below which max voting with surrounding tiles is applied.",
        "float",
        "general",
        True,
        ["predict", "test"],
    ),
    # Predict
    SettingSpec(
        "ciMethod",
        "ci_method",
        "Confidence Interval Method",
        "Method used to calculate prediction confidence intervals: 'analytic' (fast, near-identical results), 'bootstrap' (slower resampling approach) or 'both' (compute and compare both).",
        "select",
        "predict",
        True,
        ["predict"],
        choices=[
            {"value": "analytic", "label": "Analytic"},
            {"value": "bootstrap", "label": "Bootstrap"},
            {"value": "both", "label": "Both"},
        ],
    ),
    # Train
    SettingSpec(
        "modelType",
        "model_type",
        "Model Type",
        "Architecture to initialise when training a new model from scratch. Options besides CNN1 come from timm, per MODEL_DICT in model.py.",
        "select",
        "train",
        False,
        ["train"],
        # Mirrors amf.helper.model.MODEL_DICT (plus the "cnn1" custom model,
        # which isn't in that dict). "isTransformer" flags model types whose
        # timm backbone expects a fixed input resolution (currently only
        # deit3, at 224x224) - the frontend uses it to auto-manage
        # Resize Dimension when this setting changes.
        choices=[
            {"value": "cnn1", "label": "CNN1"},
            {"value": "resnet", "label": "ResNet"},
            {"value": "resnext", "label": "ResNext"},
            {"value": "efficientnet", "label": "EfficientNet"},
            {"value": "efficientnetv2", "label": "EfficientNetV2"},
            {"value": "convnext", "label": "ConvNext"},
            {
                "value": "deit3",
                "label": "DeiT3 (Transformer)",
                "isTransformer": True,
                "transformerResizeDim": 224,
            },
        ],
    ),
    SettingSpec(
        "preTrained",
        "pre_trained",
        "Use Pre-Trained Weights",
        "Load ImageNet weights when initialising a ResNet, ResNeXt or EfficientNet model.",
        "boolean",
        "train",
        True,
        ["train"],
    ),
    SettingSpec(
        "trainActiveLearning",
        "train_active_learning",
        "Train After Active Learning",
        "Use a different (typically lower) learning rate and epoch count, intended for continuing training after active-learning samples have been labelled.",
        "boolean",
        "train",
        False,
        ["train"],
    ),
    SettingSpec(
        "filterBackground",
        "filter_background",
        "Filter Background Tiles",
        "Exclude background tiles from the training set.",
        "boolean",
        "train",
        True,
        ["train"],
    ),
    SettingSpec(
        "mlFlowFlag",
        "mlflow_flag",
        "Enable MLflow Tracking",
        "Track this training run with MLflow.",
        "boolean",
        "train",
        True,
        ["train"],
    ),
    SettingSpec(
        "batchSize",
        "batch_size",
        "Batch Size",
        "Number of tiles per training batch.",
        "integer",
        "train",
        False,
        ["train"],
    ),
    SettingSpec(
        "dataAugm",
        "data_augm",
        "Data Augmentation",
        "Apply data augmentation (hue, chroma, saturation, etc.) during training.",
        "boolean",
        "train",
        False,
        ["train"],
    ),
    SettingSpec(
        "summary",
        "summary",
        "Save Model Summary",
        "Save the CNN architecture graph and model summary after training.",
        "boolean",
        "train",
        True,
        ["train"],
    ),
    SettingSpec(
        "epochs",
        "epochs",
        "Epochs",
        "Number of epochs to train for.",
        "integer",
        "train",
        False,
        ["train"],
    ),
    SettingSpec(
        "epochsActiveLearning",
        "epochs_active_learning",
        "Epochs After Active Learning",
        "Number of epochs to train for once active-learning samples have been labelled.",
        "integer",
        "train",
        True,
        ["train"],
    ),
    SettingSpec(
        "patienceE",
        "patience_e",
        "Early Stopping Patience",
        "Number of epochs with no improvement to wait before early stopping is triggered.",
        "integer",
        "train",
        True,
        ["train"],
    ),
    SettingSpec(
        "patienceR",
        "patience_r",
        "Learning Rate Reduction Patience",
        "Number of epochs with no improvement to wait before the learning rate is reduced.",
        "integer",
        "train",
        True,
        ["train"],
    ),
    SettingSpec(
        "learningRate",
        "learning_rate",
        "Learning Rate",
        "Learning rate used by the Adam optimizer.",
        "float",
        "train",
        False,
        ["train"],
    ),
    SettingSpec(
        "learningRateActiveLearning",
        "learning_rate_active_learning",
        "Learning Rate After Active Learning",
        "Learning rate used by the Adam optimizer once active-learning samples have been labelled.",
        "float",
        "train",
        True,
        ["train"],
    ),
    SettingSpec(
        "adamBeta1",
        "adam_beta1",
        "Adam Beta 1",
        "Beta 1 hyperparameter for the Adam optimizer.",
        "float",
        "train",
        True,
        ["train"],
    ),
    SettingSpec(
        "adamBeta2",
        "adam_beta2",
        "Adam Beta 2",
        "Beta 2 hyperparameter for the Adam optimizer.",
        "float",
        "train",
        True,
        ["train"],
    ),
    SettingSpec(
        "balanceFactor",
        "balance_factor",
        "Balance Factor",
        "Multiplier for balancing datasets across class sizes, oriented on colonised classes.",
        "float",
        "train",
        True,
        ["train"],
    ),
    SettingSpec(
        "vfrac",
        "vfrac",
        "Validation Fraction",
        "Proportion of tiles held out for validation.",
        "float",
        "train",
        False,
        ["train"],
    ),
    SettingSpec(
        "dropoutRate",
        "dropout_rate",
        "Dropout Rate",
        "Dropout probability used by the model, including for MC-dropout uncertainty estimation in active learning.",
        "float",
        "train",
        True,
        ["train"],
    ),
    SettingSpec(
        "dropPathRate",
        "drop_path_rate",
        "Drop Path Rate",
        "Drop-path probability for timm-based models.",
        "float",
        "train",
        True,
        ["train"],
    ),
    SettingSpec(
        "dynamicLoading",
        "dynamic_loading",
        "Dynamic Tile Loading",
        "Load tile images dynamically during training/calibration instead of all at once, reducing memory usage.",
        "boolean",
        "train",
        True,
        ["train", "calibrate"],
    ),
    SettingSpec(
        "pretiledDir",
        "pretiled_dir",
        "Pre-Tiled Images Directory",
        "Directory containing pre-tiled images, used together with dynamic tile loading.",
        "string",
        "train",
        True,
        ["train", "calibrate"],
    ),
    SettingSpec(
        "checkpointPath",
        "checkpoint_path",
        "Checkpoint Path",
        "Path to a checkpoint to load timm model weights from. If not set, weights are initialised from the Model setting and Use Pre-Trained Weights.",
        "string",
        "train",
        True,
        ["train"],
    ),
    SettingSpec(
        "weightDecay",
        "weight_decay",
        "Weight Decay",
        "Weight decay applied by the Adam optimizer.",
        "float",
        "train",
        True,
        ["train"],
    ),
    SettingSpec(
        "backboneLrMult",
        "backbone_lr_mult",
        "Backbone LR Multiplier",
        "Learning rate multiplier applied to backbone layers during training.",
        "float",
        "train",
        True,
        ["train"],
    ),
    SettingSpec(
        "freezeEpochs",
        "freeze_epochs",
        "Freeze Epochs",
        "Number of epochs to keep backbone layers frozen at the start of training.",
        "integer",
        "train",
        True,
        ["train"],
    ),
    SettingSpec(
        "earlyBreakEpoch",
        "early_break_epoch",
        "Early Break Epoch",
        "Epoch at which to deliberately stop training early. Leave blank to disable.",
        "integer",
        "train",
        True,
        ["train"],
    ),
    # Active Learning
    SettingSpec(
        "getTilesForLabellingUsingActiveLearning",
        "get_tiles_for_labelling_using_active_learning",
        "Output Tiles For Labelling",
        "Run active learning to select tiles that should be labelled next, instead of training.",
        "boolean",
        "activeLearning",
        False,
        ["train"],
    ),
    SettingSpec(
        "activeLearningMethod",
        "active_learning_method",
        "Active Learning Method",
        "Acquisition function used to select tiles for labelling.",
        "select",
        "activeLearning",
        False,
        ["train"],
        choices=[
            {"value": "bald", "label": "BALD"},
            {"value": "batchbald", "label": "BatchBALD"},
        ],
    ),
    SettingSpec(
        "numSamplesForLabelling",
        "num_samples_for_labelling",
        "Samples For Labelling",
        "Number of samples to select per file for labelling.",
        "integer",
        "activeLearning",
        False,
        ["train"],
    ),
    SettingSpec(
        "mcSamples",
        "mc_samples",
        "Monte Carlo Samples",
        "Number of Monte Carlo dropout samples used for uncertainty estimation.",
        "integer",
        "activeLearning",
        True,
        ["train"],
    ),
    # Convert
    SettingSpec(
        "threshold",
        "threshold",
        "Threshold",
        "Only convert predictions to annotations when above this confidence threshold.",
        "float",
        "convert",
        False,
        ["convert"],
    ),
    SettingSpec(
        "aggregateTiles",
        "aggregate_tiles",
        "Aggregate Tiles",
        "Upscale tiles by a factor of 2 during conversion.",
        "boolean",
        "convert",
        False,
        ["convert"],
    ),
    # TIF Conversion
    SettingSpec(
        "convertImageFileType",
        "convert_image_file_type",
        "TIF Conversion File Type",
        "Image format to convert TIF files to.",
        "select",
        "tifconversion",
        False,
        ["tifconversion"],
        choices=[
            {"value": "jpg", "label": "JPG"},
            {"value": "png", "label": "PNG"},
        ],
    ),
]


def get_settings_schema() -> list[dict[str, Any]]:
    return [spec.to_dict() for spec in SETTINGS_SCHEMA]


def get_keys_for_mode(mode: RunMode) -> list[str]:
    return [spec.key for spec in SETTINGS_SCHEMA if mode in spec.modes]


def sync_settings_to_par(settings: dict[str, Any]) -> None:
    """
    Mirrors DB-persisted settings into AmfConfig.PAR, the in-memory dict
    AmfConfig.get() actually reads from. Saving/reverting settings only
    writes the Postgres Settings table by itself - nothing else keeps PAR in
    sync with that, so callers that touch the Settings table (save, revert,
    revert-all, and once at API startup) must call this afterwards, or
    AmfConfig.get(...) keeps returning stale/default values for anything not
    explicitly threaded through a per-request *Config object (e.g. use_db,
    image_directory, outdir as read by the local-file-mode functions in
    api_utils.py).
    """
    for spec in SETTINGS_SCHEMA:
        if spec.key in settings:
            AmfConfig.set_(spec.parKey, settings[spec.key], create=True, use_none=True)
