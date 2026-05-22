#! /usr/bin/env python

# AMFinder - amf
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

import torch

import amf.helper.config as AmfConfig
import amf.mode.bald as AmfBald
import amf.mode.calibrate as AmfCalibrate
import amf.mode.colonisation as AmfColonisation
import amf.mode.convert as AmfConvert
import amf.mode.convert_image_type as AmfConvertImageType
import amf.mode.convert_tile_size as AmfConvertTileSize
import amf.mode.predict as AmfPredict
import amf.mode.test as AmfTest
import amf.mode.train as AmfTrain
from amf.helper.log import logger


def main() -> None:
    AmfConfig.initialize()
    run_mode = AmfConfig.get("run_mode")

    # Device Switch
    device = AmfConfig.get("device")  # e.g. 'cpu', 'cuda:0' or 'automatic'

    # Get the device setting
    if device == "automatic":
        if torch.cuda.is_available():
            AmfConfig.set_("device", "cuda:0")
            logger.info("Device set on automatic mode. Running via gpu")

        else:
            AmfConfig.set_("device", "cpu")
            logger.info(
                "Device set on automatic mode. CUDA is not available. Running via cpu"
            )

    elif device == "cpu":
        logger.info("Device manually set to cpu")

    elif device == "cuda:0" and torch.cuda.is_available():
        logger.info("Device manually set to gpu")

    elif device == "cuda:0" and not torch.cuda.is_available():
        raise ValueError("Device manually set to gpu. CUDA not available")

    elif device not in ["cpu", "cuda:0", "cuda:1", "automatic"]:
        raise ValueError(
            'Invalid device. Set device to "cpu"", "cuda:0", "cuda:1", "automatic"'
        )

    logger.info(f"Mode: {run_mode.upper()}")

    if run_mode == "train":
        logger.debug("This is the normal training route")
        flag = AmfConfig.get("mlflow_flag")

        logger.debug("mlflow logging enabled" if flag else "mlflow logging disabled")
        input_files = AmfConfig.get_input_files()
        get_tiles_for_labelling_using_active_learning = AmfConfig.get(
            "get_tiles_for_labelling_using_active_learning"
        )
        if get_tiles_for_labelling_using_active_learning:
            logger.info("Running active learning to get tiles for labelling.")
            AmfBald.run(input_files)

        else:
            train_active_learning = AmfConfig.get("train_active_learning")
            filter_background = AmfConfig.get("filter_background")
            dynamic_loading = AmfConfig.get("dynamic_loading")
            AmfTrain.run(
                input_files,
                flag,
                train_active_learning,
                filter_background,
                dynamic_loading,
            )

    elif run_mode == "predict":
        input_files = AmfConfig.get_input_files()
        AmfPredict.run(input_files)

    elif run_mode == "test":
        input_files = AmfConfig.get_input_files()
        AmfTest.run(input_files)

    elif run_mode == "colonisation":
        logger.debug("Get colonisation percentage for annotations")
        input_files = AmfConfig.get_input_files()
        AmfColonisation.run(input_files)

    elif run_mode == "convert":
        input_files = AmfConfig.get_input_files()
        if AmfConfig.get("aggregate_tiles"):
            AmfConvertTileSize.run(input_files)
        else:
            AmfConvert.run(input_files)

    elif run_mode == "tifconversion":
        input_files = AmfConfig.get_input_files()
        AmfConvertImageType.run(input_files)

    elif run_mode == "calibrate":
        input_files = AmfConfig.get_input_files()
        AmfCalibrate.run(input_files)

    else:
        pass


if __name__ == "__main__":
    main()
