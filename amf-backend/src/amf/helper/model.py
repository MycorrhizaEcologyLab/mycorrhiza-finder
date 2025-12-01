# AMFinder - model.py
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
ConvNet Builder.

Builds the convolutional neural networks used by AMFinder.

Functions
------------
:class ConvolutionalBlocks: Builds the Convolutional Backbone of CNN1
:class FCLayers: Builds fully connected/dropout layers.
:class CNN1: Builds CNN1 as a class
:function create_cnn1: Builds a network for root segmentation.
:function load: main function, to be called from outside.
"""

import os
from typing import Type, cast

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision import models

import amf.helper.config as AmfConfig
import amf.helper.log as AmfLog


class ConvolutionalBlocks(nn.Module):
    def __init__(self) -> None:
        super(
            ConvolutionalBlocks, self
        ).__init__()  # Perhaps use different type of super() function assignment

        kc = 32  # Initial kernel count

        # Convolution Block 1
        # Earlier: 126 x 126 --> 120 x 120.
        # With 252 x 252 Input, this means it is: 252 x 252 --> 252 x 252
        # (with padding applied)
        # With 252 x 252 Input, this means it is: 252 x 252 --> 246 x 246
        # (without padding)

        self.conv11 = nn.Conv2d(in_channels=3, out_channels=kc, kernel_size=3)

        self.conv12 = nn.Conv2d(in_channels=kc, out_channels=kc, kernel_size=3)

        self.conv13 = nn.Conv2d(in_channels=kc, out_channels=kc, kernel_size=3)

        # Adding pooling. Earlier: 120 x 120 --> 60 x 60
        # With 252 x 252 Input, this means it is: 252 x 252 --> 126 x 126
        # (with padding applied)
        # With 252 x 252 Input, this means it is: 246 x 246 --> 123 x 123
        # (without padding)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # Update kernel count in conv layers
        kc *= 2  # 64

        # Convolution Block 2
        # Earlier: 60 x 60 --> 56 x 56.
        # With 252 x 252 Input, this means it is: 126 x 126 --> 126 x 126
        # (with padding applied)
        # With 252 x 252 Input, this means it is: 123 x 123 --> 119 x 119
        # (without applied)
        self.conv21 = nn.Conv2d(in_channels=kc // 2, out_channels=kc, kernel_size=3)

        self.conv22 = nn.Conv2d(in_channels=kc, out_channels=kc, kernel_size=3)

        # Adding pooling. Earlier: 56 x 56 --> 28 x 28
        # With 252 x 252 Input, this means it is: 126 x 126 --> 63 x 63
        # (with padding applied)
        # With 252 x 252 Input, this means it is: 119 x 119 --> 59 x 59
        # (without padding applied ("half" a pixel appears to be omitted))
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # Update kernel count in conv layers
        kc *= 2  # 128

        # Convolution Block 3
        # Earlier: 28 x 28 --> 24 x 24.
        # With 252 x 252 Input, this means it is: 63 x 63 --> 63 x 63
        # (with padding applied)
        # With 252 x 252 Input, this means it is: 59 x 59 --> 55 x 55
        # (without padding applied)
        self.conv31 = nn.Conv2d(in_channels=kc // 2, out_channels=kc, kernel_size=3)

        self.conv32 = nn.Conv2d(in_channels=kc, out_channels=kc, kernel_size=3)

        # Adding pooling. Earlier: 24 x 24 --> 12 x 12
        # With 252 x 252 Input, this means it is: 63 x 63 --> 31 x 31
        # (with padding applied, ("half" a pixel appears to be omitted))
        # With 252 x 252 Input, this means it is: 55 x 55 --> 27 x 27
        # (without applied ("half" a pixel appears to be omitted))
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        # Update kernel count in conv layers
        kc *= 2  # 256

        # Convolution Block 4
        # Earlier: 12 x 12 --> 10 x 10.
        # With 252 x 252 Input, this means it is: 31 x 31 --> 31 x 31
        # (with padding applied)
        # With 252 x 252 Input, this means it is: 27 x 27 --> 25 x 25
        # (without padding applied)
        self.conv4 = nn.Conv2d(in_channels=kc // 2, out_channels=kc, kernel_size=3)

        # Adding pooling. Earlier: 10 x 10 --> 5 x 5
        # With 252 x 252 Input, this means it is: 31 x 31 --> 15 x 15
        # (with padding applied, ("half" a pixel appears to be omitted))
        # With 252 x 252 Input, this means it is: 27 x 27 --> 13 x 13
        # (without applied ("half" a pixel appears to be omitted))
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        # Flatten as per original architecture
        self.flatten = nn.Flatten()

        tile_edge = AmfConfig.get("tile_edge")
        dummy_input = torch.zeros(1, 3, tile_edge, tile_edge)
        self.flatten_size = self.get_conv_output_size(dummy_input)

        # Initialise weights
        self._initialise_weights()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Keep the input for return
        input_layer = x
        # x = x.permute(0, 3, 1, 2)

        # Convolution Block 1
        x = self.conv11(x)
        x = nn.ReLU()(x)
        x = self.conv12(x)
        x = nn.ReLU()(x)
        x = self.conv13(x)
        x = nn.ReLU()(x)
        x = self.pool1(x)

        # Convolution Block 2
        x = self.conv21(x)
        x = nn.ReLU()(x)
        x = self.conv22(x)
        x = nn.ReLU()(x)
        x = self.pool2(x)

        # Convolution Block 3
        x = self.conv31(x)
        x = nn.ReLU()(x)
        x = self.conv32(x)
        x = nn.ReLU()(x)
        x = self.pool3(x)

        # Final Convolution
        x = self.conv4(x)
        x = nn.ReLU()(x)
        x = self.pool4(x)

        # Flatten
        x = self.flatten(x)

        return (input_layer, x)

    def _initialise_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    init.zeros_(m.bias)

    def get_conv_output_size(self, input_tensor: torch.Tensor) -> int:
        # Pass the input through the convolutional layers to compute the output size
        xo = self.conv11(input_tensor)
        xo = nn.ReLU()(xo)
        xo = self.conv12(xo)
        xo = nn.ReLU()(xo)
        xo = self.conv13(xo)
        xo = nn.ReLU()(xo)
        xo = self.pool1(xo)

        xo = self.conv21(xo)
        xo = nn.ReLU()(xo)
        xo = self.conv22(xo)
        xo = nn.ReLU()(xo)
        xo = self.pool2(xo)

        xo = self.conv31(xo)
        xo = nn.ReLU()(xo)
        xo = self.conv32(xo)
        xo = nn.ReLU()(xo)
        xo = self.pool3(xo)

        xo = self.conv4(xo)
        xo = nn.ReLU()(xo)
        xo = self.pool4(xo)

        # Flatten the xo
        xo = xo.view(xo.size(0), -1)  # Flattened view for fully connected layer
        return cast(int, xo.size(1))  # Return number of features


class FCLayers(nn.Module):
    def __init__(
        self, fc_in_size: int, output_size: int = 1
    ) -> None:  # activation='sigmoid'):
        super(FCLayers, self).__init__()

        # Layers
        self.fc1 = nn.Linear(
            in_features=fc_in_size, out_features=128
        )  # in_features may need to be adjusted according to final size decision
        self.dropout1 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.dropout2 = nn.Dropout(p=0.2)
        self.output = nn.Linear(in_features=64, out_features=output_size)

        # Initialise weights
        self._initialise_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First pass through fc1
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.dropout1(x)

        # Second pass through fc2
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.dropout2(x)

        # Output
        output = self.output(x)

        # Activation function
        # output = self.activation(output)

        return cast(torch.Tensor, output)

    def _initialise_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    init.zeros_(m.bias)


class CNN1(nn.Module):
    def __init__(self) -> None:
        super(CNN1, self).__init__()
        self.conv = ConvolutionalBlocks()
        self.fc = FCLayers(
            fc_in_size=self.conv.flatten_size,
            output_size=len(AmfConfig.get("header")),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return cast(torch.Tensor, self.fc(x[1]))


def create_cnn1() -> CNN1:
    """
    Builds a single-label, multi-class classifier to discriminate
    between different classes for AM or ErM roots.
    """

    model = CNN1()

    return model


def create_resnet50(num_classes: int = 6, pre_trained: bool = False) -> torch.nn.Module:
    # Initialize ResNet50 without pre-trained weights

    if pre_trained:
        model = models.resnet50(weights="IMAGENET1K_V1")

    else:
        model = models.resnet50(weights=None)

    # Modify the final fully connected layer for your number of classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return cast(torch.nn.Module, model)


def create_resnext50(
    num_classes: int = 6, pre_trained: bool = False
) -> torch.nn.Module:
    """
    Builds ResNeXt50 model with 32x4d configuration.
    This is comparable to ResNet50 in terms of parameters.
    """
    if pre_trained:
        model = models.resnext50_32x4d(
            weights="IMAGENET1K_V1"
        )  # You may specify a pretrained flag
    else:
        model = models.resnext50_32x4d(weights=None)

    # Modify the final fully connected layer to match the number of classes you need
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return cast(torch.nn.Module, model)


def create_efficientnetb5(
    num_classes: int = 6, pre_trained: bool = False
) -> torch.nn.Module:
    # Initialise EfficientNet with pretrained weights
    if pre_trained:
        model = models.efficientnet_b5(weights="IMAGENET1K_V1")

    else:
        model = models.efficientnet_b5(weights=None)

    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)

    return cast(torch.nn.Module, model)


def create_efficientnet_v2_m(
    num_classes: int = 6, pre_trained: bool = False
) -> torch.nn.Module:
    # Initialise EfficientNet with pretrained weights
    if pre_trained:
        model = models.efficientnet_v2_m(weights="IMAGENET1K_V1")

    else:
        model = models.efficientnet_v2_m(weights=None)

    # Access the classifier layer in EfficientNet V2 M
    num_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_features, num_classes)

    return cast(torch.nn.Module, model)


def load(name: str | None = None) -> torch.nn.Module:
    """
    Loads or initialises a convolutional neural network.
    """

    colonisation_type = AmfConfig.get("colonisation_type")
    if name is not None:
        # Check if model exists in trained network, otherwise return name to allow
        # absolute paths
        path = os.path.join(AmfConfig.get_model_dir(), name)

        if not os.path.isfile(path):
            path = name

    else:
        path = (
            AmfConfig.get("model")
            if colonisation_type == "am"
            else AmfConfig.get("model_erm")
        )

    if path is not None and os.path.isfile(path):
        # This will currently only work for pth models.

        AmfLog.text(f"Model for {colonisation_type} colonisation: {path}")
        model = torch.load(path, map_location=torch.device(AmfConfig.get("device")))
        AmfLog.text("Model load successful")

        # Check model name
        if isinstance(model, torch.nn.Module):
            model_name = model.__class__.__name__  # Get class name of loaded model

            # TODO: ResNeXt has the same model.__class__.__name__ as ResNet.
            # Change to map correctly.
            if model_name in ["CNN1", "ResNet", "ResNeXt", "EfficientNet"]:
                AmfLog.text(f"Model type: {model_name}")
                return model
            else:
                raise ValueError(
                    "Not a valid model. Valid models are CNN1, ResNet, ResNeXt, "
                    "EfficientNet"
                )

        else:
            AmfLog.error(
                "The provided model is not a torch.nn.Module instance",
                exit_code=AmfLog.ERR_NO_PRETRAINED_MODEL,
            )
            raise RuntimeError(
                "Unreachable code after AmfLog.error()"
            )  # TODO replace with proper exceptions

    else:
        # Initialise a new network if no valid model was found
        if AmfConfig.get("run_mode") == "train":
            pt_flag = AmfConfig.get("pre_trained")
            num_classes = len(AmfConfig.get("header"))

            if AmfConfig.get("model_type") == "cnn1":
                if pt_flag:
                    AmfLog.error(
                        "Pre-trained weights are unanavailable for CNN1. "
                        "Please proceed with pre_trained=False",
                        exit_code=AmfLog.ERR_NO_PRETRAINED_MODEL,
                    )
                    raise RuntimeError(
                        "Unreachable code after AmfLog.error()"
                    )  # TODO replace with proper exceptions
                else:
                    model = create_cnn1()

            elif AmfConfig.get("model_type") == "resnet":
                model = create_resnet50(num_classes=num_classes, pre_trained=pt_flag)

            elif AmfConfig.get("model_type") == "resnext":
                model = create_resnext50(num_classes=num_classes, pre_trained=pt_flag)

            elif AmfConfig.get("model_type") == "efficientnet":
                model = create_efficientnetb5(
                    num_classes=num_classes, pre_trained=pt_flag
                )

            elif AmfConfig.get("model_type") == "efficientnetv2":
                model = create_efficientnet_v2_m(
                    num_classes=num_classes, pre_trained=pt_flag
                )

            else:
                AmfLog.error(
                    "Invalid model type. Please choose one of the following model "
                    "types: 'cnn1', 'resnet', 'resnext', 'efficientnet' or "
                    "'efficientnetv2'.",
                    exit_code=AmfLog.ERR_INVALID_MODEL,
                )
                raise RuntimeError(
                    "Unreachable code after AmfLog.error()"
                )  # TODO replace with proper exceptions

            model_name = AmfConfig.get("model_type")  # Get class name of loaded model
            AmfLog.text(
                f"Initialise new network. Selected model type: {model_name}. "
                f"Pre-Trained Flag: {pt_flag}"
            )

            return cast(torch.nn.Module, model)

        else:  # missing pre-trained model in prediction mode.
            AmfLog.error(
                "A pre-trained model is required in prediction/calibration/test mode",
                exit_code=AmfLog.ERR_NO_PRETRAINED_MODEL,
            )
            raise RuntimeError(
                "Unreachable code after AmfLog.error()"
            )  # TODO replace with proper exceptions


def filter_layers(
    model: torch.nn.Module, layer_type: Type[torch.nn.Module]
) -> list[torch.nn.Module]:
    """
    Return all layers from a <model> that belong to a given <layer_type>.
    """
    return [layer for layer in model.modules() if isinstance(layer, layer_type)]


class FeatureExtractor(nn.Module):
    def __init__(self, layers: list[torch.nn.Module]) -> None:
        super(FeatureExtractor, self).__init__()
        # Combine the filtered layers into a new module
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.layers(x))  # Forward through the selected layers


def get_feature_extractors(model: torch.nn.Module) -> list[FeatureExtractor]:
    """
    Builds feature extractor models for all convolutional layers (Conv2d).
    """
    conv_layers = filter_layers(model, nn.Conv2d)  # Get all Conv2D layers
    return [FeatureExtractor(conv_layers[: i + 1]) for i in range(len(conv_layers))]
