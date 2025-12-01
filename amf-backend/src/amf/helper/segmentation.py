# AMFinder - segmentation.py
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
Image Segmentation.

Crop tiles (squares) and apply diverse image modifications.

Constants
-----------
INTERPOLATION - Interpolation mode for image resizing.

Functions
------------
:function tile: Extracts a tile from a large image.
:function preprocess: Convert a tile list to NumPy array and normalise pixels.
"""

import numpy as np
from numpy.typing import NDArray
from PIL import Image

import amf.helper.config as AmfConfig

# This allows any size image
Image.MAX_IMAGE_PIXELS = None
np.random.seed(42)


def load(image_path: str) -> Image.Image:
    """
    Loads an image using the Pillow library.
    """
    return Image.open(image_path)


def get_contextual_tiles(
    image: Image.Image,
    r: int,
    c: int,
    edge: int | None = None,
    overlap: float = 0.75,
    overlap_method: str = "diagonal",  # TODO replace with enum
) -> list[NDArray[np.uint8] | None]:
    """
    Extracts context tiles around a central tile from a large image.
    Returns None for any surrounding tile that extends beyond the image boundaries.

    :param image: The source image used to extract tiles.
    :param r: The row index of the central tile.
    :param c: The column index of the central tile.
    :param edge: The size of each tile edge. Defaults to configuration if None.
    :param overlap: The fraction of overlap between central and surrounding tiles (0-1).
    :param overlap_method: Method to extract tiles - 'cardinal'
                           (top, bottom, left, right)
                           or 'diagonal'
                           (top-left, top-right, bottom-left, bottom-right).
    :return: List of tiles based on overlap_method, each as a NumPy array or None.
    """
    # Validate overlap parameter
    if overlap < 0 or overlap > 1:
        raise ValueError("Overlap must be between 0 and 1")

    # Validate overlap_method parameter
    if overlap_method not in ["cardinal", "diagonal"]:
        raise ValueError("Overlap method must be 'cardinal' or 'diagonal'")

    edge = edge if edge is not None else AmfConfig.get("tile_edge")
    img_width, img_height = image.size

    # Calculate offset based on overlap
    offset = int(edge * (1 - overlap))

    def extract_tile(x: int, y: int) -> NDArray[np.uint8] | None:
        # Check if the tile is within image boundaries
        if x < 0 or y < 0 or x + edge > img_width or y + edge > img_height:
            return None
        tile = image.crop((x, y, x + edge, y + edge))
        tile = np.array(tile)
        return np.transpose(tile.astype(np.uint8), (2, 0, 1))

    # Coordinates for the central tile (top left corner)
    x_center = c * edge
    y_center = r * edge

    if overlap_method == "cardinal":
        # Top tile
        y_top = y_center - offset
        top_tile = extract_tile(x_center, y_top)

        # Bottom tile
        y_bottom = y_center + offset
        bottom_tile = extract_tile(x_center, y_bottom)

        # Left tile
        x_left = x_center - offset
        left_tile = extract_tile(x_left, y_center)

        # Right tile
        x_right = x_center + offset
        right_tile = extract_tile(x_right, y_center)

        return [top_tile, bottom_tile, left_tile, right_tile]

    elif overlap_method == "diagonal":
        # Top-left tile
        x_tl = x_center - offset
        y_tl = y_center - offset
        top_left_tile = extract_tile(x_tl, y_tl)

        # Top-right tile
        x_tr = x_center + offset
        y_tr = y_center - offset
        top_right_tile = extract_tile(x_tr, y_tr)

        # Bottom-left tile
        x_bl = x_center - offset
        y_bl = y_center + offset
        bottom_left_tile = extract_tile(x_bl, y_bl)

        # Bottom-right tile
        x_br = x_center + offset
        y_br = y_center + offset
        bottom_right_tile = extract_tile(x_br, y_br)

        return [top_left_tile, top_right_tile, bottom_left_tile, bottom_right_tile]

    raise ValueError(f"Invalid overlap method {overlap_method}")


def tile(
    image: Image.Image, r: int, c: int, edge: int | None = None
) -> NDArray[np.uint8]:
    """
    Extracts a tile from a large image, resizes it to
    the required CNN input image size, and applies
    data augmentation (if active).

    :param image: The source image used to extract tiles.
    :param r: The row index of the tile to extract.
    :param c: The column index of the tile to extract.
    :return: Set of tile, converted to numpy arrays.
    :rtype: list
    """
    # TODO check on hhow this edge is fetched
    edge = edge if edge is not None else AmfConfig.get("tile_edge")

    # Crop the tile from the image
    tile = image.crop((c * edge, r * edge, (c + 1) * edge, (r + 1) * edge))

    width, height = tile.size
    if AmfConfig.get("tile_edge") != edge:
        # TODO this breaks when input size does not equal edge - look into this
        ratio = AmfConfig.get("tile_edge") / edge
        # NEAREST for performance
        tile = tile.resize((int(width * ratio), int(height * ratio)), Image.NEAREST)

    tile = np.array(tile)

    # 3 channels for RGB
    return np.transpose(tile.astype(np.uint8), (2, 0, 1))


def preprocess(tile_list: list[NDArray[np.uint8]]) -> NDArray[np.float32]:
    """
    Preprocess a list of tiles.

    :param tile_list: list of tiles extracted using the function above.
    :return: a numpy array containing normalised pixel values for several tiles.
    :rtype: numpy.ndarray
    """

    return np.array(tile_list, np.float32) / 255.0
