"""Image segmentation (tiling) functionality."""

import numpy as np
from numpy.typing import NDArray
from PIL import Image

import amf.helper.config as AmfConfig

# This allows any size image
Image.MAX_IMAGE_PIXELS = None
np.random.seed(42)


def load(image_path: str) -> Image.Image:
    """Load image into memory.

    Args:
        image_path: Path to the image file.

    Returns: PIL image object.
    """
    return Image.open(image_path)


def get_contextual_tiles(
    image: Image.Image,
    r: int,
    c: int,
    edge: int | None = None,
    overlap: float = 0.75,
    overlap_method: str = "diagonal",  # TODO replace with enum?
) -> list[NDArray[np.uint8] | None]:
    """Extract context tiles around a central tile from a large image.

    Returns None for any surrounding tile that extends beyond the image boundaries.

    Args:
        image: Source image from which to extract tiles.
        r: Row index of the central tile.
        c: Column index of the central tile.
        edge: Tile edge length. If None, uses default from configuration.
        overlap: Fraction of overlap between central and surrounding tiles (0-1).
        overlap_method: Method to extract tiles - 'cardinal' (top, bottom, left, right)
            or 'diagonal' (top-left, top-right, bottom-left, bottom-right).

    Returns: List of tiles based on overlap_method, each as a NumPy array or None.
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

    if overlap_method == "diagonal":
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
    """Extract a single tile from an image.

    Args:
        image: Source image from which to extract the tile.
        r: Row index of the tile to extract.
        c: Column index of the tile to extract.
        edge: Tile edge length. If None, uses default from configuration.

    Returns: Extracted tile as raw RGB array, shape (3, edge, edge).
    """
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
    """Return a single array containing all tile pixels normalised between 0 and 1.

    Args:
        tile_list: List of unnormalised tiles as numpy arrays.

    Returns: A single array where the first axis is the tile index, with pixel values
        normalised to run from 0 to 1.

    Preprocess a list of tiles.

    :param tile_list: list of tiles extracted using the function above.
    :return: a numpy array containing normalised pixel values for several tiles.
    :rtype: numpy.ndarray
    """
    return np.array(tile_list, np.float32) / 255.0
