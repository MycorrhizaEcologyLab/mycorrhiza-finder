# AMFinder - convert.py
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


import io
import json
import os
import re
from typing import cast

import imagesize
import numpy as np
import pandas as pd
from loguru import logger

import amf.helper.config as AmfConfig
from amf.helper.api_objects import AnnotationValues
from amf.helper.api_utils import (
    check_entries_for_id,
    download_entries_as_csv,
    get_enabled,
    get_tile_edge,
    save_annotations_to_db,
)
from amf.helper.db_config import connect

NROWS: int | None = None
NCOLS: int | None = None


def initialize_size(path: str) -> int:
    """
    Retrieve the number of rows and columns based on image and tile sizes.
    """
    tile_size = AmfConfig.get("tile_edge")
    width, height = imagesize.get(path)

    image_name = os.path.splitext(os.path.basename(path))[0]
    if AmfConfig.get("use_db"):
        conn = connect("amf")
        with conn, conn.cursor() as crsr:
            id_ = get_enabled(crsr, image_name)
            if id_ is not None:
                tile_edge = get_tile_edge(crsr, id_)
                tile_size = tile_edge[0]
    else:
        dirname = os.path.split(path)[0]
        settings_path = f"{image_name}_settings.json"

        if settings_path in os.listdir(dirname):
            with open(dirname + "/" + settings_path) as json_file:
                x = json.load(json_file)
                tile_size = x["tile_edge"]

    global NROWS, NCOLS
    if tile_size is None:
        raise ValueError("Tile size not configured")
    tile_size_int = cast(int, tile_size)
    NCOLS = width // tile_size_int
    NROWS = height // tile_size_int

    return tile_size_int


def preds_to_python_annot(path: str, preds_path: str | io.StringIO) -> pd.DataFrame:
    """
    Convert predictions to Python annotations.
    If useContextualConfidence is True, use the ContextualLabel column when available.
    """
    use_contextual_confidence = AmfConfig.get("use_contextual_confidence")
    # Only one file, nothing special to choose.
    preds = pd.read_csv(preds_path)

    # Get the coordinates
    coord = preds[["row", "col"]]

    threshold = AmfConfig.get("threshold")

    # Extract and handle contextual labels if needed
    contextual_labels = None
    if "ContextualLabel" in preds.columns:
        if use_contextual_confidence:
            contextual_labels = preds["ContextualLabel"].copy()
        # Always drop the ContextualLabel column
        preds = preds.drop(columns=["ContextualLabel"])

    # Get the predictions for automatic conversion
    preds_without_coord = preds.drop(["row", "col"], axis=1)
    preds_numpy = preds_without_coord.to_numpy()

    header = cast(list[str], AmfConfig.get("header"))
    if preds_numpy.shape[1] != len(header):
        raise ValueError(
            f"Predictions file {os.path.basename(str(preds_path))} has "
            f"{preds_numpy.shape[1]} class column(s) but the current "
            f"colonisation type ('{AmfConfig.get('colonisation_type')}') "
            f"expects {len(header)} ({', '.join(header)}). This usually means "
            f"the file doesn't actually hold predictions for that colonisation "
            f"type (e.g. it was accidentally overwritten with a different "
            f"file's contents) - check the file, or that the correct "
            f"colonisation type is selected, before converting."
        )

    # Filter out any preds below the threshold
    max_scores = preds_numpy.max(axis=1)
    high_conf_mask = max_scores >= threshold

    coord = coord[high_conf_mask].reset_index(drop=True)
    preds_numpy = preds_numpy[high_conf_mask]
    if contextual_labels is not None:
        contextual_labels = contextual_labels[high_conf_mask].reset_index(drop=True)

    if len(preds_numpy) == 0:
        logger.warning(
            f"No tiles in {os.path.basename(path)} met the confidence "
            f"threshold ({threshold}) - skipping annotation conversion for "
            f"this image."
        )
        return pd.DataFrame(columns=["row", "col"] + header)

    # Initialize conversion matrix
    conv = np.zeros_like(preds_numpy, dtype=np.uint8)

    # Set default predictions using argmax
    conv[np.arange(len(preds_numpy)), preds_numpy.argmax(1)] = 1

    # Override with contextual labels if enabled and available
    if use_contextual_confidence and contextual_labels is not None:
        logger.info(
            "Using contextual confidence for converting predictions to annotations"
        )
        for idx, label in enumerate(contextual_labels):
            if not pd.isnull(label):
                # Reset this row (all zeros)
                conv[idx, :] = 0
                conv[idx, header.index(label)] = 1

    logger.debug(
        os.path.basename(path)
        + "\t"
        + "\t".join([str(x) for x in np.sum(conv, axis=0)])
    )

    conv = pd.DataFrame(data=conv, columns=header)

    # Generate the final table.
    return pd.concat([coord, conv], axis=1)


def update_archive(out: pd.DataFrame, prev_path: str) -> None:
    """
    Save annotations in Python format to csv.
    """
    new_path = prev_path.replace("predictions", "annotations")
    out.to_csv(new_path, index=False)


def create_annotations(path: str, tile_size: int) -> None:
    preds = []
    image_name = os.path.splitext(os.path.basename(path))[0]

    if AmfConfig.get("use_db"):
        colonisation_type = AmfConfig.get("colonisation_type")

        conn = connect("amf")
        with conn, conn.cursor() as crsr:
            id_ = get_enabled(crsr, image_name)

            if id_ is None:
                logger.info(
                    f"Skipping {path} as no entries are saved in DB for this image"
                )
                return

            existing_entries = check_entries_for_id(crsr, id_, colonisation_type)

            if existing_entries[id_]["cnn1_annotations_exist"]:
                logger.info(
                    f"Skipping {path} as annotations already exist for enabled image"
                )
                return
            elif existing_entries[id_]["cnn1_predictions_exist"]:
                csv = download_entries_as_csv(
                    crsr, id_, "Predictions", colonisation_type
                )
                out = preds_to_python_annot(path, io.StringIO(csv))
                if out.empty:
                    logger.info(
                        f"Skipping {path} as no tiles cleared the confidence "
                        f"threshold - no annotations to save"
                    )
                    return
                cnn1results = out.values.tolist()
                values = AnnotationValues(
                    imageReferenceId=id_,
                    colonisationType=colonisation_type,
                    tileEdge=tile_size,
                    cnnOneValues=cnn1results,
                )
                save_annotations_to_db(crsr, values)
                return

            logger.info(f"Skipping {path} as no predictions could be found")
    else:
        directory = os.path.dirname(path)
        files = os.listdir(directory)

        # Escaped so regex metacharacters in an image name (e.g. '+', '(' ) are
        # matched literally. Deliberately loose about the timestamp between the
        # name and the type, so both the current and legacy filename
        # conventions are picked up.
        re_check_annotations = f"{re.escape(image_name)}.+annotations"
        re_check_predictions = f"{re.escape(image_name)}.+predictions"

        # Check if any annotations already exist (this will only allow one set of
        # annotions and predictions per image, unlike the DB which allows many)
        for file in files:
            if re.match(re_check_annotations, file):
                logger.info(f"Skipping {path} as annotations already exist")
                return

            if re.match(re_check_predictions, file):
                preds.append(file)

        if preds == []:
            logger.info(f"Skipping {path} as no predictions could be found")

        elif len(preds) == 1:
            full_preds_path = os.path.join(directory, preds[0])
            out = preds_to_python_annot(path, full_preds_path)
            if out.empty:
                logger.info(
                    f"Skipping {path} as no tiles cleared the confidence "
                    f"threshold - no annotations to save"
                )
                return
            update_archive(out, full_preds_path)

        else:
            logger.info(
                f"Skipping {path} as <amf convert> does not "
                f"support multiple prediction files."
            )


def run(input_images: list[str]) -> int:
    logger.info("Running conversion of predictions to annotations")
    logger.info("Image\t" + "\t".join(AmfConfig.human_readable_header()))

    for path in input_images:
        tile_size = initialize_size(path)
        create_annotations(path, tile_size)

    return 200
