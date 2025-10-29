# AMFinder - amfinder_convert.py
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
import os
import json
import re
from api_objects import AnnotationValues
from api_utils import (
    check_entries_for_id,
    download_entries_as_csv,
    get_enabled,
    save_annotations_to_db,
    get_tile_edge,
)
from db_config import connect
import imagesize
import numpy as np
import pandas as pd

import amfinder_log as AmfLog
import amfinder_config as AmfConfig
import amfinder_diagnose as AmfDiagnose


NROWS = None
NCOLS = None


def initialize_size(path):
    """
    Retrieve the number of rows and columns based on image and tile sizes.
    """
    tile_size = AmfConfig.get("tile_edge")
    width, height = imagesize.get(path)

    image_name = os.path.splitext(os.path.basename(path))[0]
    if AmfConfig.get("use_db"):
        conn = connect("amf")
        with conn, conn.cursor() as crsr:
            id = get_enabled(crsr, image_name)
            if id != None:
                tile_edge = get_tile_edge(crsr, id)
                tile_size = tile_edge[0]
    else:
        dirname = os.path.split(path)[0]
        settings_path = f"{image_name}_settings.json"

        if settings_path in os.listdir(dirname):
            with open(dirname + "/" + settings_path) as json_file:
                x = json.load(json_file)
                tile_size = x["tile_edge"]

    global NROWS, NCOLS
    assert tile_size is not None
    NCOLS = width // tile_size
    NROWS = height // tile_size

    return tile_size


def preds_to_python_annot_1(path, preds):
    """
    Convert level 1 predictions to Python annotations.
    If useContextualConfidence is True, use the ContextualLabel column when available.
    """
    use_contextual_confidence = AmfConfig.get("use_contextual_confidence")
    # Only one file, nothing special to choose.
    preds = pd.read_csv(preds)

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
    preds_without_coord = AmfDiagnose.remove_coordinates(preds)
    preds_numpy = preds_without_coord.to_numpy()

    # Filter out any preds below the threshold
    max_scores = preds_numpy.max(axis=1)
    high_conf_mask = max_scores >= threshold

    coord = coord[high_conf_mask].reset_index(drop=True)
    preds_numpy = preds_numpy[high_conf_mask]
    if contextual_labels is not None:
        contextual_labels = contextual_labels[high_conf_mask].reset_index(drop=True)

    # Initialize conversion matrix
    conv = np.zeros_like(preds_numpy, dtype=np.uint8)

    # Set default predictions using argmax
    conv[np.arange(len(preds_numpy)), preds_numpy.argmax(1)] = 1

    # Override with contextual labels if enabled and available
    if use_contextual_confidence and contextual_labels is not None:
        AmfLog.info(
            "Using contextual confidence for converting predictions to annotations"
        )
        header = AmfConfig.get("header")
        for idx, label in enumerate(contextual_labels):
            if not pd.isnull(label):
                # Reset this row (all zeros)
                conv[idx, :] = 0
                conv[idx, header.index(label)] = 1

    print(
        os.path.basename(path)
        + "\t"
        + "\t".join([str(x) for x in np.sum(conv, axis=0)])
    )

    conv = pd.DataFrame(data=conv, columns=AmfConfig.get("header"))

    # Generate the final table.
    return pd.concat([coord, conv], axis=1)


def preds_to_python_annot_2(path, preds):
    """
    Convert level 2 predictions to Python annotations.
    """

    # Only one file, nothing special to choose.
    preds = io.StringIO(preds)
    preds = pd.read_csv(preds)

    # Get the predictions for automatic conversion.
    coord = preds[["row", "col"]]
    preds = AmfDiagnose.remove_coordinates(preds)
    preds = preds.to_numpy()

    # Perform the conversion, ignoring ties.
    conv = np.zeros_like(preds)
    conv[preds >= AmfConfig.get("threshold")] = 1

    conv = conv.astype(np.uint8)

    print(
        os.path.basename(path)
        + "\t"
        + "\t".join([str(x) for x in np.sum(conv, axis=0)])
    )

    header = AmfConfig.get("header")
    conv = pd.DataFrame(data=conv, columns=header)

    # Generate the final table.
    final = pd.concat([coord, conv], axis=1)

    # Remove the columns without annotations
    final = final.loc[(final[header].sum(axis=1) != 0),]

    return final


def preds_to_python_annot(path, preds):
    """
    Convert predictions to Python annotations.
    """
    if AmfConfig.get("level") == 1:

        return preds_to_python_annot_1(path, preds)

    else:
        return preds_to_python_annot_2(path, preds)


def update_archive(out, prev_path):
    """
    Save annotations in Python format to csv.
    """
    new_path = prev_path.replace("predictions", "annotations")
    out.to_csv(new_path, index=False)


def create_annotations(path, tile_size):

    preds = []
    image_name = os.path.splitext(os.path.basename(path))[0]

    if AmfConfig.get("use_db"):
        level = AmfConfig.get("level")
        colonisation_type = AmfConfig.get("colonisation_type")

        conn = connect("amf")
        with conn, conn.cursor() as crsr:
            id = get_enabled(crsr, image_name)

            if id == None:
                AmfLog.info(
                    f"Skipping {path} as no entries are saved in DB for this image"
                )
                return

            existing_entries = check_entries_for_id(crsr, id, colonisation_type)

            if existing_entries[id]["cnn1_annotations_exist"]:
                AmfLog.info(
                    f"Skipping {path} as annotations already exist for enabled image"
                )
                return
            elif existing_entries[id]["cnn1_predictions_exist"]:
                csv = download_entries_as_csv(
                    crsr, id, str(level), "Predictions", colonisation_type
                )
                out = preds_to_python_annot(path, io.StringIO(csv))
                cnn1results = out.values.tolist()
                values = AnnotationValues(
                    imageReferenceId=id,
                    colonisationType=colonisation_type,
                    tileEdge=tile_size,
                    cnnOneValues=cnn1results,
                    cnnTwoValues=[],
                )
                save_annotations_to_db(crsr, values)
                return

            AmfLog.info(f"Skipping {path} as no predictions could be found")
    else:
        directory = os.path.dirname(path)
        files = os.listdir(directory)

        re_check_annotations = f"{image_name}.+annotations"
        re_check_predictions = f"{image_name}.+predictions"

        # Check if any annotations already exist (this will only allow one set of
        # annotions and predictions per image, unlike the DB which allows many)
        for file in files:
            if re.match(re_check_annotations, file):
                AmfLog.info(f"Skipping {path} as annotations already exist")
                return

            if re.match(re_check_predictions, file):
                preds.append(file)

        if preds == []:

            AmfLog.info(f"Skipping {path} as no predictions could be found")

        elif len(preds) == 1:
            full_preds_path = os.path.join(directory, preds[0])
            out = preds_to_python_annot(path, full_preds_path)
            update_archive(out, full_preds_path)

        else:

            AmfLog.info(
                f"Skipping {path} as <amf convert> does not \
                    support multiple prediction files."
            )


def run(input_images):
    print("Running conversion of predictions to annotations")
    print("Image\t" + "\t".join(AmfConfig.human_readable_header()))

    if AmfConfig.get("level") == 2 and AmfConfig.get("colonisation_type") == "erm":
        AmfLog.error(
            "There is no CNN2 functionality for ErM colonisation, so fail.",
            AmfLog.ERR_INVALID_ANNOTATION_LEVEL,
        )

        return 500

    if AmfConfig.get("level") == 2:
        AmfLog.error(
            "Currently CNN2 conversion is not supported",
            AmfLog.ERR_INVALID_MODEL,
        )

        return 500

    for path in input_images:
        tile_size = initialize_size(path)
        create_annotations(path, tile_size)

    return 200
