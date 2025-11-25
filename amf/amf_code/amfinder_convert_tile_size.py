import io
import json
import os
import re
from typing import cast

import imagesize
import pandas as pd

from . import amfinder_config as AmfConfig
from . import amfinder_log as AmfLog
from .api_objects import AnnotationValues
from .api_utils import (
    check_entries_for_id,
    download_entries_as_csv,
    get_enabled,
    get_tile_edge,
    save_annotations_to_db,
)
from .db_config import connect

SCALING_FACTOR = 2
NROWS: int
NCOLS: int


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


def rescale_annot(annotation_path: str | io.StringIO) -> pd.DataFrame:
    """
    Scale Python annotations, 1x to 4x.
    """
    annotation_data = pd.read_csv(
        annotation_path
    )  # Creates a pandas Dataframe from the annotations read from zip file.

    if "Question" in annotation_data.columns:
        num_questions = annotation_data["Question"].sum()
        if num_questions > 0:
            AmfLog.error(
                f"Cannot carry out tile conversion if questions exist, currently there "
                f"are {num_questions} question(s)",
                AmfLog.ERR_INVALID_DATA,
            )

        # Drop questions column
        annotation_data.drop("Question", axis=1, inplace=True, errors="ignore")

    if "QuestionComment" in annotation_data.columns:
        # Drop question comment column
        annotation_data.drop("QuestionComment", axis=1, inplace=True, errors="ignore")

    new_row_starts = list(
        range(0, annotation_data["row"].max(), SCALING_FACTOR)
    )  # Creating a sequence of new row indices with a step of two, based on the max
    #    row number.
    new_col_starts = list(
        range(0, annotation_data["col"].max(), SCALING_FACTOR)
    )  # Creating a sequence of new col indices with a step of two, based on the max
    #    col number.

    rescaled_tile_annotations = []

    for new_row_coord, old_row_coord in enumerate(new_row_starts):
        for new_col_coord, old_col_coord in enumerate(new_col_starts):
            tile_annotations = annotation_data.loc[
                annotation_data["row"].isin([old_row_coord, old_row_coord + 1])
            ].loc[annotation_data["col"].isin([old_col_coord, old_col_coord + 1])]

            # Create pandas Series on the frequency of each class within the new tile
            # size.
            class_totals = (
                tile_annotations[
                    [
                        "AMColonised",
                        "Uncolonised",
                        "Background",
                        "Unreadable",
                        "DSE",
                        "Hybrid",
                    ]
                ]
                .sum()
                .astype(int)
            )

            # Initialize new result dict.
            this_result = {
                "row": new_row_coord,
                "col": new_col_coord,
                "AMColonised": 0,
                "Uncolonised": 0,
                "Background": 0,
                "Unreadable": 0,
                "DSE": 0,
                "Hybrid": 0,
            }

            # Entering rule tree
            # New tiles that do not contain any of the old tiles are omitted, to stay
            # consistent with previous labelling strategies.
            if tile_annotations.shape[0] == 0:
                continue

            else:
                number_of_126tiles_in_256tiles = tile_annotations.shape[0]
                # According to documentation: Return index of FIRST occurrence of
                # maximum over requested axis.
                majority_class_name = class_totals.idxmax()
                majority_class_ratio = (
                    class_totals.max() / number_of_126tiles_in_256tiles
                )

                # If all old tiles within a new tile have the same class (independent of
                # the number of tiles), it is the new tile class.
                if majority_class_ratio == 1.0:
                    this_result[majority_class_name] = 1

                # If multiple classes exist within new tile
                else:
                    ratios = {
                        key: value / number_of_126tiles_in_256tiles
                        for key, value in class_totals.items()
                    }

                    # All AMColonised cases
                    if majority_class_name == "AMColonised":
                        # If there is any amount of DSE (including a Hybrid tile) from
                        # the old tiles, new tiles should be classified as Hybrid.
                        if ratios["DSE"] > 0.0 or ratios["Hybrid"] > 0.0:
                            this_result["Hybrid"] = 1

                        # All other cases, including the cases where the majority class
                        # ratio is only 0.33 or 0.25
                        else:
                            this_result["AMColonised"] = 1

                    # All Uncolonised cases
                    # The following consideration is true for all cases
                    # (including ratio['Uncolonised'] >= 0.25)
                    if majority_class_name == "Uncolonised":
                        # All types of colonisation are given priority, as no minimum
                        # colonisation is required.
                        # If there is any AMColonised (as minority) or DSE/Hybrid tiles
                        # present (also as minority):
                        if ratios["AMColonised"] > 0.0 and (
                            ratios["DSE"] > 0.0 or ratios["Hybrid"] > 0.0
                        ):
                            this_result["Hybrid"] = 1

                        # If there is any AMColonised (as minority), DSE/Hybrid == 0.0
                        elif (
                            ratios["AMColonised"] > 0.0
                            and ratios["DSE"] == 0.0
                            and ratios["Hybrid"] == 0.0
                        ):
                            this_result["AMColonised"] = 1

                        # If there is a single DSE tile present
                        elif (
                            ratios["AMColonised"] == 0.0
                            and ratios["DSE"] > 0.0
                            and ratios["Hybrid"] == 0.0
                        ):
                            this_result["DSE"] = 1

                        # If there is a single Hybrid tile present
                        elif (
                            ratios["AMColonised"] == 0.0
                            and ratios["DSE"] == 0.0
                            and ratios["Hybrid"] > 0.0
                        ):
                            this_result["Hybrid"] = 1

                        # If there is a single Hybrid tile present, whilst a DSE tile is
                        # present as well.
                        elif (
                            ratios["AMColonised"] == 0.0
                            and ratios["DSE"] > 0.0
                            and ratios["Hybrid"] > 0.0
                        ):
                            this_result["Hybrid"] = 1

                        # Cases where there is no AMColonised, DSE or Unreadable are all
                        # considered Uncolonised, as only 10 % need to be occupied
                        # to be classified as Uncolonised, including cases where
                        # background and unreadable are present.
                        else:
                            this_result["Uncolonised"] = 1

                    # All Background cases
                    # The following consideration is true for all cases
                    # (including ratio['Background'] >= 0.25)
                    if majority_class_name == "Background":
                        # All types of colonisation are given priority, as no minimum
                        # colonisation is required.
                        # If there is any AMColonised (as minority) or DSE/Hybrid tiles
                        # present (also as minority):
                        if ratios["AMColonised"] > 0.0 and (
                            ratios["DSE"] > 0.0 or ratios["Hybrid"] > 0.0
                        ):
                            this_result["Hybrid"] = 1

                        # If there is any AMColonised (as minority), DSE/Hybrid == 0.0
                        elif (
                            ratios["AMColonised"] > 0.0
                            and ratios["DSE"] == 0.0
                            and ratios["Hybrid"] == 0.0
                        ):
                            this_result["AMColonised"] = 1

                        # If there is a single DSE tile present
                        elif (
                            ratios["AMColonised"] == 0.0
                            and ratios["DSE"] > 0.0
                            and ratios["Hybrid"] == 0.0
                        ):
                            this_result["DSE"] = 1

                        # If there is a single Hybrid tile present
                        elif (
                            ratios["AMColonised"] == 0.0
                            and ratios["DSE"] == 0.0
                            and ratios["Hybrid"] > 0.0
                        ):
                            this_result["Hybrid"] = 1

                        # If there is a single Hybrid tile present, whilst a DSE tile is
                        # present as well.
                        elif (
                            ratios["AMColonised"] == 0.0
                            and ratios["DSE"] > 0.0
                            and ratios["Hybrid"] > 0.0
                        ):
                            this_result["Hybrid"] = 1

                        # If there is a single Uncolonised tile present, whilst all
                        # Colonisation tiles are 0, it is Uncolonised in all cases.
                        elif (
                            ratios["AMColonised"] == 0.0
                            and ratios["DSE"] == 0.0
                            and ratios["Hybrid"] == 0.0
                            and ratios["Uncolonised"] > 0.0
                        ):
                            this_result["Uncolonised"] = 1

                        # If there is a single Unreadable tile present, whilst all other
                        # classes are 0, it is Unreadable in all cases, even if
                        # Background is the majority.
                        elif (
                            ratios["AMColonised"] == 0.0
                            and ratios["DSE"] == 0.0
                            and ratios["Hybrid"] == 0.0
                            and ratios["Uncolonised"] == 0.0
                            and ratios["Unreadable"] > 0.0
                        ):
                            this_result["Unreadable"] = 1

                        else:
                            this_result["Background"] = 1

                    # All Unreadable cases
                    # The following consideration is true for all cases
                    # (including ratio['Background'] >= 0.25)
                    if majority_class_name == "Unreadable":
                        # If Unreadable is strictly larger than 50 % of the new tile, it
                        # should be considered Unreadable.
                        if ratios["Unreadable"] > 0.5:
                            this_result["Unreadable"] = 1

                        # If it is equal or less than 50 %, cases need to be
                        # distinguished.
                        elif ratios["Unreadable"] <= 0.5:
                            # Again, all types of colonisation are given priority as no
                            # minimum colonisation is required.
                            # If there is any AMColonised (as minority) or DSE/Hybrid
                            # tiles present (also as minority):
                            if ratios["AMColonised"] > 0.0 and (
                                ratios["DSE"] > 0.0 or ratios["Hybrid"] > 0.0
                            ):
                                this_result["Hybrid"] = 1

                            # If there is any AMColonised (as minority),
                            # DSE/Hybrid == 0.0
                            elif (
                                ratios["AMColonised"] > 0.0
                                and ratios["DSE"] == 0.0
                                and ratios["Hybrid"] == 0.0
                            ):
                                this_result["AMColonised"] = 1

                            # If there is a single DSE tile present
                            elif (
                                ratios["AMColonised"] == 0.0
                                and ratios["DSE"] > 0.0
                                and ratios["Hybrid"] == 0.0
                            ):
                                this_result["DSE"] = 1

                            # If there is a single Hybrid tile present
                            elif (
                                ratios["AMColonised"] == 0.0
                                and ratios["DSE"] == 0.0
                                and ratios["Hybrid"] > 0.0
                            ):
                                this_result["Hybrid"] = 1

                            # If there is a single Hybrid tile present, whilst a DSE
                            # tile is present as well.
                            elif (
                                ratios["AMColonised"] == 0.0
                                and ratios["DSE"] > 0.0
                                and ratios["Hybrid"] > 0.0
                            ):
                                this_result["Hybrid"] = 1

                            # If there is a single Uncolonised tile present, whilst all
                            # Colonisation tiles are 0, it is Uncolonised in all cases.
                            elif (
                                ratios["AMColonised"] == 0.0
                                and ratios["DSE"] == 0.0
                                and ratios["Hybrid"] == 0.0
                                and ratios["Uncolonised"] > 0.0
                            ):
                                this_result["Uncolonised"] = 1

                            else:
                                this_result["Unreadable"] = 1

                    # All DSE cases
                    # The following consideration is true for all cases
                    # (including ratio['DSE'] >= 0.25)
                    if majority_class_name == "DSE":
                        # If there is a single AMColonised or Hybrid tile present whilst
                        # DSE is present.
                        if ratios["AMColonised"] > 0.0 or ratios["Hybrid"] > 0.0:
                            this_result["Hybrid"] = 1

                        # In all other cases, AMColonised and DSE == 0.0 and the other
                        # classes have lower priority as compared to Hybrid.
                        # Technically, Hybrid could be summarised within one criterion,
                        # but this is written out explicitly for consistency.
                        else:
                            this_result["DSE"] = 1

                    # All Hybrid cases
                    # The following consideration is true for all cases
                    # (including ratio['DSE'] >= 0.25). Once a single Hybrid tile is
                    # present and has at leas the same ratio as compared to all other
                    # classes, the new tile class is Hybrid.
                    if majority_class_name == "Hybrid":
                        this_result["Hybrid"] = 1

            rescaled_tile_annotations.append(this_result)

    # Generate the final table.
    return pd.DataFrame.from_dict(rescaled_tile_annotations)


def convert_tile_size(path: str, tile_size: int) -> None:
    # Make tile size 4x bigger, by merging 4 tiles into 1
    # Write back to the annotations file

    image_name = os.path.splitext(os.path.basename(path))[0]

    if AmfConfig.get("use_db"):
        colonisation_type = AmfConfig.get("colonisation_type")

        conn = connect("amf")
        with conn, conn.cursor() as crsr:
            id_ = get_enabled(crsr, image_name)

            if id_ is None:
                AmfLog.warning(
                    f"Skipping {path} as no entries are saved in DB for this image"
                )
                return

            existing_entries = check_entries_for_id(crsr, id_, colonisation_type)

            if existing_entries[id_]["cnn1_annotations_exist"]:
                csv = download_entries_as_csv(
                    crsr, id_, "Annotations", colonisation_type
                )
                out = rescale_annot(io.StringIO(csv))
                cnn1results = out.values.tolist()
                values = AnnotationValues(
                    imageReferenceId=id_,
                    colonisationType=colonisation_type,
                    tileEdge=tile_size * SCALING_FACTOR,
                    cnnOneValues=cnn1results,
                )
                save_annotations_to_db(crsr, values)
                return

            AmfLog.warning(f"Skipping {path} as no annotations could be found")

    else:
        print(f"Searching locally for {path}")
        directory = os.path.dirname(path)
        files = os.listdir(directory)

        regex_pattern = f"{image_name}_.+annotations"
        matching_annotations = [file for file in files if re.match(regex_pattern, file)]

        # Check for annotations
        if not matching_annotations:
            AmfLog.warning(f"Skipping {path} as no annotations could be found")
            return

        if len(matching_annotations) == 1:
            annotation_path = os.path.join(directory, matching_annotations[0])

            # Rescale annotations and save to CSV
            out = rescale_annot(annotation_path)
            out.to_csv(annotation_path, index=False, mode="w")

            # Update settings in the JSON file
            settings_path = os.path.join(directory, f"{image_name}_settings.json")
            with open(settings_path, "w") as file:
                settings = {}
                settings["tile_edge"] = SCALING_FACTOR * tile_size
                json.dump(settings, file)

            AmfLog.info(
                f"Saved aggregated annotations and settings to {annotation_path} and "
                f"{settings_path}"
            )
        else:
            AmfLog.warning(
                f"Multiple annotation files found for {image_name}. "
                "Skipping processing."
            )


def run(input_images: list[str]) -> int:
    print("Running aggregation of tiles")
    print("Image\t" + "\t".join(AmfConfig.human_readable_header()))
    try:
        if AmfConfig.get("colonisation_type") != "am":
            AmfLog.error(
                f"Only support AM Colonisation for converting tile size, not "
                f"{AmfConfig.get('colonisation_type')}. Cancel operation.",
                AmfLog.ERR_INVALID_DATA,
            )

        for path in input_images:
            tile_size = initialize_size(path)
            convert_tile_size(path, tile_size)

    except Exception as e:
        print(f"Unable to process file: {path}", e)
        return 500

    return 200
