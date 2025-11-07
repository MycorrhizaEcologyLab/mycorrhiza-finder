import io
import os
import re

import pandas as pd

import amfinder_config as AmfConfig
import amfinder_log as AmfLog
from api_objects import AnnotationValues
from api_utils import (
    check_entries_for_id,
    download_entries_as_csv,
    get_enabled,
)
from db_config import connect


def collapse_annots(annotation_data):
    """
    Collapse annotations by removing various classes dependent on colonisation type.
    """
    annotation_data = pd.read_csv(
        annotation_data
    )  # Creates a pandas Dataframe from the annotations read from zip file.

    if "Question" in annotation_data.columns:
        num_questions = annotation_data["Question"].sum()
        if num_questions > 0:
            raise Exception(
                f"Cannot carry out tile conversion if questions exist, currently there are {num_questions} question(s)"
            )

        # Drop questions column
        annotation_data.drop("Question", axis=1, inplace=True, errors="ignore")

    if "QuestionComment" in annotation_data.columns:
        # Drop question comment column
        annotation_data.drop("QuestionComment", axis=1, inplace=True, errors="ignore")

    collapsed_annots = []

    colonisation_type = AmfConfig.get("colonisation_type")

    if colonisation_type == "am":
        if "AMColonised" not in annotation_data.columns:
            raise Exception(
                "CSV does not include correct columns for AM colonisation, fail"
            )
        # Collapse AM to five classes (removing Hybrid)
        for _, row in annotation_data.iterrows():
            this_result = {
                "row": row["row"],
                "col": row["col"],
                "AMColonised": 0,
                "Uncolonised": 0,
                "Background": 0,
                "Unreadable": 0,
                "DSE": 0,
            }

            if row["AMColonised"] == 1:
                this_result["AMColonised"] = 1

            if row["Uncolonised"] == 1:
                this_result["Uncolonised"] = 1

            if row["Background"] == 1:
                this_result["Background"] = 1

            if row["Unreadable"] == 1:
                this_result["Unreadable"] = 1

            if row["DSE"] == 1:
                this_result["DSE"] = 1

            if row["Hybrid"] == 1:
                this_result["AMColonised"] = 1

            collapsed_annots.append(this_result)
    else:
        # Collapse ErM to 7 classes, merging blue coils, brown coils, type two and hybrid ErM to ErMColonised
        for _, row in annotation_data.iterrows():
            if "BlueCoils" not in annotation_data.columns:
                raise Exception(
                    "CSV does not include correct columns for ErM colonisation, fail",
                )

            this_result = {
                "row": row["row"],
                "col": row["col"],
                "ErMColonised": 0,
                "Uncolonised": 0,
                "Background": 0,
                "MainRoot": 0,
                "Unreadable": 0,
                "DSE": 0,
                "HybridDse": 0,
            }

            if row["BlueCoils"] == 1:
                this_result["ErMColonised"] = 1

            if row["BrownCoils"] == 1:
                this_result["ErMColonised"] = 1

            if row["TypeTwo"] == 1:
                this_result["ErMColonised"] = 1

            if row["Uncolonised"] == 1:
                this_result["Uncolonised"] = 1

            if row["Background"] == 1:
                this_result["Background"] = 1

            if row["MainRoot"] == 1:
                this_result["MainRoot"] = 1

            if row["Unreadable"] == 1:
                this_result["Unreadable"] = 1

            if row["DSE"] == 1:
                this_result["DSE"] = 1

            if row["HybridErm"] == 1:
                this_result["ErMColonised"] = 1

            if row["HybridDse"] == 1:
                this_result["HybridDse"] = 1

            collapsed_annots.append(this_result)

    # Generate the final table.
    return pd.DataFrame.from_dict(collapsed_annots)


def collapse_classes(path):
    # Write back to the annotations file

    image_name = os.path.splitext(os.path.basename(path))[0]

    if AmfConfig.get("use_db"):
        # TODO DB schema would need to be updated to save annotations to DB.
        AmfLog.error(
            "DB access for class collapsing disabled as DB schema would need to be updated",
            AmfLog.ERR_INVALID_DATA,
        )

        colonisation_type = AmfConfig.get("colonisation_type")

        conn = connect("amf")
        with conn, conn.cursor() as crsr:
            id = get_enabled(crsr, image_name)

            if id == None:
                AmfLog.warning(
                    f"Skipping {path} as no entries are saved in DB for this image"
                )
                return

            existing_entries = check_entries_for_id(crsr, id, colonisation_type)

            if existing_entries[id]["cnn1_annotations_exist"]:
                csv = download_entries_as_csv(
                    crsr, id, "Annotations", colonisation_type
                )
                out = collapse_annots(io.StringIO(csv))
                cnn1results = out.values.tolist()
                values = AnnotationValues(
                    imageReferenceId=id,
                    colonisationType=colonisation_type,
                    cnnOneValues=cnn1results,
                )
                # save_annotations_to_db(crsr, values)
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
            out = collapse_annots(annotation_path)
            out_annotation_path = os.path.join(directory, matching_annotations[0])
            out.to_csv(out_annotation_path, index=False, mode="w")

            AmfLog.info(f"Saved collapsed annotations to {out_annotation_path}")
        else:
            AmfLog.warning(
                f"Multiple annotation files found for {image_name}. Skipping processing."
            )


def run(input_images):
    print("Running collapse of tiles")
    print("Image\t" + "\t".join(AmfConfig.human_readable_header()))

    errors = []

    for path in input_images:
        try:
            collapse_classes(path)

        except Exception as e:
            print(f"Unable to process file: {path}", e)
            errors.append(path)

    if len(errors) > 0:
        print(f"Conversion failed for {errors}")

    return 200
