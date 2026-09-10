import csv
import io
import json
import os
import re
from datetime import datetime
from typing import Any, cast
from zipfile import ZipFile

import pandas as pd
import psycopg2
from fastapi import HTTPException, Response
from loguru import logger

import amf.helper.config as AmfConfig
from amf.helper.api_objects import AnnotationValues, PredictionValues

# ---------------------------------------------------------------------------
# Local-file (no-DB) mode helpers, shared by both the batch importer and the
# manual-review fetch/save/check-entries/get-images functions below. Reads
# search both the configured Image Directory and Output Directory; writes
# always go to Output Directory. Multiple timestamped versions of the same
# image are expected to coexist (see _match_prediction_annotation_filename,
# defined further down) - nothing here collapses them to "the latest one".
# ---------------------------------------------------------------------------


def _local_search_dirs() -> list[str]:
    dirs = []
    for d in (AmfConfig.get("image_directory"), AmfConfig.get("outdir")):
        if d and os.path.isdir(d) and d not in dirs:
            dirs.append(d)
    return dirs


def _list_local_files_for_image(
    image_name: str, file_type: str
) -> list[dict[str, Any]]:
    """
    Finds every local {image_name}_*_cnn_1_{file_type}.csv across the
    configured Image Directory and Output Directory. One entry per file -
    multiple timestamped versions are expected to coexist.
    """
    entries = []
    for directory in _local_search_dirs():
        for filename in os.listdir(directory):
            match = _match_prediction_annotation_filename(filename)
            if not match or match["name"] != image_name or match["type"] != file_type:
                continue
            full_path = os.path.join(directory, filename)
            mtime = datetime.fromtimestamp(os.path.getmtime(full_path))
            timestamp = _parse_filename_timestamp(match["timestamp"])
            entries.append(
                {
                    "id": filename,
                    "path": full_path,
                    "directory": directory,
                    "timestamp": timestamp if timestamp is not None else mtime,
                    "updatedAt": mtime,
                }
            )
    return entries


def _find_local_file(id_: str) -> str | None:
    """Locates a specific local CSV by filename across the search dirs."""
    for directory in _local_search_dirs():
        candidate = os.path.join(directory, id_)
        if os.path.isfile(candidate):
            return candidate
    return None


def fetch_items_local(
    id_: str, type_: str, colonisation_type: str
) -> dict[str, list[Any]]:
    path = _find_local_file(id_)
    if path is None:
        return {}

    df = pd.read_csv(path)
    class_headers = AmfConfig.HEADERS[colonisation_type]
    builder = (
        _build_annotation_row
        if type_.lower() == "annotations"
        else _build_prediction_row
    )
    return {id_: [builder(row, class_headers) for _, row in df.iterrows()]}


def check_entries_for_image_local(
    image_name: str, colonisation_type: str
) -> dict[str, dict[str, Any]]:
    predictions = {
        e["id"]: e for e in _list_local_files_for_image(image_name, "predictions")
    }
    annotations = {
        e["id"]: e for e in _list_local_files_for_image(image_name, "annotations")
    }

    output: dict[str, dict[str, Any]] = {}
    for id_, entry in {**predictions, **annotations}.items():
        tile_edge = _read_tile_edge_sidecar(entry["directory"], image_name)
        output[id_] = {
            "timestamp": entry["timestamp"],
            "enabled": False,  # no cross-version "active" concept locally
            "tileEdge": tile_edge,
            "updatedAt": entry["updatedAt"],
            "cnn1_predictions_exist": id_ in predictions,
            "cnn1_annotations_exist": id_ in annotations,
        }
    return output


def parse_tags(value: Any) -> list[str]:
    """
    Splits a Tags cell into its individual sub-tags. Tolerates None/NaN and
    blank cells (older CSVs have no Tags column at all), trims whitespace,
    drops empties and de-duplicates while preserving order.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []

    tags: list[str] = []
    for part in text.split(AmfConfig.TAG_DELIMITER):
        tag = part.strip()
        if tag and tag not in tags:
            tags.append(tag)
    return tags


def format_tags(tags: list[str] | None) -> str:
    """
    Packs sub-tags back into a single delimited cell. Inverse of parse_tags;
    an empty list becomes an empty cell rather than a stray delimiter.
    """
    return AmfConfig.TAG_DELIMITER.join(
        parse_tags(AmfConfig.TAG_DELIMITER.join(tags or []))
    )


def save_to_local(values: "AnnotationValues | PredictionValues", type_: str) -> str:
    """
    Writes annotations/predictions to a local CSV instead of the database.
    Editing an existing identifier (values.imageReferenceId, a filename)
    overwrites that exact file; otherwise a fresh {image}_{now}_cnn_1_{type}
    .csv is written to Output Directory, and no other version is touched.
    Returns the resulting filename.
    """
    outdir = AmfConfig.get("outdir")
    if not outdir:
        raise HTTPException(
            status_code=400, detail="Output Directory is not configured"
        )
    os.makedirs(outdir, exist_ok=True)

    image_name = values.fileName
    class_headers = AmfConfig.HEADERS[values.colonisationType]

    editing_existing = isinstance(values.imageReferenceId, str)
    target_path = None
    if editing_existing:
        existing_path = _find_local_file(cast(str, values.imageReferenceId))
        if existing_path is not None:
            # Guard against a reused/stale/wrong-type id silently
            # overwriting the wrong file (e.g. an annotations id derived
            # from a predictions filename) - only treat this as "editing an
            # existing entry" if that entry is actually the same type we're
            # saving now. Otherwise fall through to creating a fresh file.
            existing_match = _match_prediction_annotation_filename(
                os.path.basename(existing_path)
            )
            if existing_match is not None and existing_match["type"] == type_:
                target_path = existing_path

    if target_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_path = os.path.join(
            outdir, f"{image_name}_{timestamp}_cnn_1_{type_}.csv"
        )

    if type_ == "annotations":
        columns = ["row", "col"] + class_headers + AmfConfig.ANNOTATION_EXTRA_COLUMNS
    else:
        columns = ["row", "col"] + class_headers + ["ContextualLabel"]

    normalized_rows = []
    for row in values.cnnOneValues or []:
        padded = list(row) + [None] * (len(columns) - len(row))
        normalized_rows.append(padded[: len(columns)])

    pd.DataFrame(normalized_rows, columns=columns).to_csv(target_path, index=False)

    settings_path = os.path.join(
        os.path.dirname(target_path), f"{image_name}_settings.json"
    )
    with open(settings_path, "w") as f:
        json.dump({"tile_edge": values.tileEdge}, f)

    return os.path.basename(target_path)


def get_images_local() -> list[tuple[str]]:
    names = set()
    for directory in _local_search_dirs():
        for filename in os.listdir(directory):
            match = _match_prediction_annotation_filename(filename)
            if match:
                names.add(match["name"])
    return [(n,) for n in sorted(names)]


def delete_local_entry(id_: str) -> None:
    path = _find_local_file(id_)
    if path is not None:
        os.remove(path)


def fetch_items(
    crsr: psycopg2.extensions.cursor,
    name: str,
    id_: str,
    type_: str,
    cnn: str,
    colonisation_type: str,
) -> dict[str, list[Any]]:
    """
    Fetches items from the database (or, when Use Local Database is off, from
    local CSV files) based on provided parameters.

    This function retrieves item data corresponding to either a name or an ID,
    where a preference is given to the ID if both are provided. If neither
    name nor ID is provided, an HTTPException is raised.

    Parameters:
    crsr (cursor): The database cursor used to execute SQL queries.
    name (str): The name reference used to fetch image IDs; can be empty.
    id_ (str): The specific ID reference for fetching item data; can be empty.
    type_ (str): A string that represents predictions or annotations.
    cnn (str): A string that represents the CNN.
    colonisation_type (str): A string that represents am or erm

    Returns:
    dict: A dictionary where keys are image IDs and values are lists of item data
          corresponding to those IDs, excluding the image reference.

    Raises:
    HTTPException: If both name and id are empty, or if the parameters do not
                   meet the expected input criteria.
    """
    if name == "" and id_ == "":
        raise HTTPException(
            status_code=400, detail="Name & id parameters are empty, please populate"
        )

    if name != "" and id_ != "":
        logger.info(
            "Both name and ID provided, name ignored in favour of more specific ID"
        )

    if not AmfConfig.get("use_db"):
        if id_ == "":
            # Local mode has no name->id table to resolve through; the
            # caller (fetch-by-id routes) is expected to pass the filename.
            return {}
        return fetch_items_local(id_, type_, colonisation_type)

    values = {}
    image_ids = []
    if id_ == "":
        fetch_image_id = "SELECT Id FROM ImageReference WHERE FileNameReference=%s"
        crsr.execute(fetch_image_id, (name,))
        image_ids = [row[0] for row in crsr.fetchall()]
    else:
        image_ids.append(id_)

    fetch_values_query = (
        f"SELECT * FROM Cnn{cnn}{type_}{colonisation_type} WHERE ImageReferenceId=%s"
    )
    for id_ in image_ids:
        crsr.execute(fetch_values_query, (id_,))
        preds = crsr.fetchall()
        out = []
        # Remove image reference
        for x in preds:
            out.append(x[1:])
        values[id_] = out

    return values


def set_to_enabled_in_db(crsr: psycopg2.extensions.cursor, id_: int) -> None:
    """
    Sets the 'Enabled' status of an image reference to true in the database for a given
    ID.

    This function updates the 'Enabled' field of the image reference to true for
    the specified ID and then disables all other image references that have the same
    file name. The function returns the file name reference of the updated image.

    Parameters:
    crsr (cursor): The database cursor used to execute SQL queries.
    id_ (int or str): The unique identifier of the image reference to be updated.

    Returns:
    str: The file name reference of the updated image.

    Raises:
    Exception: May raise exceptions related to database operations
                (e.g., if the ID does not exist).
    """
    update_enabled_query = (
        "UPDATE imageReference "
        "SET Enabled = true "
        "WHERE Id = %s "
        "RETURNING filenamereference"
    )
    crsr.execute(update_enabled_query, (id_,))
    file_name_reference = crsr.fetchone()[0]
    set_enabled_query = (
        "UPDATE imageReference "
        "SET Enabled = false "
        "WHERE FileNameReference=%s "
        "AND Id <> %s"
    )
    crsr.execute(
        set_enabled_query,
        (
            file_name_reference,
            id_,
        ),
    )


def _insert_image_reference(
    crsr: psycopg2.extensions.cursor, values: "AnnotationValues | PredictionValues"
) -> int:
    """
    Creates the ImageReference row a new annotation/prediction set hangs off,
    returning its id. When the caller supplies an uploadTimestamp (the batch
    folder importer, which recovers it from the CSV's filename) it is written
    through, so imported versions keep their original dates rather than all
    clustering at import time. Otherwise the column keeps its
    CURRENT_TIMESTAMP default.
    """
    if values.uploadTimestamp is not None:
        insert_id_query = (
            "INSERT INTO imagereference"
            "(filenamereference, TileEdge, Enabled, UploadTimestamp) "
            "VALUES (%s, %s, false, %s) "
            "RETURNING id"
        )
        params: tuple[Any, ...] = (
            values.fileName,
            values.tileEdge,
            values.uploadTimestamp,
        )
    else:
        insert_id_query = (
            "INSERT INTO imagereference(filenamereference, TileEdge, Enabled) "
            "VALUES (%s, %s, false) "
            "RETURNING id"
        )
        params = (values.fileName, values.tileEdge)

    crsr.execute(insert_id_query, params)
    return cast(int, crsr.fetchone()[0])


def save_annotations_to_db(
    crsr: psycopg2.extensions.cursor, values: AnnotationValues
) -> int | str:
    """
    Saves annotation data to the database associated with a specific image reference.

    This function first checks if a provided image reference ID exists in the database.
    If it does not exist, it inserts a new record into the `imagereference` table.

    If the annotations are to be enabled, it updates the `Enabled` field for the
    relevant image references. The function also handles the insertion of CNN
    annotations for both Type One and Type Two annotations, replacing any existing data
    for the specified image reference ID.

    Parameters:
    crsr (cursor): The database cursor used to execute SQL queries.
    values (AnnotationValues): An object containing the details of the annotations to be
        saved, including the image reference ID, file name, tile edge, enable status,
        and CNN annotation values.

    Returns:
    int: The ID of the saved image reference.

    Raises:
    Exception: May raise exceptions related to database operations (e.g., SQL errors or
               constraints violations).
    """
    if not AmfConfig.get("use_db"):
        return save_to_local(values, "annotations")

    # Check if image reference ID exists in DB
    exists = False
    if values.imageReferenceId is not None:
        # Check that given ID already has an entry in the database
        check_image_id = "SELECT EXISTS(SELECT 1 FROM imagereference WHERE id=%s)"
        crsr.execute(check_image_id, (values.imageReferenceId,))
        exists = crsr.fetchone()[0]

    image_id = 0
    if exists:
        image_id = cast(int, values.imageReferenceId)  # guaranteed to be not None here
        update_time_query = (
            "UPDATE ImageReference SET UpdatedAt = CURRENT_TIMESTAMP WHERE Id=%s"
        )
        crsr.execute(
            update_time_query,
            (image_id,),
        )
        tile_edge = get_tile_edge(crsr, image_id)
        if tile_edge[0] != values.tileEdge:
            logger.info(
                f"Tile edge does not match, update from {tile_edge[0]} to "
                f"{values.tileEdge}"
            )
            update_tile_edge_query = "UPDATE ImageReference SET TileEdge=%s WHERE Id=%s"
            crsr.execute(
                update_tile_edge_query,
                (
                    values.tileEdge,
                    image_id,
                ),
            )
    else:
        # If not, then upload file name to DB
        image_id = _insert_image_reference(crsr, values)

    # If enabled, set all other images with same reference ID to false
    if values.enabled:
        set_to_enabled_in_db(crsr, image_id)

    # Replace all annotations
    if values.cnnOneValues and len(values.cnnOneValues) != 0:
        delete_query = (
            f"DELETE FROM cnn1annotations{values.colonisationType} "
            "WHERE imagereferenceid=%s"
        )
        crsr.execute(delete_query, (image_id,))

        if values.colonisationType == "am":
            insert_values_query = f"""
                INSERT INTO cnn1annotations{values.colonisationType}
                (
                    imagereferenceid, rownum, colnum, amcolonised, uncolonised,
                    background, unreadable, dse, hybrid, question, questioncomment,
                    tags
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            # Trailing fields are read by index and guarded by length, so
            # shorter legacy rows (e.g. convert.py's 8-wide output, or a
            # pre-Tags 9/10-wide row) still land in the right columns.
            for row in values.cnnOneValues:
                row_length = len(row)
                crsr.execute(
                    insert_values_query,
                    (
                        image_id,
                        row[0],
                        row[1],
                        row[2],
                        row[3],
                        row[4],
                        row[5],
                        row[6],
                        row[7],
                        row[8] if row_length >= 9 else 0,
                        row[9] if row_length >= 10 else None,
                        format_tags(parse_tags(row[10])) if row_length >= 11 else None,
                    ),
                )
        else:
            insert_values_query = f"""
                INSERT INTO cnn1annotations{values.colonisationType}
                (
                    imagereferenceid, rownum, colnum, BlueCoils, BrownCoils, TypeTwo,
                    Uncolonised, Background, MainRoot, Unreadable, DSE, HybridErm,
                    HybridDse, Question, QuestionComment, Tags
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
            """

            for row in values.cnnOneValues:
                row_length = len(row)
                crsr.execute(
                    insert_values_query,
                    (
                        image_id,
                        row[0],
                        row[1],
                        row[2],
                        row[3],
                        row[4],
                        row[5],
                        row[6],
                        row[7],
                        row[8],
                        row[9],
                        row[10],
                        row[11],
                        row[12] if row_length >= 13 else 0,
                        row[13] if row_length >= 14 else None,
                        format_tags(parse_tags(row[14])) if row_length >= 15 else None,
                    ),
                )

    logger.info(f"Saved Image Reference ID: {image_id}")
    return image_id


def save_predictions_to_db(
    crsr: psycopg2.extensions.cursor, values: PredictionValues
) -> None:
    """
    Saves prediction data to the database associated with a specific image reference.

    This function first checks if the provided image reference ID exists in the
    database.
    If it does not exist, it inserts a new record into the `imagereference` table.
    The function then handles the insertion of CNN prediction data, deleting any
    existing data for the specified image reference ID before inserting the new
    predictions.

    Parameters:
    crsr (cursor): The database cursor used to execute SQL queries.
    values (PredictionValues): An object containing the details of the predictions to be
        saved, including the image reference ID, file name, tile edge, colonisation
        type, and CNN prediction values.

    Returns:
    None: This function does not return any value but modifies the database
          according to the provided predictions.

    Raises:
    Exception: May raise exceptions related to database operations (e.g., SQL errors or
               constraint violations).
    """
    if not AmfConfig.get("use_db"):
        save_to_local(values, "predictions")
        return

    # Check if image reference ID exists in DB
    exists = False
    if values.imageReferenceId is not None:
        # Check that given ID already has an entry in the database
        check_image_id = "SELECT EXISTS(SELECT 1 FROM imagereference WHERE id=%s)"
        crsr.execute(check_image_id, (values.imageReferenceId,))
        exists = crsr.fetchone()[0]

    image_id = 0
    if exists:
        image_id = cast(int, values.imageReferenceId)  # guaranteed to be not None here
        update_time_query = (
            "UPDATE ImageReference SET UpdatedAt = CURRENT_TIMESTAMP WHERE Id=%s"
        )
        crsr.execute(
            update_time_query,
            (image_id,),
        )
    else:
        # If not, then upload file name to DB
        image_id = _insert_image_reference(crsr, values)

    # Then either take existing image ref ID or use new one and upload under it
    if values.cnnOneValues and len(values.cnnOneValues) != 0:
        delete_query = (
            f"DELETE FROM cnn1predictions{values.colonisationType} "
            "WHERE imagereferenceid=%s"
        )
        crsr.execute(delete_query, (image_id,))

        if values.colonisationType == "am":
            insert_values_query = f"""
                INSERT INTO cnn1predictions{values.colonisationType}
                (
                    imagereferenceid, rownum, colnum, amcolonised, uncolonised,
                    background, unreadable, dse, hybrid, contextuallabel
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            for row in values.cnnOneValues:
                crsr.execute(
                    insert_values_query,
                    (
                        image_id,
                        row[0],
                        row[1],
                        row[2],
                        row[3],
                        row[4],
                        row[5],
                        row[6],
                        row[7],
                        row[8],
                    ),
                )
        else:
            insert_values_query = f"""
                INSERT INTO cnn1predictions{values.colonisationType}
                (
                    imagereferenceid, rownum, colnum, BlueCoils, BrownCoils, TypeTwo,
                    Uncolonised, Background, MainRoot, Unreadable, DSE, HybridErm,
                    HybridDse, contextuallabel
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            for row in values.cnnOneValues:
                crsr.execute(
                    insert_values_query,
                    (
                        image_id,
                        row[0],
                        row[1],
                        row[2],
                        row[3],
                        row[4],
                        row[5],
                        row[6],
                        row[7],
                        row[8],
                        row[9],
                        row[10],
                        row[11],
                        row[12],
                    ),
                )


# Matches the exact filename convention `save.py`'s prediction_table() uses:
# {image_name}_{YYYYMMDD_HHMMSS}_cnn_1_{predictions|annotations}.csv
_TIMESTAMPED_FILENAME_RE = re.compile(
    r"^(?P<name>.+?)_(?P<timestamp>\d{8}_\d{6})_cnn_1_"
    r"(?P<type>predictions|annotations)\.csv$"
)
# Older releases stamped an isoformat()-derived timestamp instead, with the
# colons replaced by underscores (they're illegal in filenames) and optional
# microseconds:
# {image_name}_{YYYY-MM-DDTHH_MM_SS[.ffffff]}_cnn_1_{predictions|annotations}.csv
# Still supported so pre-existing files keep loading; note the Existing page's
# download button also produces this shape via browser colon-sanitisation.
_LEGACY_TIMESTAMPED_FILENAME_RE = re.compile(
    r"^(?P<name>.+?)_(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}(?:\.\d+)?)"
    r"_cnn_1_(?P<type>predictions|annotations)\.csv$"
)
# Looser fallback for hand-named files with no timestamp component.
_UNTIMESTAMPED_FILENAME_RE = re.compile(
    r"^(?P<name>.+?)_cnn_1_(?P<type>predictions|annotations)\.csv$"
)

# Tried in order against whatever `timestamp` group the matchers above return.
_FILENAME_TIMESTAMP_FORMATS = (
    "%Y%m%d_%H%M%S",
    "%Y-%m-%dT%H_%M_%S.%f",
    "%Y-%m-%dT%H_%M_%S",
)


def _match_prediction_annotation_filename(filename: str) -> dict[str, Any] | None:
    for pattern in (_TIMESTAMPED_FILENAME_RE, _LEGACY_TIMESTAMPED_FILENAME_RE):
        match = pattern.match(filename)
        if match:
            return match.groupdict()
    match = _UNTIMESTAMPED_FILENAME_RE.match(filename)
    if match:
        groups = match.groupdict()
        groups["timestamp"] = None
        return groups
    return None


def _parse_filename_timestamp(timestamp: str | None) -> datetime | None:
    """
    Parses the timestamp token out of a prediction/annotation filename,
    accepting both the current and the legacy conventions. Returns None if
    there is no timestamp or it doesn't match any known format.
    """
    if not timestamp:
        return None
    for fmt in _FILENAME_TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(timestamp, fmt)
        except ValueError:
            continue
    return None


def _read_tile_edge_sidecar(folder_path: str, image_name: str) -> int:
    """
    Reads the `{image_name}_settings.json` sidecar `save.py`/`load.py` already
    read/write, falling back to the currently configured tile_edge if it's
    absent or unreadable.
    """
    settings_path = os.path.join(folder_path, f"{image_name}_settings.json")
    if os.path.isfile(settings_path):
        try:
            with open(settings_path) as f:
                settings = json.load(f)
            if "tile_edge" in settings:
                return cast(int, settings["tile_edge"])
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to read settings sidecar {settings_path}: {e}")
    return cast(int, AmfConfig.get("tile_edge"))


def _validate_import_columns(
    headers: list[str], mandatory: list[str], optional: list[str]
) -> None:
    missing = [c for c in mandatory if c not in headers]
    if missing:
        raise ValueError(f"Missing mandatory column(s): {', '.join(missing)}.")

    allowed = set(mandatory) | set(optional)
    extra = [c for c in headers if c not in allowed]
    if extra:
        raise ValueError(f"Found extra column(s): {', '.join(extra)}.")


def _build_annotation_row(row: "pd.Series[Any]", class_headers: list[str]) -> list[Any]:
    """
    Mirrors AnnotationsAndPredictionsContainer.jsx's row layout: row, col,
    one entry per class, then Question (the "?" one-hot slot),
    QuestionComment and Tags. Emitted at full fixed width so the two text
    fields sit at stable indices.
    """
    question_idx = 2 + len(class_headers)
    comment_idx = question_idx + 1
    tags_idx = comment_idx + 1

    out: list[Any] = [0] * (tags_idx + 1)
    out[comment_idx] = ""
    out[tags_idx] = ""

    out[0] = int(row["row"])
    out[1] = int(row["col"])
    for i, header in enumerate(class_headers):
        out[2 + i] = int(row[header])
    if AmfConfig.QUESTION_COLUMN in row.index and not pd.isna(
        row[AmfConfig.QUESTION_COLUMN]
    ):
        out[question_idx] = int(row[AmfConfig.QUESTION_COLUMN])
    if AmfConfig.QUESTION_COMMENT_COLUMN in row.index and not pd.isna(
        row[AmfConfig.QUESTION_COMMENT_COLUMN]
    ):
        out[comment_idx] = row[AmfConfig.QUESTION_COMMENT_COLUMN]
    # Absent in older CSVs, which simply yields no tags for that tile.
    if AmfConfig.TAGS_COLUMN in row.index:
        out[tags_idx] = format_tags(parse_tags(row[AmfConfig.TAGS_COLUMN]))
    return out


def _build_prediction_row(row: "pd.Series[Any]", class_headers: list[str]) -> list[Any]:
    """
    Mirrors AmfUpload.jsx's row layout: row, col, one entry per class, then
    ContextualLabel.
    """
    out: list[Any] = [0] * (2 + len(class_headers) + 1)
    out[0] = int(row["row"])
    out[1] = int(row["col"])
    for i, header in enumerate(class_headers):
        out[2 + i] = float(row[header])
    label_idx = 2 + len(class_headers)
    if "ContextualLabel" in row.index and not pd.isna(row["ContextualLabel"]):
        out[label_idx] = row["ContextualLabel"]
    return out


def import_predictions_and_annotations_from_folder(
    crsr: psycopg2.extensions.cursor, folder_path: str, colonisation_type: str
) -> dict[str, Any]:
    """
    Batch-imports every predictions/annotations CSV found directly under
    `folder_path` into the database, using the same filename convention and
    column layout `save.py`'s CSV output (and AmfUpload.jsx's single-file
    upload) already use. Never aborts the whole batch on one bad file.
    """
    if not os.path.isdir(folder_path):
        raise HTTPException(status_code=400, detail=f"Folder not found: {folder_path}")

    class_headers = AmfConfig.HEADERS[colonisation_type]
    imported: list[str] = []
    errors: list[dict[str, str]] = []

    for filename in sorted(os.listdir(folder_path)):
        match = _match_prediction_annotation_filename(filename)
        if match is None:
            continue

        image_name = match["name"]
        file_type = match["type"]
        # Recovered from the filename so imported versions keep their original
        # dates instead of all landing at import time. None for files with no
        # recognisable timestamp (those keep the DB's CURRENT_TIMESTAMP).
        upload_timestamp = _parse_filename_timestamp(match["timestamp"])

        try:
            df = pd.read_csv(os.path.join(folder_path, filename))

            mandatory_columns = ["row", "col"] + class_headers
            optional_columns = (
                AmfConfig.ANNOTATION_EXTRA_COLUMNS
                if file_type == "annotations"
                else ["ContextualLabel"]
            )
            _validate_import_columns(
                df.columns.tolist(), mandatory_columns, optional_columns
            )

            tile_edge = _read_tile_edge_sidecar(folder_path, image_name)

            if file_type == "annotations":
                cnn_one_values = [
                    _build_annotation_row(row, class_headers)
                    for _, row in df.iterrows()
                ]
                save_annotations_to_db(
                    crsr,
                    AnnotationValues(
                        fileName=image_name,
                        imageReferenceId=None,
                        cnnOneValues=cnn_one_values,
                        colonisationType=colonisation_type,
                        tileEdge=tile_edge,
                        uploadTimestamp=upload_timestamp,
                        enabled=True,
                    ),
                )
            else:
                cnn_one_values = [
                    _build_prediction_row(row, class_headers)
                    for _, row in df.iterrows()
                ]
                save_predictions_to_db(
                    crsr,
                    PredictionValues(
                        fileName=image_name,
                        imageReferenceId=None,
                        cnnOneValues=cnn_one_values,
                        colonisationType=colonisation_type,
                        tileEdge=tile_edge,
                        uploadTimestamp=upload_timestamp,
                    ),
                )

            imported.append(filename)
        except Exception as e:
            logger.warning(f"Failed to import {filename}: {e}")
            errors.append({"file": filename, "reason": str(e)})

    return {"imported": imported, "errors": errors}


def get_images(crsr: psycopg2.extensions.cursor) -> list[tuple[str]]:
    if not AmfConfig.get("use_db"):
        return get_images_local()

    get_all_images = "SELECT DISTINCT FileNameReference FROM imagereference"
    crsr.execute(get_all_images)
    images = cast(list[tuple[str]], crsr.fetchall())  # since FileNameReference is text

    return images


def get_most_recent_timestamp(
    crsr: psycopg2.extensions.cursor, image_name: str
) -> int | None:
    """
    Retrieves the ID of the most recently uploaded image based on the specified image
    name.

    This function queries the database for records in the `ImageReference` table that
    match the provided `image_name`. It sorts the results by upload timestamp and
    returns the ID of the most recent entry. If no images are found, it returns None.

    Parameters:
    crsr (cursor): The database cursor used to execute SQL queries.
    image_name (str): The name of the image for which to retrieve the most recent
        timestamp.

    Returns:
    int or None: The ID of the most recently uploaded image if found; otherwise, None.

    Raises:
    Exception: May raise exceptions related to database operations (e.g., SQL errors).
    """
    fetch_images = (
        "SELECT Id, UploadTimestamp FROM ImageReference WHERE FileNameReference=%s"
    )
    crsr.execute(fetch_images, (image_name,))
    images = crsr.fetchall()
    sorted_values = sorted(images, key=lambda x: x[1])
    if len(sorted_values) > 0:
        return cast(int, sorted_values[-1][0])  # since ID is integer
    else:
        return None


def get_enabled(crsr: psycopg2.extensions.cursor, image_name: str) -> int | None:
    """
    Retrieves the ID of an enabled image from the database based on the specified image
    name.

    This function queries the `ImageReference` table to check if there is an entry
    with the provided `image_name` that is marked as enabled. If an enabled image is
    found, its ID is returned. If no enabled images are present for the given name, the
    function retrieves the ID of the most recently uploaded image and sets that image to
    enabled before returning its ID.

    Parameters:
    crsr (cursor): The database cursor used to execute SQL queries.
    image_name (str): The name of the image for which to check for enabled status.

    Returns:
    int: The ID of the enabled image if found; otherwise, the ID of the most recent
        image.

    Raises:
    Exception: May raise exceptions related to database operations (e.g., SQL errors or
               constraints violations) if images cannot be fetched or processed.
    """
    fetch_images = (
        "SELECT Id FROM ImageReference WHERE (FileNameReference=%s AND Enabled=true)"
    )
    crsr.execute(fetch_images, (image_name,))
    images = crsr.fetchall()
    if len(images) > 0:
        logger.info(f"Image ID {images[0][0]} is enabled for image name {image_name}")
        return cast(int, images[0][0])  # since ID is int in database
    else:
        # If no annotations enabled, get most recent timestamp and set this to enabled
        logger.info(
            f"No enabled annotations for image {image_name}, return image with most "
            f"recent timestamp instead."
        )
        id_ = get_most_recent_timestamp(crsr, image_name)
        if id_ is not None:
            set_to_enabled_in_db(crsr, id_)
        return id_


def get_tile_edge(crsr: psycopg2.extensions.cursor, id_: int) -> tuple[int]:
    fetch_images = "SELECT TileEdge FROM ImageReference WHERE Id=%s"
    crsr.execute(fetch_images, (id_,))
    return cast(tuple[int], crsr.fetchone())  # since TileEdge is integer


def check_entries_for_image(
    crsr: psycopg2.extensions.cursor, image_name: str, colonisation_type: str
) -> dict[Any, dict[str, Any]]:
    if not AmfConfig.get("use_db"):
        return check_entries_for_image_local(image_name, colonisation_type)

    fetch_images = (
        "SELECT Id, UploadTimestamp, Enabled, TileEdge, UpdatedAt "
        "FROM ImageReference "
        "WHERE FileNameReference=%s"
    )
    crsr.execute(fetch_images, (image_name,))
    images = crsr.fetchall()

    return _get_existing_entries_for_image(crsr, colonisation_type, images)


def check_entries_for_id(
    crsr: psycopg2.extensions.cursor, id_: int | str, colonisation_type: str
) -> dict[Any, dict[str, Any]]:
    if not AmfConfig.get("use_db"):
        # Local mode has no name -> id table to resolve through; there's no
        # image name to key off here, so nothing to look up by bare id.
        return {}

    fetch_images = (
        "SELECT Id, UploadTimestamp, Enabled, TileEdge, UpdatedAt "
        "FROM ImageReference "
        "WHERE Id=%s"
    )
    crsr.execute(fetch_images, (id_,))
    images = crsr.fetchall()

    return _get_existing_entries_for_image(crsr, colonisation_type, images)


def _get_existing_entries_for_image(
    crsr: psycopg2.extensions.cursor,
    colonisation_type: str,
    images: list[tuple[int, datetime, bool, int, datetime]],
) -> dict[int, dict[str, Any]]:
    output: dict[int, dict[str, Any]] = {}

    for value in images:
        id_ = value[0]
        timestamp = value[1]
        enabled = value[2]
        tileEdge = value[3]
        updatedAt = value[4]
        cnn1_predictions_exist_query = f"""
            SELECT EXISTS(
                SELECT 1
                FROM cnn1predictions{colonisation_type}
                WHERE ImageReferenceId=%s
            )
        """
        cnn1_annotations_exist_query = f"""
            SELECT EXISTS(
                SELECT 1
                FROM cnn1annotations{colonisation_type}
                WHERE ImageReferenceId=%s
            )
        """

        crsr.execute(cnn1_predictions_exist_query, (id_,))
        cnn1_predictions_exist = crsr.fetchone()[0]

        crsr.execute(cnn1_annotations_exist_query, (id_,))
        cnn1_annotations_exist = crsr.fetchone()[0]

        if cnn1_annotations_exist or cnn1_predictions_exist:
            output[id_] = {
                "timestamp": timestamp,
                "enabled": enabled,
                "tileEdge": tileEdge,
                "updatedAt": updatedAt,
                "cnn1_predictions_exist": cnn1_predictions_exist,
                "cnn1_annotations_exist": cnn1_annotations_exist,
            }

    return output


def download_entries_as_csv(
    crsr: psycopg2.extensions.cursor, id_: int, type_: str, colonisation_type: str
) -> str:
    output = io.StringIO()

    fetch_values_query = (
        f"SELECT * FROM Cnn1{type_}{colonisation_type} WHERE ImageReferenceId=%s"
    )
    crsr.execute(fetch_values_query, (id_,))
    entries = crsr.fetchall()
    out = []
    for x in entries:
        # Remove image reference
        out.append(x[1:])

    # Write a CSV string
    output = io.StringIO()
    writer = csv.writer(output)

    # Derived from HEADERS rather than repeated as literals: the body rows come
    # from SELECT *, so a header list that drifts narrower than the table would
    # shift every column when the CSV is read back.
    class_headers = AmfConfig.HEADERS[colonisation_type]
    if type_.lower() == "annotations":
        trailing_columns = AmfConfig.ANNOTATION_EXTRA_COLUMNS
    else:
        trailing_columns = ["ContextualLabel"]
    writer.writerow(["row", "col"] + class_headers + trailing_columns)

    for row in out:
        writer.writerow([x for x in row])

    csv_string = output.getvalue()
    output.close()

    return csv_string


def delete_image(crsr: psycopg2, id_: int | str) -> None:
    if not AmfConfig.get("use_db"):
        delete_local_entry(cast(str, id_))
        return

    delete_query = "DELETE FROM ImageReference WHERE Id=%s"
    crsr.execute(delete_query, (id_,))


def zip_files_for_transit(images: dict[str, bytes]) -> Response:
    zip_filename = "image-tiles.zip"

    s = io.BytesIO()
    zf = ZipFile(s, "w")

    for key, value in images.items():
        zf.writestr(key, value)

    # Must close zip for all contents to be written
    s.seek(0)
    zf.close()

    # Grab ZIP file from in-memory, make response with correct MIME-type
    # application/x-zip-compressed another option
    resp = Response(
        s.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment;filename={zip_filename}"},
    )

    return resp


def _parse_setting(value: str, value_type: str) -> Any:
    if value_type in ("integer", "float") and (value is None or value == ""):
        return None
    if value_type == "integer":
        return int(value)
    elif value_type == "boolean":
        return value.lower() == "true"
    elif value_type == "float":
        return float(value)
    return value


def get_all_settings_from_db(crsr: psycopg2.extensions.cursor) -> dict[str, Any]:
    query = "SELECT key, value, value_type FROM settings"
    crsr.execute(query)
    out = crsr.fetchall()

    settings = {}

    for value in out:
        settings[value[0]] = _parse_setting(value[1], value[2])

    return settings


TAG_PALETTE_SETTING_KEY = "tagPalette"


def get_tag_palette(crsr: psycopg2.extensions.cursor) -> dict[str, str]:
    """
    Reads the user's tag -> colour map from its single Settings row (JSON in a
    'string' value, since the value_type CHECK allows only scalars). Tag names
    themselves are always recovered from the annotation data, so a missing or
    unreadable palette means every tag falls back to a default colour.
    """
    query = "SELECT value FROM settings WHERE key=%s"
    crsr.execute(query, (TAG_PALETTE_SETTING_KEY,))
    out = crsr.fetchall()
    if not out or not out[0][0]:
        return {}
    try:
        palette = json.loads(out[0][0])
    except json.JSONDecodeError as e:
        logger.warning(f"Ignoring unreadable tag palette: {e}")
        return {}
    if not isinstance(palette, dict):
        return {}
    return {str(k): str(v) for k, v in palette.items()}


def save_tag_palette(crsr: psycopg2.extensions.cursor, palette: dict[str, str]) -> None:
    """
    Replaces the stored tag -> colour map.
    """
    query = """
        INSERT INTO settings (key, value_type, value, default_value)
        VALUES (%s, 'string', %s, '{}')
        ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
    """
    crsr.execute(query, (TAG_PALETTE_SETTING_KEY, json.dumps(palette)))


def change_setting_to_default_in_db(
    crsr: psycopg2.extensions.cursor, key: str
) -> dict[str, Any] | int:
    query = "SELECT default_value, value_type FROM settings WHERE key=%s"
    crsr.execute(query, (key,))
    out = crsr.fetchall()

    if len(out) > 0:
        default_value = out[0][0]
        value_type = out[0][1]

        set_query = "UPDATE settings SET value=%s WHERE key=%s"

        crsr.execute(set_query, (default_value, key))

        return {"key": key, "value": _parse_setting(default_value, value_type)}
    else:
        return 500


def update_setting_in_db(
    crsr: psycopg2.extensions.cursor, key: str, value: str
) -> None:
    set_query = "UPDATE settings SET value=%s WHERE key=%s"

    crsr.execute(set_query, (value, key))
