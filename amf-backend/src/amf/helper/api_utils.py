"""Functions for interacting with the database for API calls."""

import csv
import io
import warnings
import zipfile
from datetime import datetime
from typing import Any, cast
from zipfile import ZipFile

import psycopg2
from fastapi import HTTPException, Response
from loguru import logger

from amf.helper.api_objects import AnnotationValues, PredictionValues


def fetch_items(
    crsr: psycopg2.extensions.cursor,
    name: str,
    id_: str,
    type_: str,
    cnn: str,
    colonisation_type: str,
) -> dict[str, list[Any]]:
    """Fetch items from the database for a single image.

    This function retrieves item data corresponding to either a name or an ID,
    where a preference is given to the ID if both are provided. If neither
    name nor ID is provided, an HTTPException is raised.

    Args:
        crsr: Database cursor used to execute SQL queries.
        name: Name reference used to fetch image IDs; can be empty if id_ provided.
        id_: Image ID; can be empty if name provided.
        type_: "predictions" or "annotations".
        cnn: Must be "1". Previously allowed "2" to indicate CNN1 or CNN2.
        colonisation_type: "am" or "erm" (case insensitive).

    Returns: Dictionary where keys are image IDs and values are lists of item data
        corresponding to those IDs, excluding the image reference.

    Raises:
        HTTPException: If both name and id_ are empty, or if the parameters do not
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
    """Set the enabled status of a given image to True.

    This function updates the 'Enabled' field of the image reference to true for
    the specified ID and then disables all other image references that have the same
    file name. The function returns the file name reference of the updated image.

    Args:
        crsr: Database cursor used to execute SQL queries.
        id_: Unique identifier of the image reference to be updated.

    Returns: File name reference of the updated image.
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


def save_annotations_to_db(
    crsr: psycopg2.extensions.cursor, values: AnnotationValues
) -> int:
    """Save annotations to the database for a single image.

    This function first checks if a provided image reference ID exists in the database.
    If it does not exist, it inserts a new record into the `imagereference` table.

    If the annotations are to be enabled, it updates the `Enabled` field for the
    relevant image references. The function also handles the insertion of CNN
    annotations for both Type One and Type Two annotations, replacing any existing data
    for the specified image reference ID.

    Args:
        crsr: Database cursor used to execute SQL queries.
        values: Object containing the details of the annotations to be saved, including
            image reference ID, file name, tile edge, enable status and CNN annotation
            values.

    Returns: ID of the saved image reference.
    """
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
        insert_id_query = (
            "INSERT INTO imagereference(filenamereference, TileEdge, Enabled) "
            "VALUES (%s, %s, false) "
            "RETURNING id"
        )
        crsr.execute(
            insert_id_query,
            (
                values.fileName,
                values.tileEdge,
            ),
        )
        image_id = crsr.fetchone()[0]

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
                    background, unreadable, dse, hybrid, question, questioncomment
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                        row[8] if row_length >= 9 else 0,
                        row[9] if row_length == 10 else None,
                    ),
                )
        else:
            insert_values_query = f"""
                INSERT INTO cnn1annotations{values.colonisationType}
                (
                    imagereferenceid, rownum, colnum, BlueCoils, BrownCoils, TypeTwo,
                    Uncolonised, Background, MainRoot, Unreadable, DSE, HybridErm,
                    HybridDse, Question, QuestionComment
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                        row[13] if row_length == 14 else None,
                    ),
                )

    logger.info(f"Saved Image Reference ID: {image_id}")
    return image_id


def save_predictions_to_db(
    crsr: psycopg2.extensions.cursor, values: PredictionValues
) -> None:
    """Save prediction data to the database for a single image.

    This function first checks if the provided image reference ID exists in the
    database.
    If it does not exist, it inserts a new record into the `imagereference` table.
    The function then handles the insertion of CNN prediction data, deleting any
    existing data for the specified image reference ID before inserting the new
    predictions.

    Args:
        crsr: Database cursor used to execute SQL queries.
        values: Object containing the details of the predictions to be saved, including
            image reference ID, file name, tile edge, colonisation type and CNN
            prediction values.
    """
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
        insert_id_query = """
            INSERT INTO imagereference
                (filenamereference, TileEdge, Enabled)
            VALUES (%s, %s, false)
            RETURNING id
        """
        crsr.execute(
            insert_id_query,
            (
                values.fileName,
                values.tileEdge,
            ),
        )
        image_id = crsr.fetchone()[0]

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


def get_images(crsr: psycopg2.extensions.cursor) -> list[tuple[str]]:
    """Get all image IDs from the database.

    Args:
        crsr: Database cursor used to execute SQL queries.

    Returns: List of singleton tuples containing file name references of all images.
    """
    get_all_images = "SELECT DISTINCT FileNameReference FROM imagereference"
    crsr.execute(get_all_images)
    return cast(list[tuple[str]], crsr.fetchall())  # since FileNameReference is text


def get_most_recent_timestamp(
    crsr: psycopg2.extensions.cursor, image_name: str
) -> int | None:
    """Retrieve ID of most recently uploaded image with the specified name.

    This function queries the database for records in the `ImageReference` table that
    match the provided `image_name`. It sorts the results by upload timestamp and
    returns the ID of the most recent entry. If no images are found, it returns None.

    Args:
        crsr: Database cursor used to execute SQL queries.
        image_name: Name of the image for which to retrieve the most recent timestamp.

    Returns: ID of the most recently uploaded image if found; otherwise, None.
    """
    fetch_images = (
        "SELECT Id, UploadTimestamp FROM ImageReference WHERE FileNameReference=%s"
    )
    crsr.execute(fetch_images, (image_name,))
    images = crsr.fetchall()
    sorted_values = sorted(images, key=lambda x: x[1])
    if len(sorted_values) > 0:
        return cast(int, sorted_values[-1][0])  # since ID is integer
    return None


def get_enabled(crsr: psycopg2.extensions.cursor, image_name: str) -> int | None:
    """Get the ID of an enabled image, or the most recent image.

    This function queries the `ImageReference` table to check if there is an entry
    with the provided `image_name` that is marked as enabled. If an enabled image is
    found, its ID is returned. If no enabled images are present for the given name, the
    function retrieves the ID of the most recently uploaded image and sets that image to
    enabled before returning its ID.

    Args:
        crsr: Database cursor used to execute SQL queries.
        image_name: Name of the image for which to check for enabled status.

    Returns: ID of the enabled image if found; otherwise, the ID of the most recent
        image.
    """
    fetch_images = (
        "SELECT Id FROM ImageReference WHERE (FileNameReference=%s AND Enabled=true)"
    )
    crsr.execute(fetch_images, (image_name,))
    images = crsr.fetchall()
    if len(images) > 0:
        logger.info(f"Image ID {images[0][0]} is enabled for image name {image_name}")
        return cast(int, images[0][0])  # since ID is int in database
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
    """Return tile edge length for a given image ID.

    Args:
        crsr: Database cursor used to execute SQL queries.
        id_: ID of the image for which to retrieve the tile edge length.

    Returns: Singleton tuple containing the tile edge length.
    """
    fetch_images = "SELECT TileEdge FROM ImageReference WHERE Id=%s"
    crsr.execute(fetch_images, (id_,))
    return cast(tuple[int], crsr.fetchone())  # since TileEdge is integer


def check_entries_for_image(
    crsr: psycopg2.extensions.cursor, image_name: str, colonisation_type: str
) -> dict[int, dict[str, Any]]:
    """Get all existing image IDs and properties for a given image name.

    Args:
        crsr: Database cursor used to execute SQL queries.
        image_name: Name of the image for which to check entries.
        colonisation_type: "am" or "erm" (case insensitive).

    Returns: Nested dictionary where keys are image IDs and values are dictionaries of
        properties for those IDs.
    """
    fetch_images = (
        "SELECT Id, UploadTimestamp, Enabled, TileEdge, UpdatedAt "
        "FROM ImageReference "
        "WHERE FileNameReference=%s"
    )
    crsr.execute(fetch_images, (image_name,))
    images = crsr.fetchall()

    return _get_existing_entries_for_image(crsr, colonisation_type, images)


def check_entries_for_id(
    crsr: psycopg2.extensions.cursor, id_: int, colonisation_type: str
) -> dict[int, dict[str, Any]]:
    """Get image properties for a given ID.

    Args:
        crsr: Database cursor used to execute SQL queries.
        id_: ID of the image for which to check entries.
        colonisation_type: "am" or "erm" (case insensitive).

    Returns: Nested dictionary where outer key is image ID and value is dictionary of
        properties for that ID.
    """
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
    """Get image properties for a set of images.

    Spefiically, check if CNN1 predictions or annotations exist for each image.

    Args:
        crsr: Database cursor used to execute SQL queries.
        colonisation_type: "am" or "erm" (case insensitive).
        images: List of tuples containing known image properties: ID, upload timestamp,
            enabled flag, tile edge length, modified timestamp.

    Returns: Nested dictionary where outer key is image ID and value is dictionary of
        properties for that ID, including whether CNN1 predictions or annotations exist.
    """
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
    """Return annotations or predictions for a given image as a CSV string.

    Args:
        crsr: Database cursor used to execute SQL queries.
        id_: Image ID.
        type_: "predictions" or "annotations" (case insensitive).
        colonisation_type: "am" or "erm" (case insensitive).

    Returns: CSV string of the annotations or predictions for the given image ID.
    """
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

    if colonisation_type == "am":
        if type_.lower() == "annotations":
            writer.writerow(
                [
                    "row",
                    "col",
                    "AMColonised",
                    "Uncolonised",
                    "Background",
                    "Unreadable",
                    "DSE",
                    "Hybrid",
                    "Question",
                    "QuestionComment",
                ]
            )
        else:
            writer.writerow(
                [
                    "row",
                    "col",
                    "AMColonised",
                    "Uncolonised",
                    "Background",
                    "Unreadable",
                    "DSE",
                    "Hybrid",
                    "ContextualLabel",
                ]
            )
    else:
        if type_.lower() == "annotations":
            writer.writerow(
                [
                    "row",
                    "col",
                    "BlueCoils",
                    "BrownCoils",
                    "TypeTwo",
                    "Uncolonised",
                    "Background",
                    "MainRoot",
                    "Unreadable",
                    "DSE",
                    "HybridErm",
                    "HybridDse",
                    "Question",
                    "QuestionComment",
                ]
            )
        else:
            writer.writerow(
                [
                    "row",
                    "col",
                    "BlueCoils",
                    "BrownCoils",
                    "TypeTwo",
                    "Uncolonised",
                    "Background",
                    "MainRoot",
                    "Unreadable",
                    "DSE",
                    "HybridErm",
                    "HybridDse",
                    "ContextualLabel",
                ]
            )
    for row in out:
        writer.writerow([x for x in row])

    csv_string = output.getvalue()
    output.close()

    return csv_string


def delete_image(crsr: psycopg2, id_: int) -> None:
    """Delete a single image from the database based on its ID.

    Args:
        crsr: Database cursor.
        id_: ID of the image to delete.
    """
    delete_query = "DELETE FROM ImageReference WHERE Id=%s"
    crsr.execute(delete_query, (id_,))


def zip_files_for_transit(images: dict[str, bytes]) -> Response:
    """Zip a batch of image tiles to be passed to the frontend.

    Args:
        images: dict with key `{row}.{col}` and value being
    """
    s = io.BytesIO()
    zf = ZipFile(s, "w")

    for key, value in images.items():
        zf.writestr(key, value)

    # Must close zip for all contents to be written
    s.seek(0)
    zf.close()

    # Grab ZIP file from in-memory, make response with correct MIME-type
    # application/x-zip-compressed another option
    zip_filename = "image-tiles.zip"
    return Response(
        s.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment;filename={zip_filename}"},
    )


def _parse_setting(value: str, value_type: str) -> Any:
    """Convert a value from string to its appropriate type.

    Args:
        value: The value as a string.
        value_type: The type of the value ("integer", "boolean", "float", or "string").

    Returns: The value converted to its appropriate type.
    """
    if value_type == "integer":
        return int(value)
    if value_type == "boolean":
        return value.lower() == "true"
    if value_type == "float":
        return float(value)
    return value


def get_all_settings_from_db(crsr: psycopg2.extensions.cursor) -> dict[str, Any]:
    """Retrieve all settings from the database.

    Args:
        crsr: Database cursor.

    Returns: Dictionary of settings.
    """
    query = "SELECT key, value, value_type FROM settings"
    crsr.execute(query)
    out = crsr.fetchall()

    settings = {}
    for value in out:
        settings[value[0]] = _parse_setting(value[1], value[2])
    return settings


def change_setting_to_default_in_db(
    crsr: psycopg2.extensions.cursor, key: str
) -> dict[str, Any] | int:
    """Change a single setting in the database to its default value.

    Args:
        crsr: Database cursor.
        key: Setting key to change.

    Returns: Dict with key and new value, or 500 if key not found.
    """
    query = "SELECT default_value, value_type FROM settings WHERE key=%s"
    crsr.execute(query, (key,))
    out = crsr.fetchall()

    if len(out) > 0:
        default_value = out[0][0]
        value_type = out[0][1]

        set_query = "UPDATE settings SET value=%s WHERE key=%s"

        crsr.execute(set_query, (default_value, key))

        return {"key": key, "value": _parse_setting(default_value, value_type)}
    return 500


def update_setting_in_db(
    crsr: psycopg2.extensions.cursor, key: str, value: str
) -> None:
    """Update a single setting in the database.

    Args:
        crsr: Database cursor.
        key: Setting key to update.
        value: New value.
    """
    set_query = "UPDATE settings SET value=%s WHERE key=%s"
    crsr.execute(set_query, (value, key))
