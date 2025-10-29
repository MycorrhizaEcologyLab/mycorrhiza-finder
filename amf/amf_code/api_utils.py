import io
import csv
import zipfile
from fastapi import HTTPException, Response
from api_objects import AnnotationValues, PredictionValues


def fetch_items(crsr, name, id, type, cnn, colonisation_type):
    """
    Fetches items from the database based on provided parameters.

    This function retrieves item data corresponding to either a name or an ID,
    where a preference is given to the ID if both are provided. If neither
    name nor ID is provided, an HTTPException is raised.

    Parameters:
    crsr (cursor): The database cursor used to execute SQL queries.
    name (str): The name reference used to fetch image IDs; can be empty.
    id (str): The specific ID reference for fetching item data; can be empty.
    type (str): A string that represents predictions or annotations.
    cnn (str): A string that represents the CNN.
    colonisation_type (str): A string that represents am or erm

    Returns:
    dict: A dictionary where keys are image IDs and values are lists of item data
          corresponding to those IDs, excluding the image reference.

    Raises:
    HTTPException: If both name and id are empty, or if the parameters do not
                   meet the expected input criteria.
    """
    if name == "" and id == "":
        raise HTTPException(
            status_code=400, detail="Name & id parameters are empty, please populate"
        )

    if name != "" and id != "":
        print(f"Both name and ID provided, name ignored in favour of more specific ID")

    values = {}
    image_ids = []
    if id == "":
        fetch_image_id = "SELECT Id FROM ImageReference WHERE FileNameReference=%s"
        crsr.execute(fetch_image_id, (name,))
        image_ids = [row[0] for row in crsr.fetchall()]
    else:
        image_ids.append(id)

    fetch_values_query = (
        "SELECT * FROM Cnn"
        + cnn
        + type
        + colonisation_type
        + " WHERE ImageReferenceId=%s"
    )
    for id in image_ids:
        crsr.execute(fetch_values_query, (id,))
        preds = crsr.fetchall()
        out = []
        # Remove image reference
        for x in preds:
            out.append(x[1:])
        values[id] = out

    return values


def set_to_enabled_in_db(crsr, id):
    """
    Sets the 'Enabled' status of an image reference to true in the database for a given ID.

    This function updates the 'Enabled' field of the image reference to true for
    the specified ID and then disables all other image references that have the same
    file name. The function returns the file name reference of the updated image.

    Parameters:
    crsr (cursor): The database cursor used to execute SQL queries.
    id (int or str): The unique identifier of the image reference to be updated.

    Returns:
    str: The file name reference of the updated image.

    Raises:
    Exception: May raise exceptions related to database operations
                (e.g., if the ID does not exist).
    """
    update_enabled_query = "UPDATE imageReference SET Enabled = true WHERE Id = %s RETURNING filenamereference"
    crsr.execute(update_enabled_query, (id,))
    file_name_reference = crsr.fetchone()[0]
    set_enabled_query = "UPDATE imageReference SET Enabled = false WHERE FileNameReference=%s AND Id <> %s"
    crsr.execute(
        set_enabled_query,
        (
            file_name_reference,
            id,
        ),
    )


def save_annotations_to_db(crsr, values: AnnotationValues):
    """
    Saves annotation data to the database associated with a specific image reference.

    This function first checks if a provided image reference ID exists in the database.
    If it does not exist, it inserts a new record into the `imagereference` table.

    If the annotations are to be enabled, it updates the `Enabled` field for the relevant
    image references. The function also handles the insertion of CNN annotations for
    both Type One and Type Two annotations, replacing any existing data for the specified
    image reference ID.

    Parameters:
    crsr (cursor): The database cursor used to execute SQL queries.
    values (AnnotationValues): An object containing the details of the annotations to be saved,
                               including the image reference ID, file name, tile edge, enable
                               status, and CNN annotation values.

    Returns:
    int: The ID of the saved image reference.

    Raises:
    Exception: May raise exceptions related to database operations (e.g., SQL errors or
               constraints violations).
    """
    # Check if image reference ID exists in DB
    exists = False
    if values.imageReferenceId != None:
        # Check that given ID already has an entry in the database
        check_image_id = "SELECT EXISTS(SELECT 1 FROM imagereference WHERE id=%s)"
        crsr.execute(check_image_id, (values.imageReferenceId,))
        exists = crsr.fetchone()[0]

    image_id = 0
    if exists:
        image_id = values.imageReferenceId
        update_time_query = (
            "UPDATE ImageReference SET UpdatedAt = CURRENT_TIMESTAMP WHERE Id=%s"
        )
        crsr.execute(
            update_time_query,
            (image_id,),
        )
        tile_edge = get_tile_edge(crsr, image_id)
        if tile_edge[0] != values.tileEdge:
            print(
                f"Tile edge does not match, update from {tile_edge[0]} to {values.tileEdge}"
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
        insert_id_query = "INSERT INTO imagereference(filenamereference, TileEdge, Enabled) VALUES (%s, %s, false) RETURNING id"
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
            "DELETE FROM cnn1annotations"
            + values.colonisationType
            + " WHERE imagereferenceid=%s"
        )
        crsr.execute(delete_query, (image_id,))

        if values.colonisationType == "am":
            insert_values_query = (
                """INSERT INTO cnn1annotations"""
                + values.colonisationType
                + """(
                imagereferenceid, rownum, colnum, amcolonised, uncolonised, background, unreadable, dse, hybrid, question, questioncomment)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
            )
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
            insert_values_query = (
                """INSERT INTO cnn1annotations"""
                + values.colonisationType
                + """(
            imagereferenceid, rownum, colnum, BlueCoils, BrownCoils, TypeTwo, Uncolonised, Background, MainRoot, Unreadable, DSE, HybridErm, HybridDse, Question, QuestionComment)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
            )

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

    # Only applicable for AM colonisation
    if (
        values.colonisationType == "am"
        and values.cnnTwoValues
        and len(values.cnnTwoValues) != 0
    ):
        delete_query = "DELETE FROM cnn2annotationsam WHERE imagereferenceid=%s"
        crsr.execute(delete_query, (image_id,))

        insert_values_query = (
            """INSERT INTO cnn2annotationsam"""
            + """(
        imagereferenceid, rownum, colnum, arbuscule, vesicle, hyphopodium, hypha)
        VALUES (%s, %s, %s, %s, %s, %s, %s)"""
        )
        for row in values.cnnTwoValues:
            crsr.execute(
                insert_values_query,
                (image_id, row[0], row[1], row[2], row[3], row[4], row[5]),
            )

    print("Saved Image Reference ID: ", image_id)
    return image_id


def save_predictions_to_db(crsr, values: PredictionValues):
    """
    Saves prediction data to the database associated with a specific image reference.

    This function first checks if the provided image reference ID exists in the database.
    If it does not exist, it inserts a new record into the `imagereference` table.
    The function then handles the insertion of CNN prediction data for both level one
    two predictions, deleting any existing data for the specified image reference ID
    before inserting the new predictions.

    Parameters:
    crsr (cursor): The database cursor used to execute SQL queries.
    values (PredictionValues): An object containing the details of the predictions to be saved,
                               including the image reference ID, file name, tile edge,
                               colonisation type, and CNN prediction values.

    Returns:
    None: This function does not return any value but modifies the database
          according to the provided predictions.

    Raises:
    Exception: May raise exceptions related to database operations (e.g., SQL errors or
               constraint violations).
    """

    # Check if image reference ID exists in DB
    exists = False
    if values.imageReferenceId != None:
        # Check that given ID already has an entry in the database
        check_image_id = "SELECT EXISTS(SELECT 1 FROM imagereference WHERE id=%s)"
        crsr.execute(check_image_id, (values.imageReferenceId,))
        exists = crsr.fetchone()[0]

    image_id = 0
    if exists:
        image_id = values.imageReferenceId
        update_time_query = (
            "UPDATE ImageReference SET UpdatedAt = CURRENT_TIMESTAMP WHERE Id=%s"
        )
        crsr.execute(
            update_time_query,
            (image_id,),
        )
    else:
        # If not, then upload file name to DB
        insert_id_query = "INSERT INTO imagereference(filenamereference, TileEdge, Enabled) VALUES (%s, %s, false) RETURNING id"
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
            "DELETE FROM cnn1predictions"
            + values.colonisationType
            + " WHERE imagereferenceid=%s"
        )
        crsr.execute(delete_query, (image_id,))

        if values.colonisationType == "am":
            insert_values_query = (
                """INSERT INTO cnn1predictions"""
                + values.colonisationType
                + """(
            imagereferenceid, rownum, colnum, amcolonised, uncolonised, background, unreadable, dse, hybrid, contextuallabel)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
            )
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
            insert_values_query = (
                """INSERT INTO cnn1predictions"""
                + values.colonisationType
                + """(
            imagereferenceid, rownum, colnum, BlueCoils, BrownCoils, TypeTwo, Uncolonised, Background, MainRoot, Unreadable, DSE, HybridErm, HybridDse, contextuallabel)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
            )

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

    # Only applicable for AM colonisation
    if (
        values.colonisationType == "am"
        and values.cnnTwoValues
        and len(values.cnnTwoValues) != 0
    ):
        delete_query = (
            "DELETE FROM cnn2predictions"
            + values.colonisationType
            + " WHERE imagereferenceid=%s"
        )
        crsr.execute(delete_query, (image_id,))

        insert_values_query = (
            """INSERT INTO cnn2predictions"""
            + values.colonisationType
            + """(
        imagereferenceid, rownum, colnum, arbuscule, vesicle, hyphopodium, hypha)
        VALUES (%s, %s, %s, %s, %s, %s, %s)"""
        )
        for row in values.cnnTwoValues:
            crsr.execute(
                insert_values_query,
                (image_id, row[0], row[1], row[2], row[3], row[4], row[5]),
            )


def get_images(crsr):
    get_all_images = "SELECT DISTINCT FileNameReference FROM imagereference"
    crsr.execute(get_all_images)
    images = crsr.fetchall()

    return images


def get_most_recent_timestamp(crsr, image_name):
    """
    Retrieves the ID of the most recently uploaded image based on the specified image name.

    This function queries the database for records in the `ImageReference` table that match
    the provided `image_name`. It sorts the results by upload timestamp and returns the ID
    of the most recent entry. If no images are found, it returns None.

    Parameters:
    crsr (cursor): The database cursor used to execute SQL queries.
    image_name (str): The name of the image for which to retrieve the most recent timestamp.

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
        return sorted_values[-1][0]
    else:
        return None


def get_enabled(crsr, image_name):
    """
    Retrieves the ID of an enabled image from the database based on the specified image name.

    This function queries the `ImageReference` table to check if there is an entry
    with the provided `image_name` that is marked as enabled. If an enabled image is found,
    its ID is returned. If no enabled images are present for the given name, the function
    retrieves the ID of the most recently uploaded image and sets that image to enabled before returning its ID.

    Parameters:
    crsr (cursor): The database cursor used to execute SQL queries.
    image_name (str): The name of the image for which to check for enabled status.

    Returns:
    int: The ID of the enabled image if found; otherwise, the ID of the most recent image.

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
        print(f"Image ID {images[0][0]} is enabled for image name {image_name}")
        return images[0][0]
    else:
        # If no annotations enabled, get most recent timestamp and set this to enabled
        print(
            f"No enabled annotations for image {image_name}, return image with most recent timestamp instead."
        )
        id = get_most_recent_timestamp(crsr, image_name)
        if id != None:
            set_to_enabled_in_db(crsr, id)
        return id


def get_tile_edge(crsr, id):
    fetch_images = "SELECT TileEdge FROM ImageReference WHERE Id=%s"
    crsr.execute(fetch_images, (id,))
    return crsr.fetchone()


def check_entries_for_image(crsr, image_name, colonisation_type):
    fetch_images = "SELECT Id, UploadTimestamp, Enabled, TileEdge, UpdatedAt FROM ImageReference WHERE FileNameReference=%s"
    crsr.execute(fetch_images, (image_name,))
    images = crsr.fetchall()

    return _get_existing_entries_for_image(crsr, colonisation_type, images)


def check_entries_for_id(crsr, id, colonisation_type):
    fetch_images = "SELECT Id, UploadTimestamp, Enabled, TileEdge, UpdatedAt FROM ImageReference WHERE Id=%s"
    crsr.execute(fetch_images, (id,))
    images = crsr.fetchall()

    return _get_existing_entries_for_image(crsr, colonisation_type, images)


def _get_existing_entries_for_image(crsr, colonisation_type, images):
    output = {}

    for value in images:
        id = value[0]
        timestamp = value[1]
        enabled = value[2]
        tileEdge = value[3]
        updatedAt = value[4]
        cnn1_predictions_exist_query = (
            "SELECT EXISTS(SELECT 1 FROM cnn1predictions"
            + colonisation_type
            + " WHERE ImageReferenceId=%s)"
        )
        cnn1_annotations_exist_query = (
            "SELECT EXISTS(SELECT 1 FROM cnn1annotations"
            + colonisation_type
            + " WHERE ImageReferenceId=%s)"
        )

        crsr.execute(cnn1_predictions_exist_query, (id,))
        cnn1_predictions_exist = crsr.fetchone()[0]

        crsr.execute(cnn1_annotations_exist_query, (id,))
        cnn1_annotations_exist = crsr.fetchone()[0]

        # CNN2 only applicable for AM colonisation
        if colonisation_type == "am":
            cnn2_predictions_exist_query = (
                "SELECT EXISTS(SELECT 1 FROM cnn2predictions"
                + colonisation_type
                + " WHERE ImageReferenceId=%s)"
            )
            crsr.execute(cnn2_predictions_exist_query, (id,))
            cnn2_predictions_exist = crsr.fetchone()[0]

            cnn2_annotations_exist_query = (
                "SELECT EXISTS(SELECT 1 FROM cnn2annotations"
                + colonisation_type
                + " WHERE ImageReferenceId=%s)"
            )
            crsr.execute(cnn2_annotations_exist_query, (id,))
            cnn2_annotations_exist = crsr.fetchone()[0]
        else:
            cnn2_predictions_exist = False
            cnn2_annotations_exist = False

        if (
            cnn1_annotations_exist
            or cnn1_predictions_exist
            or cnn2_annotations_exist
            or cnn2_predictions_exist
        ):
            output[id] = {
                "timestamp": timestamp,
                "enabled": enabled,
                "tileEdge": tileEdge,
                "updatedAt": updatedAt,
                "cnn1_predictions_exist": cnn1_predictions_exist,
                "cnn2_predictions_exist": cnn2_predictions_exist,
                "cnn1_annotations_exist": cnn1_annotations_exist,
                "cnn2_annotations_exist": cnn2_annotations_exist,
            }

    return output


def download_entries_as_csv(crsr, id, cnn, type, colonisation_type):
    output = io.StringIO()

    fetch_values_query = (
        "SELECT * FROM Cnn"
        + cnn
        + type
        + colonisation_type
        + " WHERE ImageReferenceId=%s"
    )
    crsr.execute(fetch_values_query, (id,))
    entries = crsr.fetchall()
    out = []
    for x in entries:
        # Remove image reference
        out.append(x[1:])

    # Write a CSV string
    output = io.StringIO()
    writer = csv.writer(output)

    if cnn == "1":
        if colonisation_type == "am":
            if type.lower() == "annotations":
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
            if type.lower() == "annotations":
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
    else:
        writer.writerow(["row", "col", "Arbuscule", "Vesicle", "Hyphopodium", "Hypha"])
    for row in out:
        writer.writerow([x for x in row])

    csv_string = output.getvalue()
    output.close()

    return csv_string


def delete_image(crsr, id):
    delete_query = "DELETE FROM ImageReference WHERE Id=%s"
    crsr.execute(delete_query, (id,))


def zip_files_for_transit(images):
    zip_filename = "image-tiles.zip"

    s = io.BytesIO()
    zf = zipfile.ZipFile(s, "w")

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


def _parse_setting(value, value_type):
    parsed_value = value
    if value_type == "integer":
        parsed_value = int(parsed_value)
    elif value_type == "boolean":
        parsed_value = parsed_value.lower() == "true"
    elif value_type == "float":
        parsed_value = float(parsed_value)

    return parsed_value


def get_all_settings_from_db(crsr):
    query = "SELECT key, value, value_type FROM settings"
    crsr.execute(query)
    out = crsr.fetchall()

    settings = {}

    for value in out:
        settings[value[0]] = _parse_setting(value[1], value[2])

    return settings


def change_setting_to_default_in_db(crsr, key):
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


def update_setting_in_db(crsr, key, value):
    set_query = "UPDATE settings SET value=%s WHERE key=%s"

    crsr.execute(set_query, (value, key))
