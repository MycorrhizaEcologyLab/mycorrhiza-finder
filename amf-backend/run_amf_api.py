import io
import os
import signal
import sys
import threading
import webbrowser
from argparse import ArgumentParser, RawTextHelpFormatter
from contextlib import asynccontextmanager
from itertools import product
from typing import Any, AsyncGenerator

import psycopg2
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image

import amf.helper.config as AmfConfig
import amf.mode.bald as AmfBald
import amf.mode.calibrate as AmfCalibrate
import amf.mode.colonisation as AmfColonisation
import amf.mode.convert as AmfConvert
import amf.mode.convert_image_type as AmfConvertImageType
import amf.mode.convert_tile_size as AmfConvertTileSize
import amf.mode.predict as AmfPredict
import amf.mode.test as AmfTest
import amf.mode.train as AmfTrain
from amf.helper.api_objects import (
    AnnotationValues,
    BaseConfig,
    CalibrateConfig,
    ConvertConfig,
    PredictionConfig,
    PredictionValues,
    Setting,
    Settings,
    TestConfig,
    TifConversionConfig,
    TileEdgeConfig,
    TrainConfig,
)
from amf.helper.api_utils import (
    change_setting_to_default_in_db,
    check_entries_for_id,
    check_entries_for_image,
    delete_image,
    download_entries_as_csv,
    fetch_items,
    get_all_settings_from_db,
    get_images,
    save_annotations_to_db,
    save_predictions_to_db,
    set_to_enabled_in_db,
    update_setting_in_db,
    zip_files_for_transit,
)
from amf.helper.db_config import connect, create_database_if_not_exists

# This allows any size image
Image.MAX_IMAGE_PIXELS = None

if getattr(sys, "frozen", False):
    wd = sys._MEIPASS  # type: ignore[attr-defined]
    scripts_dir = os.path.join(wd, "scripts")
else:
    wd = os.path.dirname(__file__)
    scripts_dir = os.path.join(wd, "database", "scripts")


### CONFIG

# Set this to True when developing and False when building exes
IS_DEV = False


def shutdown() -> Response:
    os.kill(os.getpid(), signal.SIGTERM)
    return Response(status_code=200, content="Server shutting down...")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    yield
    shutdown()


app = FastAPI(lifespan=lifespan)

# Setup serve of static files - files from npm install must be copied here
app.mount(
    "/static",
    StaticFiles(directory=os.path.join(wd, "_internal/static")),
    name="static",
)
templates = Jinja2Templates(directory=os.path.join(wd, "_internal/templates"))

# Enable CORS for locally running react app - DEV ONLY
origins = ["http://localhost:3000", "http://127.0.0.1:3000"] if IS_DEV else []
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=5)

### ENDPOINTS


@app.get("/")
async def serve_spa(request: Request) -> Response:
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/fetch-predictions-cnn-1")
def fetch_predictions_cnn1(
    name: str = "", id_: str = "", colonisation_type: str = "am"
) -> dict[str, list[Any]]:
    with conn, conn.cursor() as crsr:
        return fetch_items(crsr, name, id_, "Predictions", "1", colonisation_type)


@app.get("/fetch-annotations-cnn-1")
def fetch_annotations_cnn1(
    name: str = "", id_: str = "", colonisation_type: str = "am"
) -> dict[str, list[Any]]:
    with conn, conn.cursor() as crsr:
        return fetch_items(crsr, name, id_, "Annotations", "1", colonisation_type)


@app.post("/save-annotations")
def save_annotations(annotations: AnnotationValues) -> int:
    with conn, conn.cursor() as crsr:
        return save_annotations_to_db(crsr, annotations)


@app.post("/save-predictions")
def save_predictions(predictions: PredictionValues) -> None:  # TODO make consistent
    with conn, conn.cursor() as crsr:
        save_predictions_to_db(crsr, predictions)


@app.get("/get-image-names")
def get_image_names() -> list[tuple[str]]:
    with conn, conn.cursor() as crsr:
        return get_images(crsr)


@app.get("/check-entries-for-image")
def check_for_image(
    name: str = "", colonisation_type: str = "am"
) -> dict[int, dict[str, Any]]:
    with conn, conn.cursor() as crsr:
        return check_entries_for_image(crsr, name, colonisation_type)


@app.get("/check-entries-for-id")
def check_for_id(id_: int, colonisation_type: str = "am") -> dict[int, dict[str, Any]]:
    with conn, conn.cursor() as crsr:
        return check_entries_for_id(crsr, id_, colonisation_type)


@app.get("/download-entries")
def download(id_: int, type_: str, colonisation_type: str = "am") -> str:
    with conn, conn.cursor() as crsr:
        return download_entries_as_csv(crsr, id_, type_, colonisation_type)


@app.delete("/delete-image-reference/{id_}")
def delete_image_reference(id_: int) -> None:
    with conn, conn.cursor() as crsr:
        return delete_image(crsr, id_)  # confusing as delete_image returns None


@app.patch("/set-to-enabled/{id_}")
def set_to_enabled(id_: int) -> None:
    with conn, conn.cursor() as crsr:
        return set_to_enabled_in_db(crsr, id_)  # ditto


@app.post("/calculate-predictions")
def calculate_predictions(prediction_config: PredictionConfig) -> int:
    AmfConfig.set_predict_config(prediction_config)
    input_files = AmfConfig.get_input_files()
    return AmfPredict.run(input_files)


@app.post("/train-model")
def train_model(train_config: TrainConfig) -> int | None:
    AmfConfig.set_train_config(train_config)

    input_files = AmfConfig.get_input_files()
    get_tiles_for_labelling_using_active_learning = AmfConfig.get(
        "get_tiles_for_labelling_using_active_learning"
    )
    if get_tiles_for_labelling_using_active_learning:
        return AmfBald.run(input_files)
    else:
        train_active_learning = AmfConfig.get("train_active_learning")
        ml_flow_flag = (
            train_config.mlflowFlag if train_config.mlflowFlag is not None else False
        )
        return AmfTrain.run(input_files, ml_flow_flag, train_active_learning)


@app.post("/calculate-colonisation")
def calculate_colonisation(colonisation_config: BaseConfig) -> int:
    AmfConfig.set_colonisation_config(colonisation_config)
    input_files = AmfConfig.get_input_files()
    return AmfColonisation.run(input_files)


@app.post("/test-model")
def test_model(test_config: TestConfig) -> int | None:
    AmfConfig.set_test_config(test_config)
    input_files = AmfConfig.get_input_files()
    return AmfTest.run(input_files)


@app.post("/convert-images")
def convert_images(convert_config: ConvertConfig) -> int:
    AmfConfig.set_convert_config(convert_config)
    input_files = AmfConfig.get_input_files()
    if AmfConfig.get("aggregate_tiles"):
        return AmfConvertTileSize.run(input_files)
    else:
        return AmfConvert.run(input_files)


@app.post("/tif-conversion")
def tif_conversion(tif_conversion_config: TifConversionConfig) -> int:
    AmfConfig.set_tif_conversion_config(tif_conversion_config)
    input_files = AmfConfig.get_input_files()
    return AmfConvertImageType.run(input_files)


@app.post("/calibrate-model")
def calibrate_model(calibrate_config: CalibrateConfig) -> int:
    AmfConfig.set_calibrate_config(calibrate_config)
    input_files = AmfConfig.get_input_files()
    return AmfCalibrate.run(input_files)


@app.post("/resize-image")
async def resize_image(
    maxSize: int = Form(1000), file: UploadFile = File(...)
) -> Response:
    try:
        max_size = maxSize, maxSize
        contents = file.file.read()
        image = Image.open(io.BytesIO(contents))
        # Use LANCZOS as this offers high quality downsampling
        # so is best for reducing image size
        image.thumbnail(max_size, Image.Resampling.LANCZOS)

        img_byte_io = io.BytesIO()
        image.save(img_byte_io, format="JPEG")
        img_byte_arr = img_byte_io.getvalue()
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Something went wrong")
    finally:
        print(f"[{AmfConfig.invite()}] Returning resized image")
        file.file.close()
        return Response(content=img_byte_arr, media_type="image/jpeg")


@app.post("/tile-image")
async def tile_image(file: UploadFile = File(...)) -> dict[str, int]:
    try:
        # Set current image to an empty dict to wipe any existing image
        current_image = {}
        contents = file.file.read()
        edge = AmfConfig.get("tile_edge")
        image = Image.open(io.BytesIO(contents))
        width, height = image.size
        num_cols = width // edge
        num_rows = height // edge

        grid = product(
            range(0, height - height % edge, edge), range(0, width - width % edge, edge)
        )
        for i, j in grid:
            box = (j, i, j + edge, i + edge)
            cropped_tile = image.crop(box)
            img_byte_arr = io.BytesIO()
            cropped_tile.save(img_byte_arr, format="JPEG")
            current_image[f"{i // edge}.{j // edge}"] = img_byte_arr.getvalue()

        AmfConfig.set_("tiles", current_image)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Something went wrong")
    finally:
        print(f"[{AmfConfig.invite()}] Image tiled and saved to memory")
        file.file.close()
        return {"num_rows": num_rows, "num_cols": num_cols}


@app.get("/get-image-tile")
def get_tiles(startIndex: int, batchSize: int = 1000) -> Response:
    images = AmfConfig.get("tiles")
    endIndex = (
        startIndex + batchSize if startIndex + batchSize < len(images) else len(images)
    )
    return zip_files_for_transit(dict(list(images.items())[startIndex:endIndex]))


@app.post("/set-tile-edge")
def set_tile_edge(tileEdgeConfig: TileEdgeConfig) -> int:
    AmfConfig.set_("tile_edge", tileEdgeConfig.tileEdge)
    print(f"[{AmfConfig.invite()}] Tile edge set to {tileEdgeConfig.tileEdge}")
    return 200


@app.post("/save-settings")
def save_settings(settings: Settings) -> None:
    with conn, conn.cursor() as crsr:
        for x in settings:
            update_setting_in_db(crsr, x[0], x[1])


@app.get("/get-all-settings")
def get_all_settings() -> dict[str, Any]:
    with conn, conn.cursor() as crsr:
        return get_all_settings_from_db(crsr)


@app.post("/revert-setting-to-default")
def revert_setting_to_default(setting: Setting) -> dict[str, Any] | int:
    with conn, conn.cursor() as crsr:
        return change_setting_to_default_in_db(crsr, setting.key)


@app.post("/revert-all-settings-to-default")
def revert_all_settings_to_default(settings: Settings) -> int:
    with conn, conn.cursor() as crsr:
        for x in settings:
            change_setting_to_default_in_db(crsr, x[0])
    return 200


############# UTILS ################


def open_browser() -> None:
    threading.Timer(1.25, lambda: webbrowser.open("http://127.0.0.1:8001")).start()


def check_if_settings_exists(crsr: psycopg2.extensions.cursor) -> bool:
    # Query to check if the table exists
    crsr.execute(
        """
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE  table_schema = 'public'
            AND    table_name   = %s
        );
    """,
        ("settings",),
    )

    table_exists = crsr.fetchone()[0]

    if not table_exists:
        print("Table settings does not exist.")
        return False

    crsr.execute("SELECT COUNT(*) FROM settings")

    # Fetch the count result
    row_count = crsr.fetchone()[0]

    if row_count == 0:
        print("Settings exists but it is empty.")
        return False
    else:
        print(f"Settings exists and it contains {row_count} rows.")
        return True


### MAIN

if __name__ == "__main__":
    main = ArgumentParser(
        description="MycorrhizaFinder command-line arguments.",
        allow_abbrev=False,
        formatter_class=RawTextHelpFormatter,
    )

    main.add_argument(
        "-p",
        "--db-password",
        type=str,
        dest="db_password",
        action="store",
        required=False,
        help="Password for postgres",
    )

    par = main.parse_known_args()[0]
    password = par.db_password

    create_database_if_not_exists("amf", "postgres", password)
    conn = connect("amf", password)
    with conn, conn.cursor() as crsr:
        # Initial setup of tables
        crsr.execute(open(os.path.join(scripts_dir, "create_schema.sql"), "r").read())

        # Check if settings exist, and if not populate table with default values
        settings_exist = check_if_settings_exists(crsr)
        if not settings_exist:
            print("Populating settings.")
            crsr.execute(
                open(
                    os.path.join(wd, scripts_dir, "fill_default_settings.sql"), "r"
                ).read()
            )

    # Only open browser if running from exe - https://pyinstaller.org/en/stable/runtime-information.html
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS") and not IS_DEV:
        open_browser()

    print(f"Running in {'development' if IS_DEV else 'production'} mode")

    # Spin up FastAPI endpoints
    uvicorn.run(app, host="127.0.0.1", port=8001)
