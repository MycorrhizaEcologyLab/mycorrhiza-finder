from argparse import ArgumentParser, RawTextHelpFormatter
import io
import os
from contextlib import asynccontextmanager
import signal
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.gzip import GZipMiddleware
import webbrowser
import threading

import sys
import os

from itertools import product
from PIL import Image

# This allows any size image
Image.MAX_IMAGE_PIXELS = None

try:
    wd = sys._MEIPASS
except AttributeError:
    wd = os.getcwd()

import amfinder_train as AmfTrain
import amfinder_config as AmfConfig
import amfinder_convert as AmfConvert
import amfinder_convert_tile_size as AmfConvertTileSize
import amfinder_convert_image_type as AmfConvertImageType
import amfinder_predict as AmfPredict
import amfinder_test as AmfTest
import amfinder_bald as AmfBald
import amfinder_calibrate as AmfCalibrate
import amfinder_colonisation as AmfColonisation
import semisupervised_train as AmfSemiSupervisedTrain
import semisupervised_evaluate as AmfSemiSupervisedEvaluate
from api_utils import (
    change_setting_to_default_in_db,
    check_entries_for_id,
    delete_image,
    download_entries_as_csv,
    fetch_items,
    get_all_settings_from_db,
    save_annotations_to_db,
    get_images,
    check_entries_for_image,
    save_predictions_to_db,
    set_to_enabled_in_db,
    update_setting_in_db,
    zip_files_for_transit,
)
from db_config import connect, create_database_if_not_exists
from api_objects import (
    BaseConfig,
    CalibrateConfig,
    ConvertConfig,
    AnnotationValues,
    PredictionConfig,
    PredictionValues,
    Setting,
    Settings,
    TestConfig,
    TifConversionConfig,
    TileEdgeConfig,
    TrainConfig,
)

### CONFIG

# Set this to True when developing and False when building exes
IS_DEV = False


def shutdown():
    os.kill(os.getpid(), signal.SIGTERM)
    return Response(status_code=200, content="Server shutting down...")


@asynccontextmanager
async def lifespan(app: FastAPI):
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
async def serve_spa(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/fetch-predictions-cnn-1")
def fetch_predictions_cnn1(name: str = "", id: str = "", colonisation_type: str = "am"):
    with conn, conn.cursor() as crsr:
        return fetch_items(crsr, name, id, "Predictions", "1", colonisation_type)


@app.get("/fetch-annotations-cnn-1")
def fetch_annotations_cnn1(name: str = "", id: str = "", colonisation_type: str = "am"):
    with conn, conn.cursor() as crsr:
        return fetch_items(crsr, name, id, "Annotations", "1", colonisation_type)


@app.post("/save-annotations")
def save_annotations(annotations: AnnotationValues):
    with conn, conn.cursor() as crsr:
        return save_annotations_to_db(crsr, annotations)


@app.post("/save-predictions")
def save_predictions(predictions: PredictionValues):
    with conn, conn.cursor() as crsr:
        save_predictions_to_db(crsr, predictions)


@app.get("/get-image-names")
def get_image_names():
    with conn, conn.cursor() as crsr:
        return get_images(crsr)


@app.get("/check-entries-for-image")
def check_for_image(name: str = "", colonisation_type: str = "am"):
    with conn, conn.cursor() as crsr:
        return check_entries_for_image(crsr, name, colonisation_type)


@app.get("/check-entries-for-id")
def check_for_id(id: str = "", colonisation_type: str = "am"):
    with conn, conn.cursor() as crsr:
        return check_entries_for_id(crsr, id, colonisation_type)


@app.get("/download-entries")
def download(id, type, colonisation_type: str = "am"):
    with conn, conn.cursor() as crsr:
        return download_entries_as_csv(crsr, id, type, colonisation_type)


@app.delete("/delete-image-reference/{id}")
def delete_image_reference(id):
    with conn, conn.cursor() as crsr:
        return delete_image(crsr, id)


@app.patch("/set-to-enabled/{id}")
def set_to_enabled(id):
    with conn, conn.cursor() as crsr:
        return set_to_enabled_in_db(crsr, id)


@app.post("/calculate-predictions")
def calculate_predictions(prediction_config: PredictionConfig):
    AmfConfig.set_predict_config(prediction_config)
    input_files = AmfConfig.get_input_files()
    return AmfPredict.run(input_files)


@app.post("/train-model")
def train_model(train_config: TrainConfig):
    AmfConfig.set_train_config(train_config)
    if train_config.semiSupervised:
        return AmfSemiSupervisedTrain.run(AmfConfig.get("root_path"))

    input_files = AmfConfig.get_input_files()
    get_tiles_for_labelling_using_active_learning = AmfConfig.get(
        "get_tiles_for_labelling_using_active_learning"
    )
    if get_tiles_for_labelling_using_active_learning:
        return AmfBald.run(input_files)
    else:
        train_active_learning = AmfConfig.get("train_active_learning")
        return AmfTrain.run(input_files, train_config.mlflowFlag, train_active_learning)


@app.post("/calculate-colonisation")
def calculate_colonisation(colonisation_config: BaseConfig):
    AmfConfig.set_colonisation_config(colonisation_config)
    input_files = AmfConfig.get_input_files()
    return AmfColonisation.run(input_files)


@app.post("/test-model")
def test_model(test_config: TestConfig):
    AmfConfig.set_test_config(test_config)

    if test_config.semiSupervised:
        return AmfSemiSupervisedEvaluate.run()

    input_files = AmfConfig.get_input_files()
    return AmfTest.run(input_files)


@app.post("/convert-images")
def convert_images(convert_config: ConvertConfig):
    AmfConfig.set_convert_config(convert_config)
    input_files = AmfConfig.get_input_files()
    if AmfConfig.get("aggregate_tiles"):
        return AmfConvertTileSize.run(input_files)
    else:
        return AmfConvert.run(input_files)


@app.post("/tif-conversion")
def convert_images(tif_conversion_config: TifConversionConfig):
    AmfConfig.set_tif_conversion_config(tif_conversion_config)
    input_files = AmfConfig.get_input_files()
    return AmfConvertImageType.run(input_files)


@app.post("/calibrate-model")
def calibrate_model(calibrate_config: CalibrateConfig):
    AmfConfig.set_calibrate_config(calibrate_config)
    input_files = AmfConfig.get_input_files()
    return AmfCalibrate.run(input_files)


@app.post("/resize-image")
async def resize_image(maxSize: int = Form(1000), file: UploadFile = File(...)):
    try:
        max_size = maxSize, maxSize
        contents = file.file.read()
        image = Image.open(io.BytesIO(contents))
        # Use LANCZOS as this offers high quality downsampling
        # so is best for reducing image size
        image.thumbnail(max_size, Image.Resampling.LANCZOS)

        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="JPEG")
        img_byte_arr = img_byte_arr.getvalue()
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Something went wrong")
    finally:
        print(f"[{AmfConfig.invite()}] Returning resized image")
        file.file.close()
        return Response(content=img_byte_arr, media_type="image/jpeg")


@app.post("/tile-image")
async def tile_image(file: UploadFile = File(...)):
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

        AmfConfig.set("tiles", current_image)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Something went wrong")
    finally:
        print(f"[{AmfConfig.invite()}] Image tiled and saved to memory")
        file.file.close()
        return {"num_rows": num_rows, "num_cols": num_cols}


@app.get("/get-image-tile")
def get_tiles(startIndex: int, batchSize: int = 1000):
    images = AmfConfig.get("tiles")
    endIndex = (
        startIndex + batchSize if startIndex + batchSize < len(images) else len(images)
    )
    return zip_files_for_transit(dict(list(images.items())[startIndex:endIndex]))


@app.post("/set-tile-edge")
def set_tile_edge(tileEdgeConfig: TileEdgeConfig):
    AmfConfig.set("tile_edge", tileEdgeConfig.tileEdge)
    print(f"[{AmfConfig.invite()}] Tile edge set to {tileEdgeConfig.tileEdge}")
    return 200


@app.post("/save-settings")
def save_settings(settings: Settings):
    with conn, conn.cursor() as crsr:
        for x in settings:
            update_setting_in_db(crsr, x[0], x[1])


@app.get("/get-all-settings")
def get_all_settings():
    with conn, conn.cursor() as crsr:
        return get_all_settings_from_db(crsr)


@app.post("/revert-setting-to-default")
def revert_setting_to_default(setting: Setting):
    with conn, conn.cursor() as crsr:
        return change_setting_to_default_in_db(crsr, setting.key)


@app.post("/revert-all-settings-to-default")
def revert_all_settings_to_default(settings: Settings):
    with conn, conn.cursor() as crsr:
        for x in settings:
            change_setting_to_default_in_db(crsr, x[0])


############# UTILS ################


def open_browser():
    threading.Timer(1.25, lambda: webbrowser.open("http://127.0.0.1:8001")).start()


def check_if_settings_exists(crsr):
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
        print(f"Table settings does not exist.")
        return False

    crsr.execute("SELECT COUNT(*) FROM settings")

    # Fetch the count result
    row_count = crsr.fetchone()[0]

    if row_count == 0:
        print(f"Settings exists but it is empty.")
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
        crsr.execute(open(os.path.join(wd, "scripts/create_schema.sql"), "r").read())

        # Check if settings exist, and if not populate table with default values
        settings_exist = check_if_settings_exists(crsr)
        if not settings_exist:
            print(f"Populating settings.")
            crsr.execute(
                open(os.path.join(wd, "scripts/fill_default_settings.sql"), "r").read()
            )

    # Only open browser if running from exe - https://pyinstaller.org/en/stable/runtime-information.html
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS") and not IS_DEV:
        open_browser()

    print(f"Running in {'development' if IS_DEV else 'production'} mode")

    # Spin up FastAPI endpoints
    uvicorn.run(app, host="127.0.0.1", port=8001)
