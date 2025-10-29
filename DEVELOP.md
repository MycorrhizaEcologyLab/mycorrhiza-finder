# For Developers

## Development vs production mode

In main.py there is a global argument `IS_DEV` which sets your CORS settings. You need to set this to True when developing locally, otherwise the front-end will not be able to communicate with the backend.

## Building new versions of the tool

In order to release new versions of the tool, you will need to rebuild the executable. Note that for each platform you want to release the tool on, you need to build the tool on that corresponding platform. The default build is for Windows, but this can easily be adapted for mac and linux. Note that all of the below assumes you have already followed the [installation instructions for developers](INSTALL.md#for-developers).

### Generating a new executable

The main things for generating a new executable are creating a production build of the React UI, and generating a new pyinstaller executable. The steps to do this are included below for each platform.

**_Note:_** for the .bat scripts below, it is recommended to run each line independently by copying them into a terminal, instead of running the .bat scripts as whole. This is due to them pausing at points, which will mean the build does not complete.

1. First, traverse to the build directory via `cd amf/amf_code/build`.
1. You then want to prepare the build by following the steps in `prepare_build.bat`. Note that this is written for Windows, so you will need to change it for mac and linux by activating your venv using `source` (see [here](https://docs.python.org/3/library/venv.html#how-venvs-work) for details on activating venvs from different OS').
1. Then follow the steps in build.bat to build the executable. These are included below for transparency.

```
cd ../../../amfbrowser-react
npm install
npm run build

cd ..\amf\amf_code\build
rm -Recurse ../_internal/static/js
rm -Recurse ../_internal/static/media
rm -Recurse ../_internal/static/css
rm ../_internal/static/amfbrowser.ico
rm ../_internal/templates/index.html

cp -r ../../../amfbrowser-react/build/static/* ../_internal/static
cp -r ../../../amfbrowser-react/build/amfbrowser.ico ../_internal/static
cp -r ../../../amfbrowser-react/build/index.html ../_internal/templates

pyinstaller ../main.py --name=amf --noconfirm -i amfbrowser.ico --add-data "../trained_networks/*;./trained_networks" --add-data "../_internal/static/css/*;_internal/static/css" --add-data "../_internal/static/js/*;_internal/static/js" --add-data "../_internal/static/amfbrowser.ico;_internal/static" --add-data "../_internal/static/media/*;_internal/static/media" --add-data "../_internal/templates/index.html;_internal/templates" --add-data "../scripts/*;./scripts" --add-data "../database.ini;."
```

This will build an executable under the dist folder that you can use to run the tool. Note that this comes with an \_internal folder that you must provide with the exe for it to work - they must also stay in the same directory.

### Differences for mac and linux

For mac and linux, you must build the exe on these respective platforms. The only other difference (besides replacing the -Recurse flag with -rf) is that you must replace the pyinstaller command with the below (note that the only difference is that we replace `;` with `:`).

```
pyinstaller ../main.py --noconfirm -i amfbrowser.ico --add-data "../trained_networks/*:./trained_networks" --add-data "../_internal/static/css/*:_internal/static/css" --add-data "../_internal/static/js/*:_internal/static/js" --add-data "../_internal/static/amfbrowser.ico:_internal/static" --add-data "../_internal/static/media/*:_internal/static/media" --add-data "../_internal/templates/index.html:_internal/templates" --add-data "../scripts/*:./scripts" --add-data "../database.ini:."
```

## Batch processing (`amf`)<a name="amf"></a>

It can be useful when developing new models to directly run the amf functionalities without using the UI. In order to run the amf tool via python, you can run the below command in the amf_code repo.

```
python amf <action> --images <path_to_folder> <parameters>
```

where `<action>` is either:

- `train`: neural network training,
- `predict`: prediction of fungal colonisation (CNN1) and intraradical hyphal structures (CNN2),
- `convert`: automatic conversion of predictions to annotations,
- `colonisation`: calculates the percentage colonisation in a set of annotations,
- `tifconvert`: converts TIFs to JPGs or PNGs
- `test`: produce output performance metrics of the model on the test data,
- `diagnose`: produce diagnosis metrics for evaluating certain predictions.

`<path_to_folder>` is the paths to the folder containing images of the roots.
Details about `<parameters>` are given in the following sections.

**_Note_**: the functionality of all of the above is exhaustively explained in the documentation doc/mycorrhiza-finder-user-documentation.pdf.

### Training mode

This mode is used to train neural networks on different images.

| Short     | Long                              | Description                                                                                                    | Default value            |
| --------- | --------------------------------- | -------------------------------------------------------------------------------------------------------------- | ------------------------ |
| `-l`      | `--use-csvs`                      | Use CSVs instead of DB.                                                                                        | False                    |
| `-i`      | `--images`                        | **Mandatory**. Directory of images to process.                                                                 | None (required argument) |
| `-smi`    | `--semi_supervised`               | Enables semi-supervised training.                                                                              | False                    |
| `-tal`    | `--train-active-learning`         | Training with different default learning rate and num epochs after active learning samples have been acquired. | False                    |
| `-gtfl`   | `--get-tiles-for-labelling`       | Run active learning to get tiles for labelling.                                                                | False                    |
| `-alm`    | `--active-learning-method`        | Active learning method to use (bald, batchbald).                                                               | bald                     |
| `-sfl`    | `--samples-for-labelling`         | Number of samples to select per file for labeling.                                                             | 10                       |
| `-mc`     | `--mc-samples`                    | Number of Monte Carlo samples for uncertainty estimation.                                                      | 50                       |
| `-dr`     | `--dropout-rate`                  | Dropout rate for uncertainty-based acquisition.                                                                | 0.25                     |
| `-mfl`    | `--mlflow`                        | Enables mlflow tracking in training.                                                                           | False                    |
| `-b`      | `--batch_size`                    | Training batch size.                                                                                           | 32                       |
| `-a`      | `--data_augmentation`             | Apply data augmentation (hue, chroma, saturation, etc.).                                                       | False                    |
| `-s`      | `--summary`                       | Save CNN architecture.                                                                                         | False                    |
| `-o`      | `--outdir`                        | Folder where to save trained model and CNN architecture.                                                       | None                     |
| `-e`      | `--epochs`                        | Number of epochs to run.                                                                                       | 50                       |
| `-eal`    | `--epochs-active-learning`        | Number of epochs to run after active learning.                                                                 | 5                        |
| `-pe`     | `--patience_e`                    | Number of epochs to wait before early stopping is triggered.                                                   | 10                       |
| `-pr`     | `--patience_r`                    | Number of epochs to wait before learning rate reduction is triggered.                                          | 5                        |
| `-lr`     | `--learning_rate`                 | Learning rate used by the Adam optimizer.                                                                      | 0.000004218361045        |
| `-lral`   | `--learning_rate-active-learning` | Learning rate used by the Adam optimizer after active learning.                                                | 0.0000004                |
| `-ab1`    | `--adam_beta1`                    | Beta 1 Hyperparameter for Adam optimizer.                                                                      | 0.906450740008503        |
| `-ab2`    | `--adam_beta2`                    | Beta 2 Hyperparameter for Adam optimizer.                                                                      | 0.986390310777448        |
| `-bf`     | `--balance_factor`                | Multiplier for balancing datasets based on class sizes.                                                        | 1.24895925434138         |
| `-vf`     | `--validation_fraction`           | Proportion of tiles used for validation.                                                                       | 20%                      |
| `-net`    | `--network`                       | Name of the pre-trained network to use for training for AM.                                                    | am_252_efficientnet.pth  |
| `-neterm` | `--network_erm`                   | Name of the pre-trained network to use for training for ErM.                                                   | erm_126_efficientnet.pth |
| `-mt`     | `--model_type`                    | Choice for new model initialization.                                                                           | efficientnet             |
| `-pretr`  | `--pretrain`                      | Loads ImageNet weights if specific Model type is selected.                                                     | True                     |
| `-ct`     | `--colonisation_type`             | Choosing between Arbuscular and Ericoid colonisation.                                                          | am                       |
| `-size`   | `--tile_size`                     | Tile size (in pixels) used for image segmentation.                                                             | 252                      |

Training outputs are saved to the user directory unless the `outdir` parameter is specified. The training outputs are stored within a zip folder, with a naming convention that contains the date and time the folder was created.

### Prediction mode

This mode is used to predict fungal colonisation (CNN1) and intraradical hyphal structures (CNN2).

| Short     | Long                                | Description                                                              | Default value                              |
| --------- | ----------------------------------- | ------------------------------------------------------------------------ | ------------------------------------------ |
| `-l`      | `--use-csvs`                        | Use CSVs instead of DB.                                                  | False                                      |
| `-i`      | `--images`                          | **Mandatory**. Directory of images to process.                           | None (required argument)                   |
| `-size`   | `--tile_size`                       | Tile size (in pixels) used for image segmentation.                       | 252                                        |
| `-net`    | `--network`                         | Name of the pre-trained model to use for predictions for AM.             | am_252_efficientnet.pth                    |
| `-neterm` | `--network_erm`                     | Name of the pre-trained model to use for predictions for ErM.            | erm_126_efficientnet.pth                   |
| `-o`      | `--outdir`                          | Where to store results metrics files.                                    | Same as input images                       |
| `-ct`     | `--colonisation_type`               | Choosing between Arbuscular and Ericoid colonisation.                    | am                                         |
| `-t`      | `--temperature-factor-path`         | Name of the file containing the temperature factor for the AM model.     | am_252_efficientnet_temperature_value.txt  |
| `-term`   | `--temperature-factor-path-erm`     | Name of the file containing the temperature factor for the ErM model.    | erm_126_efficientnet_temperature_value.txt |
| `-ucc`    | `--use-contextual-confidence`       | Enable contextual confidence refinement for low confidence predictions.  | False                                      |
| `-cct`    | `--contextual-confidence-threshold` | Threshold below which max voting with surrounding tiles will be applied. | 1                                          |

Pre-trained networks to be used with the parameter `-net` are available in the folder [`trained_networks`](amf/trained_networks). You can also specify an absolute file path to this parameter in order to use networks outside the [`trained_networks`](amf/trained_networks) directory. Networks for both AM and ErM are provided - more detail on this can be found in doc/mycorrhiza-finder-user-documentation.pdf.

### Colonisation mode

This mode is used to calculate colonisation metrics for annotations. The metrics differ by colonisation type.

| Short     | Long                  | Description                                                   | Default value            |
| --------- | --------------------- | ------------------------------------------------------------- | ------------------------ |
| `-l`      | `--use-csvs`          | Use CSVs instead of DB.                                       | False                    |
| `-i`      | `--images`            | **Mandatory**. Directory of images to process.                | None (required argument) |
| `-net`    | `--network`           | Name of the pre-trained model to use for predictions for AM.  | am_252_efficientnet.pth  |
| `-neterm` | `--networkerm`        | Name of the pre-trained model to use for predictions for ErM. | erm_126_efficientnet.pth |
| `-o`      | `--outdir`            | Folder where to save trained model and CNN architecture.      | None                     |
| `-ct`     | `--colonisation_type` | Choosing between Abuscular and Ericoid colonisation.          | am                       |
| `-size`   | `--tile_size`         | Tile size (in pixels) used for image segmentation.            | 252                      |

### Conversion mode

This mode is used to convert `amf predict` predictions (i.e. probabilities) to annotations.

| Short   | Long                          | Description                                                                                        | Default value        |
| ------- | ----------------------------- | -------------------------------------------------------------------------------------------------- | -------------------- |
| `-l`    | `--use-csvs`                  | Use CSVs instead of DB.                                                                            | False                |
| `-th`   | `--threshold`                 | Threshold for conversion.                                                                          | 0.5                  |
| `-1`    | `--CNN1`                      | Convert root colonisation predictions (default).                                                   | N/A                  |
| `-2`    | `--CNN2`                      | Convert fungal hyphal structure predictions.                                                       | N/A                  |
| `-i`    | `--images`                    | Directory of the plant root images to process.                                                     | None (default empty) |
| `-ct`   | `--colonisation_type`         | Choosing between Arbuscular and Ericoid colonisation.                                              | am                   |
| `-agg`  | `--aggregate`                 | Argument to trigger tile upscaling by factor of 2.                                                 | False                |
| `-size` | `--tile_size`                 | Tile size (in pixels) used for image segmentation.                                                 | 252                  |
| `-ucc`  | `--use-contextual-confidence` | Enable contextual confidence refinement for low confidence predictions.                            | False                |
| `-c`    | `--collapse`                  | Collapse classes for AM and ErM. Note this is for dev purposes only and is NOT included in the UI. | False                |

### TIF Conversion mode

This is used to convert TIFs with corresponding SlideViewer XML annotations to JPGs or PNGs.

| Short   | Long                  | Description                                                 | Default value        |
| ------- | --------------------- | ----------------------------------------------------------- | -------------------- |
| `-l`    | `--use-csvs`          | Use CSVs instead of DB.                                     | False                |
| `-i`    | `--images`            | Directory of the plant root images to process.              | None (default empty) |
| `-ct`   | `--colonisation_type` | Choosing between Arbuscular and Ericoid colonisation.       | am                   |
| `-it`   | `--image-type`        | File type to convert TIFs to, choosing between jpg and png. | jpg                  |
| `-size` | `--tile_size`         | Tile size (in pixels) used for image segmentation.          | 252                  |

### Test mode

This mode is used to obtain metrics on the roots used to test the model.

| Short     | Long                                | Description                                                              | Default value                              |
| --------- | ----------------------------------- | ------------------------------------------------------------------------ | ------------------------------------------ |
| `-l`      | `--use-csvs`                        | Use CSVs instead of DB.                                                  | False                                      |
| `-i`      | `--images`                          | **Mandatory**. Directory of images to process.                           | None (required argument)                   |
| `-net`    | `--network`                         | Name of the pre-trained model to use for predictions for AM.             | am_252_efficientnet.pth                    |
| `-neterm` | `--network_erm`                     | Name of the pre-trained model to use for predictions for ErM.            | erm_126_efficientnet.pth                   |
| `-o`      | `--outdir`                          | Folder where to save trained model and CNN architecture.                 | None                                       |
| `-ct`     | `--colonisation_type`               | Choosing between Abuscular and Ericoid colonisation.                     | am                                         |
| `-smi`    | `--semi_supervised`                 | Enables semi-supervised training.                                        | False                                      |
| `-size`   | `--tile_size`                       | Tile size (in pixels) used for image segmentation.                       | 252                                        |
| `-f`      | `--fixmatch_results_directory`      | Folder where fixmatch results are stored.                                | "" (empty string)                          |
| `-t`      | `--temperature-factor-path`         | Name of the file containing the temperature factor for the AM model.     | am_252_efficientnet_temperature_value.txt  |
| `-ucc`    | `--use-contextual-confidence`       | Enable contextual confidence refinement for low confidence predictions.  | False                                      |
| `-cct`    | `--contextual-confidence-threshold` | Threshold below which max voting with surrounding tiles will be applied. | 1                                          |
| `-term`   | `--temperature-factor-path-erm`     | Name of the file containing the temperature factor for the ErM model.    | erm_126_efficientnet_temperature_value.txt |

Unless specified by the `outdir` parameter, the test outputs are saved to the user's default directory. The outputs are organised into a zip folder, named according to the date and time of creation. Inside the zip folder, individual subfolders are created for each root, each containing the tiles that were incorrectly predicted. Additionally, the following files are included in the output:

- **generic_metrics.csv** - overall test metrics (precision, recall, F1 score) and loss
- **file_metrics.csv** - metrics for each file
- **confusion \_matrix.png** - confusion matrix
- **class_metrics.csv** - metrics for each class
- **class_metrics_after_manual_labelling** - this includes the metrics for the model if we were to relabel every file under a certain confidence level with the correct label. There are excels for each confidence step 0-1 in 0.1 intervals. This emulates manual labelling of low confidence predictions.
- **tile_class_changes.csv** - this gives an overview of the classes that were changed if contextual predictions were used.
- **num_tiles_per_threshold.csv** - this gives the percentage of tiles in the test set that has the maximum confidence below a certain threshold, with thresholds of 0-1 in 0.1 step intervals.

### Diagnostic mode

NOTE: diagnose needs further development work for productive use and as such is not included in the UI.

Diagnose can be used to compare predictions and annotations and output statistic regarding their differences. The settings are as follows.

| Short     | Long                  | Description                                              | Default value            |
| --------- | --------------------- | -------------------------------------------------------- | ------------------------ |
| `-l`      | `--use-csvs`          | Use CSVs instead of DB.                                  | False                    |
| `-net`    | `--network`           | Name of pre-trained model to use for diagnostic for AM.  | am_252_efficientnet.pth  |
| `-neterm` | `--network_erm`       | Name of pre-trained model to use for diagnostic for ErM. | erm_126_efficientnet.pth |
| `-i`      | `--images`            | **Mandatory**. Directory of images to process.           | None (required argument) |
| `-o`      | `--outdir`            | Folder where to save trained model and CNN architecture. | None                     |
| `-ct`     | `--colonisation_type` | Choosing between Abuscular and Ericoid colonisation.     | am                       |

### Calibration mode

This is used to a calibrate a trained model to align output of model to likelihood by calculating a temperature factor.

| Short     | Long                  | Description                                              | Default value            |
| --------- | --------------------- | -------------------------------------------------------- | ------------------------ |
| `-l`      | `--use-csvs`          | Use CSVs instead of DB.                                  | False                    |
| `-net`    | `--network`           | Name of the model to be calibrated for AM.               | am_252_efficientnet.pth  |
| `-neterm` | `--network_erm`       | Name of the model to be calibrated for ErM.              | erm_126_efficientnet.pth |
| `-i`      | `--images`            | **Mandatory**. Directory of images to calibrate on.      | None (required argument) |
| `-o`      | `--outdir`            | Folder where to save trained model and CNN architecture. | None                     |
| `-ct`     | `--colonisation_type` | Choosing between Arbuscular and Ericoid colonisation.    | am                       |
| `-size`   | `--tile_size`         | Tile size (in pixels) used for image segmentation.       | 252                      |

#### AzureML Training Runs

One can also carry out training runs on AzureML, and notebooks for this are provided, which start with azure\_\*.ipynb.

These notebooks are meant for usage on a dedicated Microsoft Azure workspace to train neural network architectures on resources on cloud infrastructure.
To run these notebooks, the requirements are a Microsoft Azure subscription including quota, an correctly setup Docker environment as well as a blob storage containing the training and test data meant for training.
If the requrements are met, the notebooks can be used by individual cell execution:

- Pipeline configuration sets up core variables such as the ML Client including credentials, AzureML tracking, and other pipeline configurations
- Default paths maps the paths of the input and output for the given AMFinder task to be conducted on Azure
- Environment defines the AzureML environment that is meant to be used for the given task. The commented sectin acts as an example on how an enviornment can be setup using the PythonSDK, but does not reflect the final environment configuration used for training.
- Pipeline step definition: Defines the pipeline step displayed in Azure and passes the command ought to be executed in the desired Azure environment. In our setup, it is simlar to executing amf through a terminal, but requires an explicit execute in the sub-virtual environment requirements_aml within docker that hold all required dependencies.
  Finally a pipeline is defined.
- Sweeps on the other hand contain intervals from where a RandomSelection is conducted to trigger multiple runs. Subsequent steps define a truncation policy for improved sweep efficieny, so that low-performing runs are cancelled early. Note that the EarlyStopping function has to be disabled in train to ensure the truncation policy is functional.

There are also a Dockerfile and yml file for python requirements - these can be found in amf/aml_environment_files.

#### Required Dataset Structure for Training

The training scripts expect the dataset to be organised as follows:

`/path/to/dataset/folder` should contain `train` and `test` subdirectories. These folders contain the roots to train the neural network.

Each folder should contain roots, separated into different folders for different roots. When annotating roots, the annotations are stored in the Postgres Database using the name of the image as an ID. Therefore, you must not change the name of an image once you have annotated it, otherwise this will lead to the associated annotations not being recognised by the program.
