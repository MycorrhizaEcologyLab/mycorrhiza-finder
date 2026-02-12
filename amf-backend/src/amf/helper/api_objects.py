"""Structures used to pass data between frontend and backend."""

from pydantic import BaseModel


class BaseValues(BaseModel):
    """Base class for annotation and prediction values for a single image."""

    fileName: str | None = None
    imageReferenceId: int | None = None
    colonisationType: str | None = "am"
    tileEdge: int | None = 252
    enabled: bool | None = False


class AnnotationValues(BaseValues):
    """Annotations for a single image.

    Outer list is tiles, inner list is values for each tile as they appear in the DB:
    row, column, then each class in one-hot form, plus question comment (hence str).
    """

    cnnOneValues: list[list[int | str]]


class PredictionValues(BaseValues):
    """Predictions for a single image.

    Same structure as AnnotationValues, but with floats for class probabilities and str
    for contextual class prediction, or None if no contextual class prediction.
    """

    cnnOneValues: list[list[float | str | None]]


class BaseConfig(BaseModel):
    """Base class for configuration options for the various modes of the tool."""

    inputFiles: str
    outdir: str | None = None
    colonisationType: str | None = "am"
    useDb: bool | None = True
    device: str | None = "automatic"
    tileEdge: int | None = 252


class PredictionConfig(BaseConfig):
    """Config options for `predict` mode."""

    model: str | None = "efficientnet_252_6class_D2.pth"
    modelErm: str | None = "250411_126_ErM_EfficientNet_82.pth"
    temperatureFactorPath: str | None = (
        "efficientnet_252_6class_D2_temperature_value.txt"
    )
    useContextualConfidence: bool | None = True
    contextualConfidenceThreshold: float | None = 1
    temperatureFactorPathErm: str | None = (
        "250411_126_ErM_EfficientNet_82_temperature_value.txt"
    )


class TrainConfig(BaseConfig):
    """Config options for `train` mode."""

    batchSize: int | None = 32
    learningRate: float | None = 0.001
    learningRateActiveLearning: float | None = 0.000001
    getTilesForLabellingUsingActiveLearning: bool | None = False
    activeLearningMethod: str | None = "bald"
    numSamplesForLabelling: int | None = 10
    mcSamples: int | None = 50
    dropoutRate: float | None = 0.25
    adamBeta1: float | None = 0.9
    adamBeta2: float | None = 0.999
    balanceFactor: float | None = 1.0
    epochs: int | None = 50
    epochsActiveLearning: int | None = 5
    model: str | None = "efficientnet_252_6class_D2.pth"
    modelErm: str | None = "250411_126_ErM_EfficientNet_82.pth"
    modelType: str | None = "cnn1"
    preTrained: bool | None = False
    vfrac: float | None = 0.2
    dataAugm: bool | None = False
    summary: bool | None = False
    mlflowFlag: bool | None = False
    patienceE: int | None = 10
    patienceR: int | None = 5
    trainActiveLearning: int | None = False


class TestConfig(BaseConfig):
    """Config options for `test` mode."""

    model: str | None = "efficientnet_252_6class_D2.pth"
    modelErm: str | None = "250411_126_ErM_EfficientNet_82.pth"
    temperatureFactorPath: str | None = (
        "efficientnet_252_6class_D2_temperature_value.txt"
    )
    temperatureFactorPathErm: str | None = (
        "250411_126_ErM_EfficientNet_82_temperature_value.txt"
    )
    useContextualConfidence: bool | None = True
    contextualConfidenceThreshold: float | None = 1


class CalibrateConfig(BaseConfig):
    """Config options for `calibrate` mode."""

    model: str | None = "efficientnet_252_6class_D2.pth"
    modelErm: str | None = "250411_126_ErM_EfficientNet_82.pth"


class ConvertConfig(BaseConfig):
    """Config options for `convert` mode."""

    threshold: float | None = 0.5
    aggregateTiles: bool | None = False
    useContextualConfidence: bool | None = True


class TifConversionConfig(BaseConfig):
    """Config options for `tifconversion` mode."""

    convertImageFileType: str | None = "jpg"


class TileEdgeConfig(BaseModel):
    """Structure for passing tile edge length in API calls."""

    tileEdge: int


class Setting(BaseModel):
    """Structure for passing a single setting key-value pair in API calls."""

    key: str
    value: str | None


class Settings(BaseModel):
    """All settings that are persisted in the database."""

    outdir: str | None = None
    useDb: bool | None = True
    device: str | None = "automatic"
    tileEdge: int | None = 252
    batchSize: int | None = 32
    learningRate: float | None = 0.001
    learningRateActiveLearning: float | None = 0.000001
    getTilesForLabellingUsingActiveLearning: bool | None = False
    activeLearningMethod: str | None = "bald"
    numSamplesForLabelling: int | None = 10
    mcSamples: int | None = 50
    dropoutRate: float | None = 0.25
    adamBeta1: float | None = 0.9
    adamBeta2: float | None = 0.999
    balanceFactor: float | None = 1.0
    epochs: int | None = 50
    epochsActiveLearning: int | None = 5
    model: str | None = "efficientnet_252_6class_D2.pth"
    modelErm: str | None = "250411_126_ErM_EfficientNet_82.pth"
    modelType: str | None = "cnn1"
    preTrained: bool | None = False
    vfrac: float | None = 0.2
    dataAugm: bool | None = False
    summary: bool | None = False
    mlflowFlag: bool | None = False
    patienceE: int | None = 10
    patienceR: int | None = 5
    trainActiveLearning: int | None = False
    temperatureFactorPath: str | None = (
        "efficientnet_252_6class_D2_temperature_value.txt"
    )
    temperatureFactorPathErm: str | None = (
        "250411_126_ErM_EfficientNet_82_temperature_value.txt"
    )
    threshold: float | None = 0.5
    aggregateTiles: bool | None = False
    convertImageFileType: str | None = "jpg"
