from datetime import datetime
from typing import Dict, List, Optional, Union

from pydantic import BaseModel


class BaseValues(BaseModel):
    fileName: Optional[str] = None
    # int when saved to the database; a local CSV filename (str) when in
    # local (no-DB) mode.
    imageReferenceId: Optional[Union[int, str]] = None
    colonisationType: Optional[str] = "am"
    tileEdge: Optional[int] = 252
    enabled: Optional[bool] = False
    # Only set by the batch folder importer, to carry the timestamp parsed out
    # of the CSV's filename through to ImageReference.UploadTimestamp. Left
    # None everywhere else, so the column keeps its CURRENT_TIMESTAMP default.
    uploadTimestamp: Optional[datetime] = None


class AnnotationValues(BaseValues):
    cnnOneValues: List[List[Union[int, str]]]


class PredictionValues(BaseValues):
    cnnOneValues: List[List[Union[float, str, None]]]


class BaseConfig(BaseModel):
    inputFiles: str
    outdir: Optional[str] = None
    colonisationType: Optional[str] = "am"
    useDb: Optional[bool] = True
    device: Optional[str] = "automatic"
    tileEdge: Optional[int] = 252


class PredictionConfig(BaseConfig):
    model: Optional[str] = "efficientnet_252_6class_D2.pth"
    modelErm: Optional[str] = "250411_126_ErM_EfficientNet_82.pth"
    modelPath: Optional[str] = None
    temperatureFactorPath: Optional[str] = (
        "efficientnet_252_6class_D2_temperature_value.txt"
    )
    useContextualConfidence: Optional[bool] = True
    contextualConfidenceThreshold: Optional[float] = 1
    temperatureFactorPathErm: Optional[str] = (
        "250411_126_ErM_EfficientNet_82_temperature_value.txt"
    )
    resizeDim: Optional[int] = None
    ciMethod: Optional[str] = "analytic"


class TrainConfig(BaseConfig):
    batchSize: Optional[int] = 32
    learningRate: Optional[float] = 0.001
    learningRateActiveLearning: Optional[float] = 0.000001
    getTilesForLabellingUsingActiveLearning: Optional[bool] = False
    activeLearningMethod: Optional[str] = "bald"
    numSamplesForLabelling: Optional[int] = 10
    mcSamples: Optional[int] = 50
    dropoutRate: Optional[float] = 0.0
    adamBeta1: Optional[float] = 0.9
    adamBeta2: Optional[float] = 0.999
    balanceFactor: Optional[float] = 1.0
    epochs: Optional[int] = 50
    epochsActiveLearning: Optional[int] = 5
    model: Optional[str] = "efficientnet_252_6class_D2.pth"
    modelErm: Optional[str] = "250411_126_ErM_EfficientNet_82.pth"
    modelType: Optional[str] = "efficientnet"
    preTrained: Optional[bool] = True
    vfrac: Optional[float] = 0.2
    dataAugm: Optional[bool] = False
    summary: Optional[bool] = False
    mlFlowFlag: Optional[bool] = False
    patienceE: Optional[int] = 10
    patienceR: Optional[int] = 5
    trainActiveLearning: Optional[int] = False
    filterBackground: Optional[bool] = False
    dynamicLoading: Optional[bool] = False
    pretiledDir: Optional[str] = None
    checkpointPath: Optional[str] = None
    resizeDim: Optional[int] = None
    weightDecay: Optional[float] = 0.0
    backboneLrMult: Optional[float] = 1.0
    freezeEpochs: Optional[int] = 0
    dropPathRate: Optional[float] = 0.0
    earlyBreakEpoch: Optional[int] = None


class TestConfig(BaseConfig):
    model: Optional[str] = "efficientnet_252_6class_D2.pth"
    modelErm: Optional[str] = "250411_126_ErM_EfficientNet_82.pth"
    modelPath: Optional[str] = None
    temperatureFactorPath: Optional[str] = (
        "efficientnet_252_6class_D2_temperature_value.txt"
    )
    temperatureFactorPathErm: Optional[str] = (
        "250411_126_ErM_EfficientNet_82_temperature_value.txt"
    )
    useContextualConfidence: Optional[bool] = True
    contextualConfidenceThreshold: Optional[float] = 1
    resizeDim: Optional[int] = None


class CalibrateConfig(BaseConfig):
    model: Optional[str] = "efficientnet_252_6class_D2.pth"
    modelErm: Optional[str] = "250411_126_ErM_EfficientNet_82.pth"
    dynamicLoading: Optional[bool] = False
    pretiledDir: Optional[str] = None
    modelPath: Optional[str] = None
    resizeDim: Optional[int] = None


class ColonisationConfig(BaseConfig):
    resizeDim: Optional[int] = None


class ConvertConfig(BaseConfig):
    threshold: Optional[float] = 0.5
    aggregateTiles: Optional[bool] = False
    useContextualConfidence: Optional[bool] = True


class TifConversionConfig(BaseConfig):
    convertImageFileType: Optional[str] = "jpg"


class TileEdgeConfig(BaseModel):
    tileEdge: int


class ImportFolderRequest(BaseModel):
    folderPath: str
    colonisationType: Optional[str] = "am"


class Setting(BaseModel):
    key: str
    value: Optional[str]


class TagPalette(BaseModel):
    # Annotation sub-tag name -> CSS colour. Persisted as JSON in the single
    # 'tagPalette' Settings row
    colours: Dict[str, str] = {}


class Settings(BaseModel):
    outdir: Optional[str] = None
    imageDirectory: Optional[str] = None
    useDb: Optional[bool] = True
    device: Optional[str] = "automatic"
    tileEdge: Optional[int] = 252
    numWorkers: Optional[int] = 0
    batchSize: Optional[int] = 32
    learningRate: Optional[float] = 0.001
    learningRateActiveLearning: Optional[float] = 0.000001
    getTilesForLabellingUsingActiveLearning: Optional[bool] = False
    activeLearningMethod: Optional[str] = "bald"
    numSamplesForLabelling: Optional[int] = 10
    mcSamples: Optional[int] = 50
    dropoutRate: Optional[float] = 0.0
    dropPathRate: Optional[float] = 0.0
    adamBeta1: Optional[float] = 0.9
    adamBeta2: Optional[float] = 0.999
    balanceFactor: Optional[float] = 1.0
    epochs: Optional[int] = 50
    epochsActiveLearning: Optional[int] = 5
    model: Optional[str] = "efficientnet_252_6class_D2.pth"
    modelErm: Optional[str] = "250411_126_ErM_EfficientNet_82.pth"
    modelPath: Optional[str] = None
    modelType: Optional[str] = "efficientnet"
    preTrained: Optional[bool] = True
    vfrac: Optional[float] = 0.2
    dataAugm: Optional[bool] = False
    summary: Optional[bool] = False
    mlFlowFlag: Optional[bool] = False
    patienceE: Optional[int] = 10
    patienceR: Optional[int] = 5
    trainActiveLearning: Optional[int] = False
    filterBackground: Optional[bool] = False
    dynamicLoading: Optional[bool] = False
    pretiledDir: Optional[str] = None
    checkpointPath: Optional[str] = None
    resizeDim: Optional[int] = None
    weightDecay: Optional[float] = 0.0
    backboneLrMult: Optional[float] = 1.0
    freezeEpochs: Optional[int] = 0
    earlyBreakEpoch: Optional[int] = None
    temperatureFactorPath: Optional[str] = (
        "efficientnet_252_6class_D2_temperature_value.txt"
    )
    temperatureFactorPathErm: Optional[str] = (
        "250411_126_ErM_EfficientNet_82_temperature_value.txt"
    )
    useContextualConfidence: Optional[bool] = True
    contextualConfidenceThreshold: Optional[float] = 1
    ciMethod: Optional[str] = "analytic"
    threshold: Optional[float] = 0.5
    aggregateTiles: Optional[bool] = False
    convertImageFileType: Optional[str] = "jpg"
