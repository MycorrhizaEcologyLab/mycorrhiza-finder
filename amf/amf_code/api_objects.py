from typing import List, Optional
from typing import List, Optional, Union
from pydantic import BaseModel


class BaseValues(BaseModel):
    fileName: Optional[str] = None
    imageReferenceId: Optional[int] = None
    colonisationType: Optional[str] = "am"
    tileEdge: Optional[int] = 252
    enabled: Optional[bool] = False


class AnnotationValues(BaseValues):
    cnnOneValues: List[List[Union[int, str]]]
    cnnTwoValues: List[List[int]]


class PredictionValues(BaseValues):
    cnnOneValues: List[List[Union[float, str, None]]]
    cnnTwoValues: List[List[float]]


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
    temperatureFactorPath: Optional[str] = (
        "efficientnet_252_6class_D2_temperature_value.txt"
    )
    useContextualConfidence: Optional[bool] = True
    contextualConfidenceThreshold: Optional[float] = 1
    temperatureFactorPathErm: Optional[str] = (
        "250411_126_ErM_EfficientNet_82_temperature_value.txt"
    )


class TrainConfig(BaseConfig):
    batchSize: Optional[int] = 32
    learningRate: Optional[float] = 0.001
    learningRateActiveLearning: Optional[float] = 0.000001
    getTilesForLabellingUsingActiveLearning: Optional[bool] = False
    activeLearningMethod: Optional[str] = "bald"
    numSamplesForLabelling: Optional[int] = 10
    mcSamples: Optional[int] = 50
    dropoutRate: Optional[float] = 0.25
    adamBeta1: Optional[float] = 0.9
    adamBeta2: Optional[float] = 0.999
    balanceFactor: Optional[float] = 1.0
    epochs: Optional[int] = 50
    epochsActiveLearning: Optional[int] = 5
    model: Optional[str] = "efficientnet_252_6class_D2.pth"
    modelErm: Optional[str] = "250411_126_ErM_EfficientNet_82.pth"
    modelType: Optional[str] = "cnn1"
    preTrained: Optional[bool] = False
    level: Optional[int] = 1
    vfrac: Optional[float] = 0.2
    dataAugm: Optional[bool] = False
    summary: Optional[bool] = False
    semiSupervised: Optional[bool] = False
    mlflowFlag: Optional[bool] = False
    patienceE: Optional[int] = 10
    patienceR: Optional[int] = 5
    trainActiveLearning: Optional[int] = False


class TestConfig(BaseConfig):
    model: Optional[str] = "efficientnet_252_6class_D2.pth"
    modelErm: Optional[str] = "250411_126_ErM_EfficientNet_82.pth"
    temperatureFactorPath: Optional[str] = (
        "efficientnet_252_6class_D2_temperature_value.txt"
    )
    temperatureFactorPathErm: Optional[str] = (
        "250411_126_ErM_EfficientNet_82_temperature_value.txt"
    )
    semiSupervised: Optional[bool] = False
    fixmatchResultsDirectory: Optional[str] = ""
    useContextualConfidence: Optional[bool] = True
    contextualConfidenceThreshold: Optional[float] = 1


class CalibrateConfig(BaseConfig):
    model: Optional[str] = "efficientnet_252_6class_D2.pth"
    modelErm: Optional[str] = "250411_126_ErM_EfficientNet_82.pth"


class ConvertConfig(BaseConfig):
    level: Optional[int] = 1
    threshold: Optional[float] = 0.5
    aggregateTiles: Optional[bool] = False
    useContextualConfidence: Optional[bool] = True


class TifConversionConfig(BaseConfig):
    convertImageFileType: Optional[str] = "jpg"


class TileEdgeConfig(BaseModel):
    tileEdge: int


class Setting(BaseModel):
    key: str
    value: Optional[str]


class Settings(BaseModel):
    outdir: Optional[str] = None
    useDb: Optional[bool] = True
    device: Optional[str] = "automatic"
    tileEdge: Optional[int] = 252
    batchSize: Optional[int] = 32
    learningRate: Optional[float] = 0.001
    learningRateActiveLearning: Optional[float] = 0.000001
    getTilesForLabellingUsingActiveLearning: Optional[bool] = False
    activeLearningMethod: Optional[str] = "bald"
    numSamplesForLabelling: Optional[int] = 10
    mcSamples: Optional[int] = 50
    dropoutRate: Optional[float] = 0.25
    adamBeta1: Optional[float] = 0.9
    adamBeta2: Optional[float] = 0.999
    balanceFactor: Optional[float] = 1.0
    epochs: Optional[int] = 50
    epochsActiveLearning: Optional[int] = 5
    model: Optional[str] = "efficientnet_252_6class_D2.pth"
    modelErm: Optional[str] = "250411_126_ErM_EfficientNet_82.pth"
    modelType: Optional[str] = "cnn1"
    preTrained: Optional[bool] = False
    level: Optional[int] = 1
    vfrac: Optional[float] = 0.2
    dataAugm: Optional[bool] = False
    summary: Optional[bool] = False
    semiSupervised: Optional[bool] = False
    mlflowFlag: Optional[bool] = False
    patienceE: Optional[int] = 10
    patienceR: Optional[int] = 5
    trainActiveLearning: Optional[int] = False
    temperatureFactorPath: Optional[str] = (
        "efficientnet_252_6class_D2_temperature_value.txt"
    )
    temperatureFactorPathErm: Optional[str] = (
        "250411_126_ErM_EfficientNet_82_temperature_value.txt"
    )
    semiSupervised: Optional[bool] = False
    fixmatchResultsDirectory: Optional[str] = ""
    level: Optional[int] = 1
    threshold: Optional[float] = 0.5
    aggregateTiles: Optional[bool] = False
    convertImageFileType: Optional[str] = "jpg"
