import ErrorOutlineOutlinedIcon from "@mui/icons-material/ErrorOutlineOutlined";
import InfoOutlinedIcon from "@mui/icons-material/InfoOutlined";
import Divider from "@mui/material/Divider";
import TextField from "@mui/material/TextField";
import Tooltip from "@mui/material/Tooltip";
import { useContext, useEffect, useState } from "react";
import { toast } from "react-toastify";
import PredictionsApi from "../../api/amfinderApi";
import { GlobalContextProvider } from "../../contexts/Contexts";
import PrimaryButton from "../Utils/PrimaryButton";
import { isValidFilePath } from "../Utils/utils";
import ValidationIcon from "../Utils/ValidationIcon";
import AmfToolActionButtons from "./AmfToolActionButtons";

import "./styles/amftool.css";

export default function AmfToolContainer() {
  const {
    colonisationType,
    tileEdge,
    setTileEdge,
    settings,
    setSettings,
    amfToolFilePath,
    setAmfToolFilePath,
  } = useContext(GlobalContextProvider);

  const [isFilePathValid, setIsFilePathValid] = useState(false);
  const [selectedAction, setSelectedAction] = useState(null);
  const [submissionStatus, setSubmissionStatus] = useState("none");

  const filePathValidationStatus = isFilePathValid ? "success" : "error";

  const actionsValidationStatus =
    selectedAction && isFilePathValid ? "success" : "none";

  useEffect(() => {
    if (colonisationType === "am") {
      setTileEdge(252);
    } else {
      setTileEdge(126);
    }
  }, [colonisationType]);

  useEffect(() => {
    const valid = isValidFilePath(amfToolFilePath);
    setIsFilePathValid(valid);
  }, [amfToolFilePath]);

  const fetchInitialSettings = async () => {
    const initialSettings = await PredictionsApi.getAllSettings();
    setSettings(initialSettings);
  };

  useEffect(() => {
    if (!settings || Object.keys(settings ?? {}).length === 0) {
      fetchInitialSettings();
    }
  }, []);

  const trainModel = async () => {
    const toastId = "trainModel";

    let body = {
      inputFiles: amfToolFilePath,
      batchSize: settings.batchSize,
      epochs: settings.epochs,
      model: settings.model,
      modelErm: settings.modelErm,
      vfrac: settings.vfrac,
      dataAugm: settings.dataAugm,
      summary: settings.summary,
      outdir: settings.outdir,
      colonisationType: colonisationType,
      useDb: settings.useDb,
      learningRate: settings.learningRate,
      adamBeta1: settings.adamBeta1,
      adamBeta2: settings.adamBeta2,
      balanceFactor: settings.balanceFactor,
      mlFlowFlag: settings.mlFlowFlag,
      device: settings.device,
      modelType: settings.modelType,
      preTrained: settings.preTrained,
      patienceE: settings.patienceE,
      patienceR: settings.patienceR,
      learningRateActiveLearning: settings.learningRateActiveLearning,
      getTilesForLabellingUsingActiveLearning:
        settings.getTilesForLabellingUsingActiveLearning,
      activeLearningMethod: settings.activeLearningMethod,
      numSamplesForLabelling: settings.numSamplesForLabelling,
      mcSamples: settings.mcSamples,
      dropoutRate: settings.dropoutRate,
      epochsActiveLearning: settings.epochsActiveLearning,
      trainActiveLearning: settings.trainActiveLearning,
      tileEdge: tileEdge,
    };

    toast.info("Training model", {
      toastId,
      autoClose: false,
    });

    setSubmissionStatus("submitting");
    let res = await PredictionsApi.trainModel(body);

    if (res === 200) {
      setSubmissionStatus("success");
      toast.update(toastId, {
        type: "success",
        render: "Model succesfully trained",
        autoClose: 5000,
      });
    } else {
      setSubmissionStatus("error");
      toast.update(toastId, {
        type: "error",
        render: "Error training model",
        autoClose: 5000,
      });
    }
  };

  const calculatePredictions = async () => {
    const toastId = "calculatePredictions";

    let body = {
      inputFiles: amfToolFilePath,
      tileEdge: tileEdge,
      model: settings.model,
      modelErm: settings.modelErm,
      outdir: settings.outdir,
      colonisationType: colonisationType,
      useDb: settings.useDb,
      device: settings.device,
      temperatureFactorPath: settings.temperatureFactorPath,
      temperatureFactorPathErm: settings.temperatureFactorPathErm,
      useContextualConfidence: settings.useContextualConfidence,
      contextualConfidenceThreshold: settings.contextualConfidenceThreshold,
    };

    toast.info("Calculating predictions", {
      toastId,
      autoClose: false,
    });

    setSubmissionStatus("submitting");
    let res = await PredictionsApi.calculatePredictions(body);

    if (res === 200) {
      setSubmissionStatus("success");
      toast.update(toastId, {
        type: "success",
        render: "Predictions successfully calculated",
        autoClose: 5000,
      });
    } else {
      setSubmissionStatus("error");
      toast.update(toastId, {
        type: "error",
        render: "Error calculating predictions",
        autoClose: 5000,
      });
    }
  };

  const convertImages = async () => {
    const toastId = "convertImages";

    let body = {
      inputFiles: amfToolFilePath,
      threshold: settings.threshold,
      colonisationType: colonisationType,
      useDb: settings.useDb,
      aggregateTiles: settings.aggregateTiles,
      device: settings.device,
      outdir: settings.outdir,
      tileEdge: tileEdge,
      useContextualConfidence: settings.useContextualConfidence,
    };

    toast.info("Converting images", {
      toastId,
      autoClose: false,
    });

    setSubmissionStatus("submitting");
    let res = await PredictionsApi.convertImages(body);

    if (res === 200) {
      setSubmissionStatus("success");
      toast.update(toastId, {
        type: "success",
        render: "Images successfully converted",
        autoClose: 5000,
      });
    } else {
      setSubmissionStatus("error");
      toast.update(toastId, {
        type: "error",
        render: "Error converting images",
        autoClose: 5000,
      });
    }
  };

  const tifConversion = async () => {
    const toastId = "tifConversion";

    let body = {
      inputFiles: amfToolFilePath,
      colonisationType: colonisationType,
      useDb: settings.useDb,
      device: settings.device,
      outdir: settings.outdir,
      convertImageFileType: settings.convertImageFileType,
      tileEdge: tileEdge,
    };

    toast.info("Converting TIFs", {
      toastId,
      autoClose: false,
    });

    setSubmissionStatus("submitting");
    let res = await PredictionsApi.tifConversion(body);

    if (res === 200) {
      setSubmissionStatus("success");
      toast.update(toastId, {
        type: "success",
        render: "TIFs successfully converted",
        autoClose: 5000,
      });
    } else {
      setSubmissionStatus("error");
      toast.update(toastId, {
        type: "error",
        render: "Error converting TIFs",
        autoClose: 5000,
      });
    }
  };

  const testModel = async () => {
    const toastId = "testModel";

    let body = {
      inputFiles: amfToolFilePath,
      model: settings.model,
      modelErm: settings.modelErm,
      colonisationType: colonisationType,
      useDb: settings.useDb,
      device: settings.device,
      outdir: settings.outdir,
      tileEdge: tileEdge,
      temperatureFactorPath: settings.temperatureFactorPath,
      temperatureFactorPathErm: settings.temperatureFactorPathErm,
      useContextualConfidence: settings.useContextualConfidence,
      contextualConfidenceThreshold: settings.contextualConfidenceThreshold,
    };

    toast.info("Testing model", {
      toastId,
      autoClose: false,
    });

    setSubmissionStatus("submitting");
    let res = await PredictionsApi.testModel(body);

    if (res === 200) {
      setSubmissionStatus("success");
      toast.update(toastId, {
        type: "success",
        render: "Model successfully tested",
        autoClose: 5000,
      });
    } else {
      setSubmissionStatus("error");
      toast.update(toastId, {
        type: "error",
        render: "Error testing model",
        autoClose: 5000,
      });
    }
  };

  const calculateColonisation = async () => {
    const toastId = "calculateColonisation";

    let body = {
      inputFiles: amfToolFilePath,
      colonisationType: colonisationType,
      useDb: settings.useDb,
      device: settings.device,
      outdir: settings.outdir,
      tileEdge: tileEdge,
    };

    toast.info("Calculating colonisation percentage", {
      toastId,
      autoClose: false,
    });

    setSubmissionStatus("submitting");
    let res = await PredictionsApi.calculateColonisation(body);

    if (res === 200) {
      setSubmissionStatus("success");
      toast.update(toastId, {
        type: "success",
        render: "Successfully calculated colonisation percentages",
        autoClose: 5000,
      });
    } else {
      setSubmissionStatus("error");
      toast.update(toastId, {
        type: "error",
        render: "Error calculating colonisation percentages",
        autoClose: 5000,
      });
    }
  };

  const calibrateModel = async () => {
    const toastId = "calibrateModel";

    let body = {
      inputFiles: amfToolFilePath,
      model: settings.model,
      modelErm: settings.modelErm,
      colonisationType: colonisationType,
      useDb: settings.useDb,
      device: settings.device,
      outdir: settings.outdir,
      tileEdge: tileEdge,
    };

    toast.info("Calibrating model", {
      toastId,
      autoClose: false,
    });

    setSubmissionStatus("submitting");
    let res = await PredictionsApi.calibrateModel(body);

    if (res === 200) {
      setSubmissionStatus("success");
      toast.update(toastId, {
        type: "success",
        render: "Model successfully calibrated",
        autoClose: 5000,
      });
    } else {
      setSubmissionStatus("error");
      toast.update(toastId, {
        type: "error",
        render: "Error calibrating model",
        autoClose: 5000,
      });
    }
  };

  const handleSubmit = () => {
    switch (selectedAction) {
      case "train":
        trainModel();
        break;
      case "predict":
        calculatePredictions();
        break;
      case "convert":
        convertImages();
        break;
      case "test":
        testModel();
        break;
      case "calibrate":
        calibrateModel();
        break;
      case "colonisation":
        calculateColonisation();
        break;
      case "tifconversion":
        tifConversion();
        break;
      default:
        toast.error("Error submitting action");
        break;
    }
  };

  return (
    <div id="amfToolContainer" className="fullHeight fullWidth flexRowCenter">
      <div
        className="fullHeight fullWidth flexColumn"
        style={{ overflowY: "auto" }}
      >
        <div
          className="fullHeight fullWidth"
          style={{
            minWidth: "925px",
            minHeight: "500px",
          }}
        >
          <div className="fullHeight fullWidth">
            <div className="fullHeight fullWidth flexColumnCenter">
              <div
                className="fileNameWrapper fullWidth flexRow"
                style={{
                  height: "70px",
                  minHeight: "70px",
                  justifyContent: "center",
                  alignItems: "end",
                }}
              >
                <div
                  className="dummaryContainer"
                  style={{
                    width: "10%",
                    minWidth: "50px",
                    display: "flex",
                    alignItems: "center",
                  }}
                />
                <div className="flexRowCenter" style={{ width: "80%" }}>
                  {/* Empty div to ensure central alignment of TextField with connector */}
                  <div style={{ width: "24px" }} />
                  <TextField
                    id="fungal-image-path"
                    label="File path"
                    variant="outlined"
                    sx={{
                      width: "300px",
                      marginLeft: "10px",
                      marginRight: "10px",
                    }}
                    error={!isFilePathValid}
                    value={amfToolFilePath}
                    disabled={submissionStatus === "submitting"}
                    onChange={(event) => {
                      setAmfToolFilePath(event.currentTarget.value);
                    }}
                  />
                  {isFilePathValid && (
                    <Tooltip
                      id="filePathTooltip"
                      title="File path needs to be an absolute path to a folder e.g. C:\Test"
                    >
                      <InfoOutlinedIcon
                        sx={{ color: "black", width: "22px", height: "22px" }}
                      />
                    </Tooltip>
                  )}
                  {!isFilePathValid && (
                    <Tooltip
                      id="filePathTooltip"
                      title="File path needs to be an absolute path to a folder e.g. C:\Test"
                    >
                      <ErrorOutlineOutlinedIcon
                        sx={{ width: "22px", height: "22px" }}
                        color={"error"}
                      />
                    </Tooltip>
                  )}
                </div>
                <div
                  className="validationIconContainer"
                  style={{
                    paddingRight: "20px",
                    width: "10%",
                    minWidth: "50px",
                    display: "flex",
                    alignItems: "center",
                  }}
                >
                  <ValidationIcon status={filePathValidationStatus} />
                </div>
              </div>
              <div className="amfToolConnectorWrapper">
                <Divider
                  orientation="vertical"
                  variant="middle"
                  className="amfToolDivider"
                  flexItem
                  sx={{
                    backgroundColor: isFilePathValid ? "#1976d2" : "lightgrey",
                    marginTop: "16px",
                    marginBottom: "16px",
                    width: "2px",
                  }}
                />
              </div>
              <div
                className="actionWrapper"
                style={{
                  width: "100%",
                  height: "125px",
                  minHeight: "125px",
                  display: "flex",
                  justifyContent: "center",
                  alignItems: "center",
                }}
              >
                <div
                  className="dummaryContainer"
                  style={{
                    width: "10%",
                    minWidth: "50px",
                    display: "flex",
                    alignItems: "center",
                  }}
                />
                <div
                  style={{
                    width: "80%",
                    display: "flex",
                    justifyContent: "center",
                  }}
                >
                  <AmfToolActionButtons
                    selectedAction={selectedAction}
                    setSelectedAction={setSelectedAction}
                    isFilePathValid={isFilePathValid}
                    submissionStatus={submissionStatus}
                    setSubmissionStatus={setSubmissionStatus}
                  />
                </div>

                <div
                  className="validationIconContainer"
                  style={{
                    paddingRight: "20px",
                    width: "10%",
                    minWidth: "50px",
                    display: "flex",
                    alignItems: "center",
                  }}
                >
                  <ValidationIcon status={actionsValidationStatus} />
                </div>
              </div>
              <div className="amfToolConnectorWrapper">
                <Divider
                  orientation="vertical"
                  variant="middle"
                  className="amfToolDivider"
                  flexItem
                  sx={{
                    backgroundColor:
                      isFilePathValid && selectedAction
                        ? "#1976d2"
                        : "lightgrey",
                    marginTop: "16px",
                    marginBottom: "16px",
                    width: "2px",
                  }}
                />
              </div>
              <div
                className="submitButtonWrapper"
                style={{
                  width: "100%",
                  height: "40px",
                  minHeight: "40px",
                  display: "flex",
                  justifyContent: "center",
                  alignItems: "start",
                }}
              >
                <div
                  className="dummaryContainer"
                  style={{
                    width: "10%",
                    minWidth: "50px",
                    display: "flex",
                    alignItems: "center",
                  }}
                />
                <div
                  style={{
                    width: "80%",
                    display: "flex",
                    justifyContent: "center",
                  }}
                >
                  <PrimaryButton
                    className="amfToolSubmitButton"
                    disabled={
                      !isFilePathValid ||
                      !selectedAction ||
                      submissionStatus === "submitting"
                    }
                    onClick={handleSubmit}
                    sx={{ height: "40px" }}
                  >
                    Submit
                  </PrimaryButton>
                </div>

                <div
                  className="validationIconContainer"
                  style={{
                    paddingRight: "20px",
                    width: "10%",
                    minWidth: "50px",
                    display: "flex",
                    alignItems: "center",
                  }}
                >
                  <ValidationIcon status={submissionStatus} />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
