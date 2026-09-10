import ModelTrainingIcon from "@mui/icons-material/ModelTraining";
import OnlinePredictionIcon from "@mui/icons-material/OnlinePrediction";
import TransformIcon from "@mui/icons-material/Transform";
import BugReportIcon from "@mui/icons-material/BugReport";
import PercentIcon from "@mui/icons-material/Percent";
import TroubleshootIcon from "@mui/icons-material/Troubleshoot";
import PrecisionManufacturingIcon from "@mui/icons-material/PrecisionManufacturing";
import AmfToolActionButton from "./AmfToolActionButton";

const AmfToolActionButtons = ({
  selectedAction,
  setSelectedAction,
  isFilePathValid,
  submissionStatus,
  setSubmissionStatus,
}) => {
  const isActionInProgress = submissionStatus === "submitting";

  const handleActionSelected = (action) => {
    submissionStatus !== "none" && setSubmissionStatus("none");
    setSelectedAction(action);
  };
  return (
    <div
      className="amfinder-buttons-container"
      // No minWidth: the seven squares need ~1015px in a single row, so
      // forcing one would prevent them wrapping on a narrower viewport.
      style={{ maxWidth: "1300px" }}
    >
      <AmfToolActionButton
        isSelected={selectedAction === "predict"}
        disabled={!isFilePathValid || isActionInProgress}
        onClick={() => handleActionSelected("predict")}
      >
        <OnlinePredictionIcon sx={{ height: "70px", width: "70px" }} />
        <div style={{ paddingTop: "5px" }}>Predict</div>
      </AmfToolActionButton>
      <AmfToolActionButton
        isSelected={selectedAction === "convert"}
        disabled={!isFilePathValid || isActionInProgress}
        onClick={() => handleActionSelected("convert")}
      >
        <TransformIcon sx={{ height: "70px", width: "70px" }} />
        <div style={{ paddingTop: "5px" }}>Convert</div>
      </AmfToolActionButton>
      <AmfToolActionButton
        isSelected={selectedAction === "tifconversion"}
        disabled={!isFilePathValid || isActionInProgress}
        onClick={() => handleActionSelected("tifconversion")}
      >
        <PrecisionManufacturingIcon sx={{ height: "70px", width: "70px" }} />
        <div style={{ paddingTop: "5px" }}>TIF Convert</div>
      </AmfToolActionButton>
      <AmfToolActionButton
        isSelected={selectedAction === "colonisation"}
        disabled={!isFilePathValid || isActionInProgress}
        onClick={() => handleActionSelected("colonisation")}
      >
        <PercentIcon sx={{ height: "70px", width: "70px" }} />
        <div style={{ paddingTop: "5px" }}>Colonisation</div>
      </AmfToolActionButton>
      <AmfToolActionButton
        isSelected={selectedAction === "train"}
        disabled={!isFilePathValid || isActionInProgress}
        onClick={() => handleActionSelected("train")}
      >
        <ModelTrainingIcon sx={{ height: "70px", width: "70px" }} />
        <div style={{ paddingTop: "5px" }}>Train</div>
      </AmfToolActionButton>
      <AmfToolActionButton
        isSelected={selectedAction === "test"}
        disabled={!isFilePathValid || isActionInProgress}
        onClick={() => handleActionSelected("test")}
      >
        <BugReportIcon sx={{ height: "70px", width: "70px" }} />
        <div style={{ paddingTop: "5px" }}>Test</div>
      </AmfToolActionButton>
      <AmfToolActionButton
        isSelected={selectedAction === "calibrate"}
        disabled={!isFilePathValid || isActionInProgress}
        onClick={() => handleActionSelected("calibrate")}
      >
        <TroubleshootIcon sx={{ height: "70px", width: "70px" }} />
        <div style={{ paddingTop: "5px" }}>Calibrate</div>
      </AmfToolActionButton>
    </div>
  );
};

export default AmfToolActionButtons;
