import FormControlLabel from "@mui/material/FormControlLabel";
import { styled } from "@mui/material/styles";
import Switch from "@mui/material/Switch";
import { useState } from "react";
import { useNavigate } from "react-router-dom";

import NumericField from "../../Utils/NumericField";
import PrimaryButton from "../../Utils/PrimaryButton";

const DrawerHeader = styled("div")(({ theme }) => ({
  display: "flex",
  flexDirection: "column",
  height: "65px",
  width: "100%",
  alignItems: "center",
  justifyContent: "center",
  padding: theme.spacing(0, 1),
}));

const SidebarDivider = () => (
  <div
    className="annotationsSidebarDivider fullWidth"
    style={{
      borderBottomStyle: "solid",
      borderBottomColor: "lightgray",
      borderBottomWidth: "2px",
    }}
  />
);

const PredictionsRightSidebar = ({
  onConvert,
  currentValidationIdx,
  validationMetricIdxMap,
  handleSelectedValidationIdxChange,
  predictionsLength,
  setCurrentValidationIdx,
  validationMetrics,
  currentConfidenceIdx,
  handleSelectedConfidenceIdxChange,
  setCurrentConfidenceIdx,
  confidenceMetrics,
  confidenceMetricIdxMap,
  setSelectedTile,
  tileEdge,
  conversionThreshold,
  convertWithContext,
  disableConvertWithContext,
  setConvertWithContext,
}) => {
  const navigate = useNavigate();

  const [showValidate, setShowValidate] = useState(false);

  const settingsRoute = () => {
    // TODO get scroll to ID to work
    navigate("/settings#convert");
  };

  const onValidatePrev = () => {
    if (currentValidationIdx === undefined) {
      // If idx is undefined then set to 0th element
      let splitKey = validationMetrics[0][0].split(".");
      const newRow = parseInt(splitKey[0]);
      const newCol = parseInt(splitKey[1]);
      setSelectedTile({
        row: newRow,
        col: newCol,
      });
      setCurrentValidationIdx(0);

      const newConfidenceIdx = confidenceMetricIdxMap.get(
        `${newRow}.${newCol}`,
      );
      setCurrentConfidenceIdx(newConfidenceIdx);
    } else {
      // Either reduce idx by 1 or set to last elt in list
      let newIndex =
        currentValidationIdx === 0
          ? validationMetrics.length - 1
          : currentValidationIdx - 1;
      let splitKey = validationMetrics[newIndex][0].split(".");
      const newRow = parseInt(splitKey[0]);
      const newCol = parseInt(splitKey[1]);
      setSelectedTile({
        row: newRow,
        col: newCol,
      });
      setCurrentValidationIdx(newIndex);

      const newConfidenceIdx = confidenceMetricIdxMap.get(
        `${newRow}.${newCol}`,
      );
      setCurrentConfidenceIdx(newConfidenceIdx);
    }
  };

  const onValidateNext = () => {
    if (currentValidationIdx === undefined) {
      // If idx is undefined then set to 0th element
      let splitKey = validationMetrics[0][0].split(".");
      const newRow = parseInt(splitKey[0]);
      const newCol = parseInt(splitKey[1]);
      setSelectedTile({
        row: newRow,
        col: newCol,
      });
      setCurrentValidationIdx(0);

      const newConfidenceIdx = confidenceMetricIdxMap.get(
        `${newRow}.${newCol}`,
      );
      setCurrentConfidenceIdx(newConfidenceIdx);
    } else {
      // Either increase idx by 1 or set to first elt in list
      let newIndex =
        currentValidationIdx === validationMetrics.length - 1
          ? 0
          : currentValidationIdx + 1;
      let splitKey = validationMetrics[newIndex][0].split(".");
      const newRow = parseInt(splitKey[0]);
      const newCol = parseInt(splitKey[1]);
      setSelectedTile({
        row: newRow,
        col: newCol,
      });
      setCurrentValidationIdx(newIndex);

      const newConfidenceIdx = confidenceMetricIdxMap.get(
        `${newRow}.${newCol}`,
      );
      setCurrentConfidenceIdx(newConfidenceIdx);
    }
  };

  const onConfidencePrev = () => {
    if (currentConfidenceIdx === undefined) {
      // If idx is undefined then set to 0th element
      let splitKey = confidenceMetrics[0][0].split(".");
      const newRow = parseInt(splitKey[0]);
      const newCol = parseInt(splitKey[1]);
      setSelectedTile({
        row: newRow,
        col: newCol,
      });
      setCurrentConfidenceIdx(0);

      const newValidationIdx = validationMetricIdxMap.get(
        `${newRow}.${newCol}`,
      );
      setCurrentValidationIdx(newValidationIdx);
    } else {
      // Either reduce idx by 1 or set to last elt in list
      let newIndex =
        currentConfidenceIdx === 0
          ? confidenceMetrics.length - 1
          : currentConfidenceIdx - 1;
      let splitKey = confidenceMetrics[newIndex][0].split(".");
      const newRow = parseInt(splitKey[0]);
      const newCol = parseInt(splitKey[1]);
      setSelectedTile({
        row: newRow,
        col: newCol,
      });
      setCurrentConfidenceIdx(newIndex);

      const newValidationIdx = validationMetricIdxMap.get(
        `${newRow}.${newCol}`,
      );
      setCurrentValidationIdx(newValidationIdx);
    }
  };

  const onConfidenceNext = () => {
    if (currentConfidenceIdx === undefined) {
      // If idx is undefined then set to 0th element
      let splitKey = confidenceMetrics[0][0].split(".");
      const newRow = parseInt(splitKey[0]);
      const newCol = parseInt(splitKey[1]);
      setSelectedTile({
        row: newRow,
        col: newCol,
      });
      setCurrentConfidenceIdx(0);

      const newValidationIdx = validationMetricIdxMap.get(
        `${newRow}.${newCol}`,
      );
      setCurrentValidationIdx(newValidationIdx);
    } else {
      // Either increase idx by 1 or set to first elt in list
      let newIndex =
        currentConfidenceIdx === confidenceMetrics.length - 1
          ? 0
          : currentConfidenceIdx + 1;
      let splitKey = confidenceMetrics[newIndex][0].split(".");
      const newRow = parseInt(splitKey[0]);
      const newCol = parseInt(splitKey[1]);
      setSelectedTile({
        row: newRow,
        col: newCol,
      });
      setCurrentConfidenceIdx(newIndex);

      const newValidationIdx = validationMetricIdxMap.get(
        `${newRow}.${newCol}`,
      );
      setCurrentValidationIdx(newValidationIdx);
    }
  };

  return (
    <div
      id="AnnotationsRightSidebar"
      className="sidebarColumn"
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100%",
        width: "150px",
        overflowY: "auto",
        borderLeft: "2px solid lightgray",
      }}
    >
      <DrawerHeader>
        <div style={{ fontSize: "12px", color: "rgb(120, 117, 101)" }}>
          Actions
        </div>
      </DrawerHeader>

      {/* Tile size */}
      <SidebarDivider />
      <div
        style={{
          width: "100%",
          height: "150px",
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          alignItems: "center",
        }}
      >
        <div style={{ fontSize: "12px", fontWeight: 600, margin: "5px 0px" }}>
          Tile size
        </div>
        <div style={{ fontSize: "12px", margin: "5px 0px" }}>{tileEdge}</div>
      </div>
      <SidebarDivider />

      {/* Show validate */}
      <div
        style={{
          width: "100%",
          height: "75px",
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          alignItems: "center",
        }}
      >
        <div style={{ fontSize: "12px", fontWeight: 600, margin: "5px 0px" }}>
          Show Validate
        </div>
        <Switch
          checked={showValidate}
          onChange={(e) => setShowValidate(e.target.checked)}
        />
      </div>
      <SidebarDivider />

      {/* Validate */}
      {showValidate && (
        <div>
          <div
            style={{
              width: "100%",
              height: "150px",
              display: "flex",
              flexDirection: "column",
              justifyContent: "center",
              alignItems: "center",
            }}
          >
            <div
              style={{ fontSize: "12px", fontWeight: 600, margin: "5px 0px" }}
            >
              Validate (std dev)
            </div>
            <div style={{ fontSize: "12px", margin: "5px 0px" }}>
              <NumericField
                value={
                  currentValidationIdx !== undefined ? currentValidationIdx : ""
                }
                onChange={handleSelectedValidationIdxChange}
                max={predictionsLength - 1}
              />
            </div>
            <div style={{ display: "flex" }}>
              <PrimaryButton
                onClick={onValidatePrev}
                sx={{
                  height: "30px",
                  minWidth: "40px",
                  margin: "5px",
                  fontSize: "12px",
                }}
              >
                Prev
              </PrimaryButton>
              <PrimaryButton
                onClick={onValidateNext}
                sx={{
                  height: "30px",
                  minWidth: "40px",
                  margin: "5px",
                  fontSize: "12px",
                }}
              >
                Next
              </PrimaryButton>
            </div>
          </div>

          <SidebarDivider />
        </div>
      )}

      {/* Confidence */}
      <div
        style={{
          width: "100%",
          height: "150px",
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          alignItems: "center",
        }}
      >
        <div style={{ fontSize: "12px", fontWeight: 600, margin: "5px 0px" }}>
          Check Confidence
        </div>
        <div style={{ fontSize: "12px", margin: "5px 0px" }}>
          <NumericField
            value={
              currentConfidenceIdx !== undefined ? currentConfidenceIdx : ""
            }
            onChange={handleSelectedConfidenceIdxChange}
            max={predictionsLength - 1}
          />
        </div>
        <div style={{ display: "flex" }}>
          <PrimaryButton
            onClick={onConfidencePrev}
            sx={{
              height: "30px",
              minWidth: "40px",
              margin: "5px",
              fontSize: "12px",
            }}
          >
            Prev
          </PrimaryButton>
          <PrimaryButton
            onClick={onConfidenceNext}
            sx={{
              height: "30px",
              minWidth: "40px",
              margin: "5px",
              fontSize: "12px",
            }}
          >
            Next
          </PrimaryButton>
        </div>
      </div>

      <SidebarDivider />

      {/* Threshold */}
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          alignItems: "center",
          width: "100%",
          height: "150px",
        }}
      >
        <div style={{ fontSize: "12px", fontWeight: 600, margin: "5px 0px" }}>
          Threshold
        </div>
        <div style={{ fontSize: "16px", fontWeight: 600, margin: "5px 0px" }}>
          {conversionThreshold}
        </div>
        <PrimaryButton
          onClick={() => settingsRoute()}
          sx={{ height: "30px", fontSize: "12px", margin: "5px 0px" }}
        >
          Settings
        </PrimaryButton>
      </div>
      <SidebarDivider />

      {/* Convert */}
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          alignItems: "center",
          width: "100%",
          height: "150px",
        }}
      >
        <PrimaryButton
          onClick={() => onConvert()}
          sx={{ height: "30px", fontSize: "12px" }}
        >
          Convert
        </PrimaryButton>
        <FormControlLabel
          control={
            <Switch
              disabled={disableConvertWithContext}
              checked={convertWithContext}
              onChange={(e) => setConvertWithContext(e.target.checked)}
            />
          }
          label={<span style={{ fontSize: "12px" }}>Convert with Context</span>}
          labelPlacement="bottom"
        />
      </div>
      <SidebarDivider />
    </div>
  );
};

export default PredictionsRightSidebar;
