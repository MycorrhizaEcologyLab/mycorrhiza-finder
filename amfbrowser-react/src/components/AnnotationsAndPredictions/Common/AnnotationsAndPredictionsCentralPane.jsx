import Checkbox from "@mui/material/Checkbox";
import FormControlLabel from "@mui/material/FormControlLabel";
import Slider from "@mui/material/Slider";
import Typography from "@mui/material/Typography";
import { useState } from "react";

import NumericField from "../../Utils/NumericField";
import PrimaryButton from "../../Utils/PrimaryButton";
import AnnotationsViewer from "../Annotations/AnnotationsViewer";
import PredictionsViewer from "../Predictions/PredictionsViewer";
import AnnotationsAndPredictionsButtonsContainer from "./AnnotationsAndPredictionsButtonsContainer";
import ImageViewer from "./ImageViewer";

const AnnotationsAndPredictionsCentralPane = ({
  icons,
  selectedTile,
  colorMapping,
  colorMappingTransparent,
  setTileValueWithIcon,
  cnn1Annotations,
  cnn1Predictions,
  iconSize,
  selectIcons,
  setSelectedOverlay,
  handleModeChange,
  onQuestionClick,
  onQuestionNext,
  selectFontSize,
  iconFontSize,
  showAnnotations,
  maxGridSize,
  setSelectedTile,
  numRows,
  numCols,
  tileEdge,
  gridSize,
  tileImages,
  showPredictions,
  validationMetricIdxMap,
  setCurrentValidationIdx,
  confidenceMetricIdxMap,
  setCurrentConfidenceIdx,
  handleMouseScroll,
  toCaptureRef,
  // AnnotationsViewer
  gridFontSize,
  topLeftTile,
  selectedOverlay,
  setDragActivated,
  setFocusedTile,
  focusedTile,
  dragActivated,
  setQuestionCommentOpen,
  questionCommentOpen,
  questionMarkComments,
  questionMarkCommentActions,
  tileTags,
  allTags,
  tagPalette,
  tagBadgeSize,
  tagBadgeFontSize,
  // Annotation buttons additional props
  saveAnnotations,
  areAnnotationsSaving,
  captureScreenshot,
  setShowImageOverview,
  conversionThreshold,
  convertWithContext,
  contextualLabelSize,
  contextualLabelFontSize,
  autoClassifyBackground,
  onAutoClassifyBackgroundToggle,
  canUndoBackgroundFill,
  onToggleBackgroundFill,
  backgroundThreshold,
  onBackgroundThresholdChange,
  onBackgroundThresholdBlur,
}) => {
  const [numTiles, setNumTiles] = useState(3);
  const [wheelEvent, setWheelEvent] = useState(null);
  const [wheelFunction, setWheelFunction] = useState(null);
  const [wheelOpt, setWheelOpt] = useState(null);

  const handleSelectedRowChange = (value) => {
    if (value !== null && value !== "" && !isNaN(value)) {
      let parsedValue = parseInt(value);
      const newRow = (parsedValue + numRows) % numRows;
      const newCol = selectedTile.col;

      if (showPredictions) {
        // If we are viewing predictions, then also update the idx of the validation metric
        // when switching between tiles on click of the image grid
        const newValidationIdx = validationMetricIdxMap.get(
          `${newRow}.${newCol}`,
        );
        setCurrentValidationIdx(newValidationIdx);

        const newConfidenceIdx = confidenceMetricIdxMap.get(
          `${newRow}.${newCol}`,
        );
        setCurrentConfidenceIdx(newConfidenceIdx);
      }

      setSelectedTile({
        row: newRow,
        col: newCol,
      });
    }
  };

  const handleSelectedColChange = (value) => {
    if (value !== null && value !== "" && !isNaN(value)) {
      let parsedValue = parseInt(value);
      const newRow = selectedTile.row;
      const newCol = (parsedValue + numCols) % numCols;

      if (showPredictions) {
        // If we are viewing predictions, then also update the idx of the validation metric
        // when switching between tiles on click of the image grid
        const newValidationIdx = validationMetricIdxMap.get(
          `${newRow}.${newCol}`,
        );
        setCurrentValidationIdx(newValidationIdx);

        const newConfidenceIdx = confidenceMetricIdxMap.get(
          `${newRow}.${newCol}`,
        );
        setCurrentConfidenceIdx(newConfidenceIdx);
      }

      setSelectedTile({
        row: newRow,
        col: newCol,
      });
    }
  };

  function preventDefault(e) {
    e?.preventDefault();
  }

  function toggleScroll(toggle) {
    var supportsPassive = false;
    try {
      window.addEventListener(
        "test",
        null,
        Object.defineProperty({}, "passive", {
          get: function () {
            supportsPassive = true;
            return 0;
          },
        }),
      );
    } catch (e) {}

    if (toggle) {
      var wheelOptLocal = supportsPassive ? { passive: false } : false;
      var wheelEventLocal =
        "onwheel" in document.createElement("div") ? "wheel" : "mousewheel";
      var preventDefaultLocal = preventDefault;

      // Store these in state so they are identical when removing event listener
      setWheelEvent(wheelEventLocal);
      setWheelFunction(() => preventDefaultLocal);
      setWheelOpt(wheelOptLocal);
      window.addEventListener(
        wheelEventLocal,
        preventDefaultLocal,
        wheelOptLocal,
      );
    } else {
      window.removeEventListener(wheelEvent, wheelFunction, wheelOpt);
    }
  }

  return (
    <div
      id="AnnotationsAndPredictionsCentralPane"
      className="fullHeight flexColumn"
      style={{
        flex: 1,
        width: "auto",
        overflowY: "auto",
      }}
    >
      <div
        id="minSizeWrapper"
        style={{
          // Dynamic gridSize calculation: 2 * gridSize + 2 * 50px (side margins of the two columns)
          minWidth: 2 * gridSize + 50,
          // Dynamic gridSize calculation: 110px (icons) + 75px (zoomBox) + gridSize + 20px (grid margin) + 110px (buttons, 2 rows)
          minHeight: 110 + 75 + gridSize + 20 + 110,
        }}
      >
        <div
          className="selectionButtons"
          style={{
            height: "110px",
            width: "100%",
          }}
        >
          <AnnotationsAndPredictionsButtonsContainer
            icons={icons}
            selectedTile={selectedTile}
            colorMapping={colorMapping}
            onClick={setTileValueWithIcon}
            gridData={cnn1Annotations}
            size={iconSize}
            selectIcons={selectIcons}
            setSelectedOverlay={setSelectedOverlay}
            handleModeChange={handleModeChange}
            onQuestionClick={onQuestionClick}
            onQuestionNext={onQuestionNext}
            selectFontSize={selectFontSize}
            iconFontSize={iconFontSize}
            showAnnotations={showAnnotations}
          />
        </div>
        <div
          className="fullWidth flexRow"
          ref={toCaptureRef}
          style={{
            height: 75 + 20 + gridSize, // Dynamic gridSize calculation: 75px (zoomBox height) + 20px (margins) + gridSize
          }}
        >
          <div
            className="column1 fullHeight flexColumn"
            style={{
              width: "50%",
              alignItems: "end",
              margin: "0px 25px",
            }}
          >
            <div
              className="zoomBox flexColumnCenter"
              style={{
                width: gridSize, // Dynamic gridSize calculation: gridSize
                height: "75px",
              }}
            >
              <Typography
                id="track-inverted-slider gutterBottom"
                sx={{
                  fontSize: "14px",
                  fontWeight: "500",
                  fontFamily: "Franklin Gothic",
                }}
              >
                Tiles: {numTiles}
              </Typography>

              <Slider
                aria-label="Zoom"
                valueLabelDisplay="auto"
                marks
                min={1}
                max={Number(maxGridSize)}
                onChange={(event, value) =>
                  event.type === "mousemove" && setNumTiles(value)
                }
                value={numTiles}
                sx={{ width: "100%", margin: "0 5px" }}
              />
            </div>
            <div
              className="imageViewer fullWidth flexRow"
              style={{
                height: gridSize + 20, // Dynamic gridSize calculation: gridSize + 20px (margins)
                alignItems: "center",
                justifyContent: "end",
                overflowY: "clip",
              }}
            >
              <div
                className="image-components-container"
                onWheel={handleMouseScroll}
                onMouseEnter={() => toggleScroll(true)}
                onMouseLeave={() => toggleScroll(false)}
              >
                <ImageViewer
                  key="image-viewer"
                  selectedTile={selectedTile}
                  setSelectedTile={setSelectedTile}
                  numRows={numRows}
                  numCols={numCols}
                  edge={tileEdge}
                  numTiles={numTiles}
                  gridSize={gridSize}
                  tileImages={tileImages}
                  showPredictions={showPredictions}
                  validationMetricIdxMap={validationMetricIdxMap}
                  setCurrentValidationIdx={setCurrentValidationIdx}
                  confidenceMetricIdxMap={confidenceMetricIdxMap}
                  setCurrentConfidenceIdx={setCurrentConfidenceIdx}
                />
              </div>
            </div>
          </div>
          <div
            className="column2 fullHeight flexColumn"
            style={{
              width: "50%",
              alignItems: "start",
              margin: "0px 25px",
            }}
          >
            <div
              className="cellSelection"
              style={{
                width: gridSize, // Dynamic gridSize calculation: gridSize
                height: "75px",
                display: "flex",
                justifyContent: "center",
              }}
            >
              <div
                style={{
                  display: "flex",
                }}
              >
                <div
                  style={{
                    marginRight: "10px",
                    display: "flex",
                    alignItems: "center",
                  }}
                >
                  <NumericField
                    label="Row"
                    value={selectedTile.row}
                    onChange={handleSelectedRowChange}
                  />
                </div>
                <div
                  style={{
                    marginLeft: "10px",
                    display: "flex",
                    alignItems: "center",
                  }}
                >
                  <NumericField
                    label="Col"
                    value={selectedTile.col}
                    onChange={handleSelectedColChange}
                  />
                </div>
              </div>
            </div>
            <div
              className="annotationsViewer fullWidth flexRow"
              style={{
                height: gridSize + 20, // Dynamic gridSize calculation: gridSize + 20px (top and bottom margins)
                display: "flex",
                alignItems: "center",
                justifyContent: "start",
              }}
            >
              <div
                className="image-components-container"
                onWheel={handleMouseScroll}
                onMouseEnter={() => toggleScroll(true)}
                onMouseLeave={() => toggleScroll(false)}
              >
                {showAnnotations && (
                  <AnnotationsViewer
                    key="annotations-viewer"
                    selectedTile={selectedTile}
                    gridDataCnn1={cnn1Annotations}
                    numRows={numRows}
                    numCols={numCols}
                    setSelectedTile={setSelectedTile}
                    colorMapping={colorMappingTransparent}
                    gridSize={gridSize}
                    gridFontSize={gridFontSize}
                    topLeftTile={topLeftTile}
                    selectedOverlay={selectedOverlay}
                    setDragActivated={setDragActivated}
                    setFocusedTile={setFocusedTile}
                    focusedTile={focusedTile}
                    dragActivated={dragActivated}
                    tileImages={tileImages}
                    setQuestionCommentOpen={setQuestionCommentOpen}
                    questionCommentOpen={questionCommentOpen}
                    questionMarkComments={questionMarkComments}
                    questionMarkCommentActions={questionMarkCommentActions}
                    tileTags={tileTags}
                    allTags={allTags}
                    tagPalette={tagPalette}
                    tagBadgeSize={tagBadgeSize}
                    tagBadgeFontSize={tagBadgeFontSize}
                  />
                )}
                {showPredictions && (
                  <PredictionsViewer
                    key="predictions-viewer"
                    selectedTile={selectedTile}
                    gridData={cnn1Predictions}
                    numRows={numRows}
                    numCols={numCols}
                    setSelectedTile={setSelectedTile}
                    gridSize={gridSize}
                    topLeftTile={topLeftTile}
                    colorMapping={colorMapping}
                    gridFontSize={gridFontSize}
                    validationMetricIdxMap={validationMetricIdxMap}
                    setCurrentValidationIdx={setCurrentValidationIdx}
                    confidenceMetricIdxMap={confidenceMetricIdxMap}
                    setCurrentConfidenceIdx={setCurrentConfidenceIdx}
                    conversionThreshold={conversionThreshold}
                    convertWithContext={convertWithContext}
                    contextualLabelSize={contextualLabelSize}
                    contextualLabelFontSize={contextualLabelFontSize}
                  />
                )}
              </div>
            </div>
          </div>
        </div>
        <div
          className="actionButtons"
          style={{
            // No maxHeight: the buttons below wrap onto a second row on
            // narrow panes, and capping the height made the overflow paint
            // over the neighbouring content instead of pushing it down.
            minHeight: "110px",
            width: "100%",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            flexShrink: 0,
            gap: "5px",
            paddingBottom: "10px",
          }}
        >
          <div className="buttonRow">
            {showAnnotations && (
              <PrimaryButton
                disabled={areAnnotationsSaving}
                sx={{ minWidth: "200px" }}
                onClick={() => saveAnnotations(false)}
              >
                Save Annotations
              </PrimaryButton>
            )}

            {showAnnotations && (
              <PrimaryButton
                sx={{ minWidth: "200px" }}
                onClick={onToggleBackgroundFill}
                title={
                  canUndoBackgroundFill
                    ? "Undo the last fill of remaining tiles as Background. Hotkey: Shift+*"
                    : "Classifies all remaining/unlabelled tiles as Background. Hotkey: Shift+*"
                }
              >
                {canUndoBackgroundFill ? "Undo X Fill" : "Fill X"}
              </PrimaryButton>
            )}

            <PrimaryButton
              sx={{ minWidth: "200px" }}
              onClick={captureScreenshot}
            >
              Take screenshot
            </PrimaryButton>
            <PrimaryButton
              sx={{ minWidth: "200px" }}
              onClick={() => setShowImageOverview(true)}
            >
              Show image overview
            </PrimaryButton>
          </div>

          {showAnnotations && (
            <div className="buttonRow">
              <FormControlLabel
                sx={{ margin: 0 }}
                control={
                  <Checkbox
                    checked={autoClassifyBackground}
                    onChange={onAutoClassifyBackgroundToggle}
                  />
                }
                label="Auto-classify background tiles"
                title="Automatically labels tiles detected as background (via mean pixel intensity, same heuristic as the Filter Background Tiles training option) as Background. Unchecking removes only the tiles it added. Adjust the threshold alongside to control sensitivity."
              />
              <div>
                <NumericField
                  label="Threshold"
                  value={backgroundThreshold}
                  onChange={onBackgroundThresholdChange}
                  onBlur={onBackgroundThresholdBlur}
                  min={0}
                  max={1}
                  step={0.01}
                />
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default AnnotationsAndPredictionsCentralPane;
