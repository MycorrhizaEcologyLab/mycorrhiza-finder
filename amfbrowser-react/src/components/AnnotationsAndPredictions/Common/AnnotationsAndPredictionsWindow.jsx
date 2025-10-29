import { useLocation } from "react-router-dom";
import AnnotationsLeftSidebar from "../Annotations/AnnotationsLeftSidebar";
import AnnotationsAndPredictionsCentralPane from "./AnnotationsAndPredictionsCentralPane";
import AnnotationsRightSidebar from "../Annotations/AnnotationsRightSidebar";
import PredictionsLeftSidebar from "../Predictions/PredictionsLeftSidebar";
import PredictionsRightSidebar from "../Predictions/PredictionsRightSidebar";

const AnnotationsAndPredictionsWindow = ({
  colorMappingCnn1,
  colorMappingCnn2,
  colorMappingCnn1Transparent,
  headerToValueMapCnn1,
  headerToValueMapCnn2,
  keyBindingsCnn1,
  keyBindingsCnn2,
  selectedTile,
  setTileValueWithIcon,
  cnn1Annotations,
  cnn2Annotations,
  cnn1Predictions,
  cnn2Predictions,
  iconSize,
  selectIcons,
  setSelectedOverlay,
  handleModeChange,
  onQuestionClick,
  onQuestionNext,
  selectFontSize,
  iconFontSize,
  headerToCountsMapCnn1,
  mode,
  level,
  setLevel,
  maxGridSize,
  setSelectedTile,
  numRows,
  numCols,
  tileEdge,
  gridSize,
  tileImages,
  validationMetricIdxMap,
  currentValidationIdx,
  setCurrentValidationIdx,
  confidenceMetricIdxMap,
  currentConfidenceIdx,
  setCurrentConfidenceIdx,
  handleSelectedConfidenceIdxChange,
  confidenceMetrics,
  handleMouseScroll,
  toCaptureRef,
  //AnnotationsViewer props
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
  // Action buttons additional props
  saveAnnotations,
  areAnnotationsSaving,
  captureScreenshot,
  setShowImageOverview,
  totalTiles,
  onConvert,
  handleSelectedValidationIdxChange,
  validationMetrics,
  conversionThreshold,
  convertWithContext,
  disableConvertWithContext,
  setConvertWithContext,
  contextualLabelSize,
  contextualLabelFontSize,
}) => {
  const { pathname } = useLocation();
  const showAnnotations = pathname.includes("annotations");
  const showPredictions = pathname.includes("predictions");

  return (
    <div
      id="browserMainWindow"
      className="fullWidth flexRowCenter"
      style={{ height: "calc(100% - 60px)" }}
    >
      {/* Left Sidebar */}
      {showAnnotations && (
        <AnnotationsLeftSidebar
          mode={mode}
          level={level}
          setLevel={setLevel}
          headerToCountsMapCnn1={headerToCountsMapCnn1}
          colorMapping={level === "CNN 1" ? colorMappingCnn1 : colorMappingCnn2}
          icons={
            level === "CNN 1" ? headerToValueMapCnn1 : headerToValueMapCnn2
          }
          handleModeChange={handleModeChange}
          keyBindings={level === "CNN 1" ? keyBindingsCnn1 : keyBindingsCnn2}
        />
      )}
      {showPredictions && (
        <PredictionsLeftSidebar
          level={level}
          setLevel={setLevel}
          colorMapping={level === "CNN 1" ? colorMappingCnn1 : colorMappingCnn2}
          icons={
            level === "CNN 1" ? headerToValueMapCnn1 : headerToValueMapCnn2
          }
          keyBindings={level === "CNN 1" ? keyBindingsCnn1 : keyBindingsCnn2}
        />
      )}

      <AnnotationsAndPredictionsCentralPane
        level={level}
        icons={level === "CNN 1" ? headerToValueMapCnn1 : headerToValueMapCnn2}
        selectedTile={selectedTile}
        colorMappingTransparent={colorMappingCnn1Transparent}
        colorMapping={level === "CNN 1" ? colorMappingCnn1 : colorMappingCnn2}
        setTileValueWithIcon={setTileValueWithIcon}
        cnn1Annotations={cnn1Annotations}
        cnn2Annotations={cnn2Annotations}
        cnn1Predictions={cnn1Predictions}
        cnn2Predictions={cnn2Predictions}
        iconSize={iconSize}
        selectIcons={selectIcons}
        setSelectedOverlay={setSelectedOverlay}
        handleModeChange={handleModeChange}
        onQuestionClick={onQuestionClick}
        onQuestionNext={onQuestionNext}
        selectFontSize={selectFontSize}
        iconFontSize={iconFontSize}
        showAnnotations={showAnnotations}
        maxGridSize={maxGridSize}
        setSelectedTile={setSelectedTile}
        numRows={numRows}
        numCols={numCols}
        tileEdge={tileEdge}
        gridSize={gridSize}
        tileImages={tileImages}
        showPredictions={showPredictions}
        validationMetricIdxMap={validationMetricIdxMap}
        setCurrentValidationIdx={setCurrentValidationIdx}
        confidenceMetricIdxMap={confidenceMetricIdxMap}
        setCurrentConfidenceIdx={setCurrentConfidenceIdx}
        handleMouseScroll={handleMouseScroll}
        toCaptureRef={toCaptureRef}
        // AnnotationsViewer additional props
        gridFontSize={gridFontSize}
        topLeftTile={topLeftTile}
        selectedOverlay={selectedOverlay}
        setDragActivated={setDragActivated}
        setFocusedTile={setFocusedTile}
        focusedTile={focusedTile}
        dragActivated={dragActivated}
        setQuestionCommentOpen={setQuestionCommentOpen}
        questionCommentOpen={questionCommentOpen}
        questionMarkComments={questionMarkComments}
        questionMarkCommentActions={questionMarkCommentActions}
        // Action buttons additional props
        saveAnnotations={saveAnnotations}
        areAnnotationsSaving={areAnnotationsSaving}
        captureScreenshot={captureScreenshot}
        setShowImageOverview={setShowImageOverview}
        conversionThreshold={conversionThreshold}
        convertWithContext={convertWithContext}
        contextualLabelSize={contextualLabelSize}
        contextualLabelFontSize={contextualLabelFontSize}
      />
      {showAnnotations && (
        <AnnotationsRightSidebar
          headerToCountsMapCnn1={headerToCountsMapCnn1}
          tileEdge={tileEdge}
          cnn1Annotations={cnn1Annotations}
          totalTiles={totalTiles}
        />
      )}
      {showPredictions && (
        <PredictionsRightSidebar
          onConvert={onConvert}
          currentValidationIdx={currentValidationIdx}
          handleSelectedValidationIdxChange={handleSelectedValidationIdxChange}
          predictionsLength={validationMetrics.length}
          setCurrentValidationIdx={setCurrentValidationIdx}
          validationMetrics={validationMetrics}
          setCurrentConfidenceIdx={setCurrentConfidenceIdx}
          confidenceMetrics={confidenceMetrics}
          currentConfidenceIdx={currentConfidenceIdx}
          handleSelectedConfidenceIdxChange={handleSelectedConfidenceIdxChange}
          setSelectedTile={setSelectedTile}
          tileEdge={tileEdge}
          validationMetricIdxMap={validationMetricIdxMap}
          confidenceMetricIdxMap={confidenceMetricIdxMap}
          conversionThreshold={conversionThreshold}
          convertWithContext={convertWithContext}
          disableConvertWithContext={disableConvertWithContext}
          setConvertWithContext={setConvertWithContext}
        />
      )}
    </div>
  );
};

export default AnnotationsAndPredictionsWindow;
