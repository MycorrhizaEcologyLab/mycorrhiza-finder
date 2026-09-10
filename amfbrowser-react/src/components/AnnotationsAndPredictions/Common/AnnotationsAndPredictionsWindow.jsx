import { useLocation } from "react-router-dom";
import AnnotationsLeftSidebar from "../Annotations/AnnotationsLeftSidebar";
import AnnotationsRightSidebar from "../Annotations/AnnotationsRightSidebar";
import PredictionsLeftSidebar from "../Predictions/PredictionsLeftSidebar";
import PredictionsRightSidebar from "../Predictions/PredictionsRightSidebar";
import AnnotationsAndPredictionsCentralPane from "./AnnotationsAndPredictionsCentralPane";

const AnnotationsAndPredictionsWindow = ({
  colorMappingCnn1,
  colorMappingCnn1Transparent,
  headerToValueMapCnn1,
  keyBindingsCnn1,
  selectedTile,
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
  headerToCountsMapCnn1,
  mode,
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
  // Sub-tags
  tileTags,
  allTags,
  tagPalette,
  selectedTileTags,
  hasSelectedClass,
  onCreateTag,
  onDeleteTag,
  onToggleTag,
  onClearTags,
  setTagInputOpen,
  tagBadgeSize,
  tagBadgeFontSize,
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
  autoClassifyBackground,
  onAutoClassifyBackgroundToggle,
  canUndoBackgroundFill,
  onToggleBackgroundFill,
  backgroundThreshold,
  onBackgroundThresholdChange,
  onBackgroundThresholdBlur,
}) => {
  const { pathname } = useLocation();
  const showAnnotations = pathname.includes("annotations");
  const showPredictions = pathname.includes("predictions");

  return (
    <div
      id="browserMainWindow"
      className="fullWidth flexRowCenter"
      // Fills whatever the header leaves rather than assuming it is exactly
      // 60px (it grows when the Image Directory banner shows).
      style={{ flex: 1, minHeight: 0 }}
    >
      {/* Left Sidebar */}
      {showAnnotations && (
        <AnnotationsLeftSidebar
          mode={mode}
          headerToCountsMapCnn1={headerToCountsMapCnn1}
          colorMapping={colorMappingCnn1}
          icons={headerToValueMapCnn1}
          handleModeChange={handleModeChange}
          keyBindings={keyBindingsCnn1}
          allTags={allTags}
          tagPalette={tagPalette}
          selectedTileTags={selectedTileTags}
          hasSelectedClass={hasSelectedClass}
          onCreateTag={onCreateTag}
          onDeleteTag={onDeleteTag}
          onToggleTag={onToggleTag}
          onClearTags={onClearTags}
          setTagInputOpen={setTagInputOpen}
        />
      )}
      {showPredictions && (
        <PredictionsLeftSidebar
          colorMapping={colorMappingCnn1}
          icons={headerToValueMapCnn1}
          keyBindings={keyBindingsCnn1}
        />
      )}

      <AnnotationsAndPredictionsCentralPane
        icons={headerToValueMapCnn1}
        selectedTile={selectedTile}
        colorMappingTransparent={colorMappingCnn1Transparent}
        colorMapping={colorMappingCnn1}
        setTileValueWithIcon={setTileValueWithIcon}
        cnn1Annotations={cnn1Annotations}
        cnn1Predictions={cnn1Predictions}
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
        tileTags={tileTags}
        allTags={allTags}
        tagPalette={tagPalette}
        tagBadgeSize={tagBadgeSize}
        tagBadgeFontSize={tagBadgeFontSize}
        // Action buttons additional props
        saveAnnotations={saveAnnotations}
        areAnnotationsSaving={areAnnotationsSaving}
        captureScreenshot={captureScreenshot}
        setShowImageOverview={setShowImageOverview}
        conversionThreshold={conversionThreshold}
        convertWithContext={convertWithContext}
        contextualLabelSize={contextualLabelSize}
        contextualLabelFontSize={contextualLabelFontSize}
        autoClassifyBackground={autoClassifyBackground}
        onAutoClassifyBackgroundToggle={onAutoClassifyBackgroundToggle}
        canUndoBackgroundFill={canUndoBackgroundFill}
        onToggleBackgroundFill={onToggleBackgroundFill}
        backgroundThreshold={backgroundThreshold}
        onBackgroundThresholdChange={onBackgroundThresholdChange}
        onBackgroundThresholdBlur={onBackgroundThresholdBlur}
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
