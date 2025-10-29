import React, { useMemo } from "react";
import TiledPredictionsGrid from "./TiledPredictionsGrid";

const PredictionsViewer = ({
  selectedTile,
  gridData,
  numRows,
  numCols,
  setSelectedTile,
  gridSize,
  topLeftTile,
  colorMapping,
  gridFontSize,
  validationMetricIdxMap,
  setCurrentValidationIdx,
  confidenceMetricIdxMap,
  setCurrentConfidenceIdx,
  conversionThreshold,
  convertWithContext,
  contextualLabelSize,
  contextualLabelFontSize,
}) => {
  const renderVisibleTiles = useMemo(() => {
    const visibleTiles = [];
    for (let x = 0; x < 3; x++) {
      for (let y = 0; y < 3; y++) {
        let currentRow = (topLeftTile.row + x + numRows) % numRows;
        let currentCol = (topLeftTile.col + y + numCols) % numCols;
        let values = gridData.get(
          currentRow.toString() + "." + currentCol.toString(),
        );
        visibleTiles.push({
          predictions: values?.predictions,
          annotations: values?.annotations,
          contextualLabel: values?.contextualLabel,
          row: currentRow,
          col: currentCol,
        });
      }
    }
    return visibleTiles;
  }, [gridData, numCols, numRows, topLeftTile]);

  return (
    <div
      style={{
        display: "flex",
        flexFlow: "column",
        justifyContent: "center",
        alignItems: "flex-end",
        marginLeft: "10px",
      }}
    >
      <TiledPredictionsGrid
        tiles={renderVisibleTiles}
        selectedTile={selectedTile}
        setSelectedTile={setSelectedTile}
        gridSize={gridSize}
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
    </div>
  );
};

export default PredictionsViewer;
