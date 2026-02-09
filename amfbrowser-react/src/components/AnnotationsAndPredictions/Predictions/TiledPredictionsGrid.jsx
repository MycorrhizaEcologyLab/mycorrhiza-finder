import { useEffect, useState } from "react";
import PredictionsTile from "./PredictionsTile";

const TiledPredictionsGrid = ({
  tiles,
  selectedTile,
  setSelectedTile,
  gridSize,
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
  const [tileWidth, setTileWidth] = useState(0);

  useEffect(() => {
    function calcTileWidth() {
      // Subtract 4 for margins
      return Math.floor(gridSize / 3) - 4;
    }
    setTileWidth(calcTileWidth());
  }, [gridSize]);

  return (
    <div
      style={{
        display: "flex",
        flexFlow: "row wrap",
        width: `${gridSize}px`,
        height: `${gridSize}px`,
      }}
    >
      {tiles.map((tile, index) => {
        let selected =
          tile.row === selectedTile.row && tile.col === selectedTile.col;
        return (
          <PredictionsTile
            key={index}
            selected={selected}
            tile={tile}
            setSelectedTile={setSelectedTile}
            tileWidth={tileWidth}
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
        );
      })}
    </div>
  );
};

export default TiledPredictionsGrid;
