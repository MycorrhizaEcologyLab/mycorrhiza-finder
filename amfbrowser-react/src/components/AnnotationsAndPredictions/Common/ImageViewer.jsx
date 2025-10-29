import React, { useEffect, useState, useMemo } from "react";
import TiledImageGrid from "./TiledImageGrid";

const ImageViewer = ({
  selectedTile,
  setSelectedTile,
  numRows,
  numCols,
  edge,
  numTiles,
  gridSize,
  tileImages,
  showPredictions,
  validationMetricIdxMap,
  setCurrentValidationIdx,
  confidenceMetricIdxMap,
  setCurrentConfidenceIdx,
}) => {
  const [tileWidth, setTileWidth] = useState(0);

  useEffect(() => {
    function calcTileWidth() {
      return Math.round((gridSize - numTiles * 5) / numTiles);
    }

    setTileWidth(calcTileWidth());
  }, [gridSize, numTiles]);

  const renderVisibleTiles = useMemo(() => {
    const visibleTiles = [];
    // Compute ranges for looping through image tiles (i.e. give an even spread around the center tile)
    const lowerBound = -Math.floor(numTiles / 2);
    const upperBound =
      numTiles % 2 === 0 ? numTiles / 2 : Math.floor(numTiles / 2) + 1;

    for (let x = lowerBound; x < upperBound; x++) {
      for (let y = lowerBound; y < upperBound; y++) {
        visibleTiles.push({
          row: (selectedTile.row + x + numRows + 1) % (numRows + 1),
          col: (selectedTile.col + y + numCols + 1) % (numCols + 1),
        });
      }
    }
    return visibleTiles;
  }, [selectedTile, numRows, numCols, numTiles]);

  return (
    <div
      style={{
        display: "flex",
        flexFlow: "column",
        justifyContent: "center",
        alignItems: "center",
      }}
    >
      <div>
        <TiledImageGrid
          tiles={renderVisibleTiles}
          edge={edge}
          numTiles={numTiles}
          setSelectedTile={setSelectedTile}
          tileWidth={tileWidth}
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
  );
};

export default ImageViewer;
