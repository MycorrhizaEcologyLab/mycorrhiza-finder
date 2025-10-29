import React, { useMemo } from "react";
import ImageTile from "./ImageTile";

const TiledImageGrid = ({
  tiles,
  numTiles,
  setSelectedTile,
  tileWidth,
  gridSize,
  tileImages,
  showPredictions,
  validationMetricIdxMap,
  setCurrentValidationIdx,
  confidenceMetricIdxMap,
  setCurrentConfidenceIdx,
}) => {
  // Find center tile of grid
  const centerNumber = useMemo(
    () =>
      numTiles % 2 === 0
        ? (numTiles ** 2 + numTiles) / 2
        : Math.floor(numTiles ** 2 / 2),
    [numTiles],
  );

  return (
    <div
      className="tiled-image-grid"
      style={{
        display: "flex",
        flexFlow: "row wrap",
        justifyContent: "center",
        alignItems: "center",
        width: `${gridSize}px`,
        height: `${gridSize}px`,
      }}
    >
      {tiles.map((tile, index) => {
        let center = index === centerNumber;
        return (
          <ImageTile
            key={index}
            tile={tile}
            center={center}
            tileWidth={tileWidth}
            setSelectedTile={setSelectedTile}
            tileImages={tileImages}
            showPredictions={showPredictions}
            validationMetricIdxMap={validationMetricIdxMap}
            setCurrentValidationIdx={setCurrentValidationIdx}
            confidenceMetricIdxMap={confidenceMetricIdxMap}
            setCurrentConfidenceIdx={setCurrentConfidenceIdx}
          />
        );
      })}
    </div>
  );
};

export default TiledImageGrid;
