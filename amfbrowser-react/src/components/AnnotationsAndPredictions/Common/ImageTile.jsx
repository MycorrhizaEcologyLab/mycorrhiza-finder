import React, { useMemo, useCallback, memo } from "react";

const ImageTile = memo(
  ({
    tile,
    tileWidth,
    setSelectedTile,
    center,
    tileImages,
    showPredictions,
    validationMetricIdxMap,
    setCurrentValidationIdx,
    confidenceMetricIdxMap,
    setCurrentConfidenceIdx,
  }) => {
    const imageUrl = tileImages.get(`${tile.row}.${tile.col}`);

    const tileStyle = useMemo(
      () => ({
        width: `${tileWidth}px`,
        height: `${tileWidth}px`,
        outline: `${center ? "2px solid red" : "1px solid black"}`,
        margin: "2px",
        cursor: "pointer",
      }),
      [tileWidth, center],
    );

    const onClick = useCallback(() => {
      if (showPredictions) {
        // If we are viewing predictions, then also update the idx of the validation metric
        // when switching between tiles on click of the image grid
        const newValidationIdx = validationMetricIdxMap.get(
          `${tile?.row}.${tile?.col}`,
        );
        setCurrentValidationIdx(newValidationIdx);

        const newConfidenceIdx = confidenceMetricIdxMap.get(
          `${tile?.row}.${tile?.col}`,
        );
        setCurrentConfidenceIdx(newConfidenceIdx);
      }
      setSelectedTile({ row: tile.row, col: tile.col });
    }, [
      setSelectedTile,
      tile,
      validationMetricIdxMap,
      confidenceMetricIdxMap,
      showPredictions,
      setCurrentValidationIdx,
      setCurrentConfidenceIdx,
    ]);

    return imageUrl === "border" ? (
      <div style={tileStyle} className="border-image"></div>
    ) : (
      <img
        className="noselect"
        style={tileStyle}
        onClick={onClick}
        src={imageUrl}
        alt={`Tile ${tile.row}-${tile.col}`}
      />
    );
  },
);

export default ImageTile;
