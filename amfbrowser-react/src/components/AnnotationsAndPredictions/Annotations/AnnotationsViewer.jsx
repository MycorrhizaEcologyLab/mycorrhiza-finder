import React, { useState, useEffect, useMemo } from "react";
import TiledAnnotationsGrid from "./TiledAnnotationsGrid";

const AnnotationsViewer = ({
  selectedTile,
  gridDataCnn1,
  numRows,
  numCols,
  setSelectedTile,
  colorMapping,
  gridSize,
  gridFontSize,
  topLeftTile,
  selectedOverlay,
  setDragActivated,
  setFocusedTile,
  focusedTile,
  dragActivated,
  tileImages,
  setQuestionCommentOpen,
  questionCommentOpen,
  questionMarkComments,
  questionMarkCommentActions,
}) => {
  const [tileWidth, setTileWidth] = useState(0);

  useEffect(() => {
    function calcTileWidth() {
      // Subtract 4 for margins
      return Math.floor(gridSize / 10) - 4;
    }

    setTileWidth(calcTileWidth());
  }, [gridSize]);

  const renderVisibleTiles = useMemo(() => {
    const tiles = [];
    const gridData = gridDataCnn1;

    for (let x = 0; x < 10; x++) {
      for (let y = 0; y < 10; y++) {
        const currentRow = (topLeftTile.row + x + numRows) % numRows;
        const currentCol = (topLeftTile.col + y + numCols) % numCols;
        const key = `${currentRow}.${currentCol}`;

        const disabled = false;
        const tileValue = gridData.get(key);

        tiles.push({
          value: tileValue,
          row: currentRow,
          col: currentCol,
          disabled,
        });
      }
    }

    return tiles;
  }, [topLeftTile, gridDataCnn1, numRows, numCols]);

  return (
    <div
      onMouseDown={() => {
        setDragActivated(true);
        setSelectedTile({ row: focusedTile?.row, col: focusedTile?.col });
      }}
      onMouseUp={() => {
        setDragActivated(false);
        setSelectedTile({ row: focusedTile?.row, col: focusedTile?.col });
      }}
    >
      <TiledAnnotationsGrid
        tiles={renderVisibleTiles}
        selectedTile={selectedTile}
        setSelectedTile={setSelectedTile}
        colorMapping={colorMapping}
        gridSize={gridSize}
        gridFontSize={gridFontSize}
        tileWidth={tileWidth}
        selectedOverlay={selectedOverlay}
        setFocusedTile={setFocusedTile}
        dragActivated={dragActivated}
        tileImages={tileImages}
        setQuestionCommentOpen={setQuestionCommentOpen}
        questionCommentOpen={questionCommentOpen}
        questionMarkCommentActions={questionMarkCommentActions}
        questionMarkComments={questionMarkComments}
      />
    </div>
  );
};

export default AnnotationsViewer;
