import AnnotationsTile from "./AnnotationsTile";

const TiledAnnotationsGrid = ({
  tiles,
  selectedTile,
  setSelectedTile,
  colorMapping,
  gridSize,
  gridFontSize,
  tileWidth,
  selectedOverlay,
  setFocusedTile,
  dragActivated,
  tileImages,
  setQuestionCommentOpen,
  questionCommentOpen,
  questionMarkComments,
  questionMarkCommentActions,
  allTags,
  tagPalette,
  tagBadgeSize,
  tagBadgeFontSize,
}) => {
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
          <AnnotationsTile
            key={index}
            selected={selected}
            tile={tile}
            setSelectedTile={setSelectedTile}
            tileWidth={tileWidth}
            colorMapping={colorMapping}
            gridFontSize={gridFontSize}
            selectedOverlay={selectedOverlay}
            setFocusedTile={setFocusedTile}
            dragActivated={dragActivated}
            tileImages={tileImages}
            setQuestionCommentOpen={setQuestionCommentOpen}
            questionCommentOpen={questionCommentOpen}
            questionMarkCommentActions={questionMarkCommentActions}
            allTags={allTags}
            tagPalette={tagPalette}
            tagBadgeSize={tagBadgeSize}
            tagBadgeFontSize={tagBadgeFontSize}
            questionMarkComments={questionMarkComments}
          />
        );
      })}
    </div>
  );
};

export default TiledAnnotationsGrid;
