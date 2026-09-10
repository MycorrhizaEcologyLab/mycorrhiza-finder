import { memo, useEffect, useRef, useState } from "react";
import { FaEyeSlash } from "react-icons/fa";
import { getTagColour } from "../../../config/AnnotationsAndPredictionsConfig";

const AnnotationsTile = memo(
  ({
    selected,
    tile,
    setSelectedTile,
    tileWidth,
    colorMapping,
    gridFontSize,
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
    const imageUrl = tileImages.get(`${tile.row}.${tile.col}`);
    const [showQuestionComment, setShowQuestionComment] = useState(false);
    const wrapperRef = useRef(null);

    // Only a small fraction of tiles carry sub-tags; untagged ones must render
    // exactly as before so the tagged ones stand out when scanning the grid.
    const tags = tile?.tags ?? [];
    const hasTags = tags.length > 0;
    const firstTagColour = hasTags
      ? getTagColour(tags[0], tagPalette, allTags)
      : null;

    useEffect(() => {
      function handleClickOutside(event) {
        if (wrapperRef.current && !wrapperRef.current.contains(event.target)) {
          setShowQuestionComment(false);
          setQuestionCommentOpen(false);
        }
      }

      document.addEventListener("mousedown", handleClickOutside);
      return () => {
        document.removeEventListener("mousedown", handleClickOutside);
      };
    }, [wrapperRef, setQuestionCommentOpen]);

    return (
      <div
        className="noselect annotation-tile"
        title={hasTags ? `Tags: ${tags.join(", ")}` : undefined}
        style={{
          width: `${tileWidth}px`,
          height: `${tileWidth}px`,
          outline: `solid ${selected ? "3px red" : "1px grey"}`,
          background: `url(${imageUrl})`,
          // An inset ring rather than the outline, which is already spoken
          // for by the selection state. White then the tag's own colour, so
          // it reads against both light and dark tile colours.
          boxShadow: hasTags
            ? `inset 0 0 0 2px #fff, inset 0 0 0 4px ${firstTagColour}`
            : undefined,
          position: hasTags ? "relative" : undefined,
        }}
        onClick={() => {
          setSelectedTile({ row: tile?.row, col: tile?.col });
        }}
        onDoubleClick={() => {
          if (tile?.value === "?" && !questionCommentOpen) {
            setShowQuestionComment(true);
            setQuestionCommentOpen(true);
          }
        }}
        onMouseEnter={() => {
          setFocusedTile({ row: tile?.row, col: tile?.col });
          if (dragActivated) {
            setSelectedTile({ row: tile?.row, col: tile?.col });
          }
        }}
      >
        {showQuestionComment && (
          <div ref={wrapperRef} className="question-comment-wrapper">
            <label
              htmlFor="questionComment"
              title="Comment re. question tile"
            >
              Question comment:
            </label>
            <input
              type="text"
              value={questionMarkComments.get(`${tile?.row}.${tile?.col}`)}
              onChange={(e) => {
                let newKey = `${tile?.row}.${tile?.col}`;
                if (e.target.value !== "") {
                  questionMarkCommentActions.set(newKey, e.target.value);
                } else {
                  questionMarkCommentActions.remove(newKey);
                }
              }}
              name="questionComment"
            />
          </div>
        )}
        {selectedOverlay === null && (
          <div
            style={{
              width: `${tileWidth}px`,
              height: `${tileWidth}px`,
              lineHeight: `${tileWidth}px`,
              borderRadius: "10px",
              backgroundColor: colorMapping.get(tile?.value),
              color: "white",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: `${gridFontSize}rem`,
            }}
          >
            {tile?.value}
          </div>
        )}
        {selectedOverlay !== null && tile?.value !== selectedOverlay && (
          <div
            style={{
              width: `${tileWidth}px`,
              height: `${tileWidth}px`,
              lineHeight: `${tileWidth}px`,
              borderRadius: "10px",
              backgroundColor: "lightgrey",
              color: "white",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: `${gridFontSize}rem`,
            }}
          >
            <FaEyeSlash />
          </div>
        )}
        {selectedOverlay !== null && tile?.value === selectedOverlay && (
          <div
            style={{
              width: `${tileWidth}px`,
              height: `${tileWidth}px`,
              lineHeight: `${tileWidth}px`,
              borderRadius: "10px",
              backgroundColor: colorMapping.get(tile?.value),
              color: "white",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: `${gridFontSize}rem`,
            }}
          >
            {tile?.value}
          </div>
        )}
        {hasTags && (
          <div
            style={{
              // A dot for a single tag, the count once there are several -
              // a digit is unreadable at the smallest tile sizes otherwise.
              width: `${tags.length > 1 ? tagBadgeSize : Math.round(tagBadgeSize * 0.6)}px`,
              height: `${tags.length > 1 ? tagBadgeSize : Math.round(tagBadgeSize * 0.6)}px`,
              lineHeight: `${tagBadgeSize}px`,
              borderRadius: "50%",
              position: "absolute",
              bottom: "3px",
              right: "3px",
              backgroundColor: firstTagColour,
              border: "1px solid #fff",
              color: "white",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: `${tagBadgeFontSize}px`,
              fontWeight: 700,
              boxSizing: "border-box",
              pointerEvents: "none",
            }}
          >
            {tags.length > 1 ? tags.length : ""}
          </div>
        )}
      </div>
    );
  },
);

export default AnnotationsTile;
