import React, { memo, useState, useRef, useEffect } from "react";
import { FaLock, FaEyeSlash } from "react-icons/fa";

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
  }) => {
    const imageUrl = tileImages.get(`${tile.row}.${tile.col}`);
    const [showQuestionComment, setShowQuestionComment] = useState(false);
    const wrapperRef = useRef(null);

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
        style={{
          width: `${tileWidth}px`,
          height: `${tileWidth}px`,
          outline: `solid ${selected ? "3px red" : "1px grey"}`,
          background: `url(${imageUrl})`,
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
      </div>
    );
  },
);

export default AnnotationsTile;
