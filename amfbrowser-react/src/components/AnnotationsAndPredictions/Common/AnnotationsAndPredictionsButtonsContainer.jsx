import React, { useContext } from "react";
import AnnotationsAndPredictionsButtons from "./AnnotationsAndPredictionsButtons";
import { GlobalContextProvider } from "../../../contexts/Contexts";
import "../styles/selectors.css";

const AnnotationsAndPredictionsButtonsContainer = ({
  icons,
  selectedTile,
  colorMapping,
  onClick,
  gridData,
  size,
  isCnn1,
  selectIcons,
  onQuestionClick,
  onQuestionNext,
  iconFontSize,
  showAnnotations,
}) => {
  const { colonisationType } = useContext(GlobalContextProvider);

  let iconArray = Array.from(icons);
  let row1IconArray, row2IconArray;
  if (colonisationType !== "am") {
    row1IconArray = iconArray.slice(0, 5);
    row2IconArray = iconArray.slice(5, 10);
  }

  return (
    <div
      id="AnnotationsAndPredictionsButtonsWrapper"
      className="fullHeight fullWidth flexRow"
      style={{
        backgroundColor: `${selectIcons ? "rgb(243, 240, 237)" : "rgb(234, 230, 226)"}`,
      }}
    >
      <div className="buttonWrapper fullHeight fullWidth flexRowCenter">
        <div
          className="mainButtonArea fullHeight flexRow"
          style={{ alignItems: "center" }}
        >
          {colonisationType === "am" && (
            <AnnotationsAndPredictionsButtons
              iconArray={iconArray}
              gridData={gridData}
              selectedTile={selectedTile}
              showAnnotations={showAnnotations}
              isCnn1={isCnn1}
              size={size}
              iconFontSize={iconFontSize}
              colorMapping={colorMapping}
              onClick={onClick}
            />
          )}
          {colonisationType !== "am" && (
            <div style={{ display: "flex", flexDirection: "column" }}>
              <AnnotationsAndPredictionsButtons
                iconArray={row1IconArray}
                gridData={gridData}
                selectedTile={selectedTile}
                showAnnotations={showAnnotations}
                isCnn1={isCnn1}
                size={size}
                iconFontSize={iconFontSize}
                colorMapping={colorMapping}
                onClick={onClick}
              />
              <AnnotationsAndPredictionsButtons
                iconArray={row2IconArray}
                gridData={gridData}
                selectedTile={selectedTile}
                showAnnotations={showAnnotations}
                isCnn1={isCnn1}
                size={size}
                iconFontSize={iconFontSize}
                colorMapping={colorMapping}
                onClick={onClick}
              />
            </div>
          )}
        </div>

        {isCnn1 && showAnnotations && (
          <div
            className="questionButtonArea flexRow"
            style={{
              height: "80px",
              marginLeft: "70px",
              alignItems: "center",
              fontSize: `${iconFontSize}rem`,
            }}
          >
            <div
              className="noselect question-mark flexRowCenter"
              style={{
                height: size * 0.5,
                width: size,
                borderRadius: "20px",
                lineHeight: 1.5,
                marginRight: "10px",
                fontSize: `${iconFontSize}rem`,
              }}
              onClick={() => onQuestionClick()}
            >
              ?
            </div>
            <div
              className="noselect question-next flexRowCenter"
              style={{
                height: size * 0.5,
                width: size,
                borderRadius: "20px",
                lineHeight: 1.5,
                marginLeft: "10px",
                fontSize: `${iconFontSize}rem`,
              }}
              onClick={() => onQuestionNext()}
            >
              Next
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default AnnotationsAndPredictionsButtonsContainer;
