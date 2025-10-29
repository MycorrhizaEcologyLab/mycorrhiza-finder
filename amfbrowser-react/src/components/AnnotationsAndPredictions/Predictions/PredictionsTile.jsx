import React, { memo, useContext } from "react";
import "chart.js/auto";
import { Pie } from "react-chartjs-2";
import { GlobalContextProvider } from "../../../contexts/Contexts";

const PredictionsTile = memo(
  ({
    selected,
    tile,
    setSelectedTile,
    tileWidth,
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
    const { colonisationType } = useContext(GlobalContextProvider);

    const values = tile?.predictions.map((item) => item[1]);
    const confidenceIsBelowThreshold =
      Math.max(...values) < conversionThreshold;

    const data = {
      labels: tile?.predictions.map((item) => item[0]),
      datasets: [
        {
          data: values,
          backgroundColor:
            colonisationType === "am"
              ? [
                  "#002060",
                  "#e8c775",
                  "black",
                  "#03F0FC",
                  "#F84646",
                  "#7030A0",
                  "orange",
                ]
              : [
                  "#002060",
                  "#824201",
                  "#04bd4e",
                  "#e8c775",
                  "black",
                  "#c40041",
                  "#03F0FC",
                  "#F84646",
                  "#c1d602",
                  "#7030A0",
                  "#FFA500",
                ],
        },
      ],
    };

    const options = {
      plugins: {
        legend: {
          display: false,
        },
      },
    };

    return (
      <div
        className="noselect"
        style={{
          width: `${tileWidth}px`,
          height: `${tileWidth}px`,
          lineHeight: `${tileWidth}px`,
          borderRadius: "10px",
          margin: "2px",
          outline: `solid ${selected ? "3px red" : "1px grey"}`,
          textAlign: "center",
          verticalAlign: "middle",
          color: "white",
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          position: "relative",
          backgroundColor: `${confidenceIsBelowThreshold ? "#ffcccb" : ""}`,
        }}
        onClick={() => {
          const newValidationIdx = validationMetricIdxMap.get(
            `${tile?.row}.${tile?.col}`,
          );
          setCurrentValidationIdx(newValidationIdx);

          const newConfidenceIdx = confidenceMetricIdxMap.get(
            `${tile?.row}.${tile?.col}`,
          );
          setCurrentConfidenceIdx(newConfidenceIdx);

          setSelectedTile({ row: tile?.row, col: tile?.col });
        }}
      >
        {tile?.annotations && (
          <div
            style={{
              width: `${tileWidth}px`,
              height: `${tileWidth}px`,
              lineHeight: `${tileWidth}px`,
              borderRadius: "10px",
              textAlign: "center",
              verticalAlign: "middle",
              backgroundColor: colorMapping.get(tile.annotations),
              color: "white",
              fontSize: `${gridFontSize}rem`,
            }}
            onClick={() => setSelectedTile({ row: tile?.row, col: tile?.col })}
          >
            {tile.annotations}
          </div>
        )}
        {tile?.predictions && !tile?.annotations && (
          <div
            className="chart-div"
            style={{
              width: `${tileWidth - 20}px`,
              height: `${tileWidth - 20}px`,
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
              borderRadius: "5px",
            }}
          >
            <Pie data={data} options={options} />
          </div>
        )}
        {convertWithContext && tile?.contextualLabel && (
          <div
            title="Contextual Label"
            style={{
              width: `${contextualLabelSize}px`,
              height: `${contextualLabelSize}px`,
              lineHeight: `${contextualLabelSize}px`,
              borderRadius: "5px",
              position: "absolute",
              bottom: "4px",
              right: "4px",
              textAlign: "center",
              verticalAlign: "middle",
              backgroundColor: colorMapping.get(tile.contextualLabel),
              color: "white",
              fontSize: `${contextualLabelFontSize}px`,
            }}
          >
            {tile.contextualLabel}
          </div>
        )}
      </div>
    );
  },
);

export default PredictionsTile;
