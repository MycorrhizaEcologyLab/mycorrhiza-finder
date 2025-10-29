const AnnotationsAndPredictionsButtons = ({
  iconArray,
  gridData,
  selectedTile,
  showAnnotations,
  isCnn1,
  size,
  iconFontSize,
  colorMapping,
  onClick,
}) => {
  return (
    <div style={{ display: "flex" }}>
      {iconArray
        // Remove ? from icons, as it is already handled in it's own box
        .filter(([key, value]) => value !== "?")
        .map(([key, value]) => {
          let selectedValue = gridData.get(
            selectedTile.row.toString() + "." + selectedTile.col.toString(),
          );
          let isSelected = false;
          // Only show selected for annotations
          if (showAnnotations) {
            if (isCnn1) {
              isSelected = value === selectedValue;
            } else {
              isSelected = selectedValue?.includes(value);
            }
          }
          return (
            <div
              key={key}
              className="noselect flexColumnCenter"
              style={{
                width: `${size}px`,
                height: size * 0.5,
                borderRadius: "20px",
                margin: "5px 10px",
                outline: `solid ${isSelected ? "2px red" : "1px grey"}`,
                textAlign: "center",
                verticalAlign: "middle",
                color: "white",
                fontSize: `${iconFontSize}rem`,
                backgroundColor: `${colorMapping.get(value)}`,
                cursor: "pointer",
              }}
              onClick={() => onClick(value)}
            >
              <div>{value}</div>
            </div>
          );
        })}
    </div>
  );
};

export default AnnotationsAndPredictionsButtons;
