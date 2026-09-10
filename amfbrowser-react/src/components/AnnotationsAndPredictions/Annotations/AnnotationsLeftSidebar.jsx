import { useState } from "react";
import IconButton from "@mui/material/IconButton";
import ChevronLeftIcon from "@mui/icons-material/ChevronLeft";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";
import FormLabel from "@mui/material/FormLabel";
import { styled } from "@mui/material/styles";
import SidebarRadioGroup from "../Common/SidebarRadioGroup";
import AnnotationsTagPanel from "./AnnotationsTagPanel";

const DrawerHeader = styled("div")(({ theme }) => ({
  display: "flex",
  flexDirection: "column",
  height: "65px",
  width: "100%",
  alignItems: "center",
  justifyContent: "center",
  padding: theme.spacing(0, 1),
}));

const SidebarDivider = () => (
  <div
    className="annotationsSidebarDivider"
    style={{
      width: "100%",
      borderBottomStyle: "solid",
      borderBottomColor: "lightgray",
      borderBottomWidth: "2px",
    }}
  />
);

const AnnotationsLeftSidebar = ({
  mode,
  headerToCountsMapCnn1,
  colorMapping,
  icons,
  handleModeChange,
  keyBindings,
  allTags,
  tagPalette,
  selectedTileTags,
  hasSelectedClass,
  onCreateTag,
  onDeleteTag,
  onToggleTag,
  onClearTags,
  setTagInputOpen,
}) => {
  const [isOpen, setIsOpen] = useState(false);

  const handleSidebarChange = () => {
    setIsOpen(!isOpen);
  };

  return (
    <div
      id="AnnotationsLeftSidebar"
      className="sidebarColumn"
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100%",
        // Wider when open so the tag name field, colour swatch and add
        // button fit on one row.
        width: isOpen ? "260px" : "135px",
        overflowY: "auto",
        overflowX: "clip",
        borderRight: "2px solid lightgray",
      }}
    >
      <DrawerHeader>
        <IconButton onClick={handleSidebarChange}>
          {isOpen ? <ChevronLeftIcon /> : <ChevronRightIcon />}
        </IconButton>
        <div style={{ fontSize: "12px", color: "rgb(120, 117, 101)" }}>
          Settings
        </div>
      </DrawerHeader>
      <SidebarDivider />
      <SidebarRadioGroup
        label="Mode"
        value={mode}
        onChange={handleModeChange}
        options={["Select", "Overlay"]}
      />
      <SidebarDivider />

      {/* Legend */}
      <div
        id="sidebarLegend"
        style={{ padding: "20px 18px", flexShrink: 0, minWidth: 0 }}
      >
        <FormLabel
          sx={{
            display: "flex",
            paddingBottom: "20px",
            color: "black",
            fontWeight: 600,
            fontSize: "12px",
          }}
        >
          Legend
        </FormLabel>
        {Array.from(icons).map(([key, value]) => {
          return (
            <div
              key={key + "-wrapper"}
              style={{
                width: "100%",
                height: "30px",
                display: "flex",
              }}
            >
              <div
                key={key + "-label"}
                style={{
                  width: "50px",
                  height: "20px",
                  display: "flex",
                  backgroundColor: `${colorMapping.get(value)}`,
                  color: "white",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: "13px",
                  borderRadius: "10px",
                  borderWidth: "1px",
                  borderColor: "darkgray",
                  borderStyle: "solid",
                }}
              >
                {value}
              </div>
              <div
                key={key + "-longName"}
                className="legendLongNameText"
                style={{
                  fontSize: "13px",
                  paddingLeft: "10px",
                  width: isOpen ? "120px" : "0px",
                  opacity: isOpen ? 1 : 0,
                }}
              >
                {key} ({keyBindings.get(value)})
              </div>
              <div
                key={key + "-count"}
                style={{ width: "35px", fontSize: "13px", textAlign: "right" }}
              >
                {headerToCountsMapCnn1.get(value)}
              </div>
            </div>
          );
        })}
      </div>

      <SidebarDivider />

      <AnnotationsTagPanel
        allTags={allTags}
        tagPalette={tagPalette}
        selectedTileTags={selectedTileTags}
        hasSelectedClass={hasSelectedClass}
        onCreateTag={onCreateTag}
        onDeleteTag={onDeleteTag}
        onToggleTag={onToggleTag}
        onClearTags={onClearTags}
        setTagInputOpen={setTagInputOpen}
        isOpen={isOpen}
      />
    </div>
  );
};

export default AnnotationsLeftSidebar;
