import React, { useState } from "react";
import IconButton from "@mui/material/IconButton";
import ChevronLeftIcon from "@mui/icons-material/ChevronLeft";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";
import FormLabel from "@mui/material/FormLabel";
import { styled } from "@mui/material/styles";
import SidebarRadioGroup from "../Common/SidebarRadioGroup";

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

const PredictionsLeftSidebar = ({
  level,
  setLevel,
  colorMapping,
  icons,
  keyBindings,
}) => {
  const [isOpen, setIsOpen] = useState(false);

  const handleSidebarChange = () => {
    // To potentially be moved up to the parent component
    setIsOpen(!isOpen);
  };

  return (
    <div
      id="PredictionsLeftSidebar"
      className="fullHeight flexColumn"
      style={{
        width: isOpen ? "200px" : "100px",
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
        label="Model"
        value={level}
        onChange={(e) => setLevel(e.target.value)}
        options={["CNN 1", "CNN 2"]}
        disabledOptions={["CNN 2"]}
      />
      <SidebarDivider />

      {/* Legend */}
      <div id="sidebarLegend" style={{ padding: "20px 18px" }}>
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
                className="flexRowCenter"
                style={{
                  width: "50px",
                  height: "20px",
                  backgroundColor: `${colorMapping.get(value)}`,
                  color: "white",
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
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default PredictionsLeftSidebar;
