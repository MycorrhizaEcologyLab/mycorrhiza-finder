import React, { useState, useEffect, useContext } from "react";
import { styled } from "@mui/material/styles";
import CircularProgressWithLabel from "../../Utils/CircularProgressWithLabel";
import { GlobalContextProvider } from "../../../contexts/Contexts";

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

const AnnotationsRightSidebar = ({
  headerToCountsMapCnn1,
  tileEdge,
  cnn1Annotations,
  totalTiles,
}) => {
  const { colonisationType } = useContext(GlobalContextProvider);

  // useState
  const [amColPc, setAmColPc] = useState(0);
  const [blueCoilsColPc, setBlueCoilsColPc] = useState(0);
  const [brownCoilsColPc, setBrownCoilsColPc] = useState(0);
  const [typeTwoCoilsColPc, setTypeTwoColPc] = useState(0);
  const [dseColPc, setDseColPc] = useState(0);
  const [totalColPc, setTotalColPc] = useState(0);

  useEffect(() => {
    if (colonisationType === "am") {
      // Total ignoring background & unreadable
      const rootTotal =
        headerToCountsMapCnn1.get("AM+") +
        headerToCountsMapCnn1.get("N-") +
        headerToCountsMapCnn1.get("D") +
        headerToCountsMapCnn1.get("AD+");
      if (rootTotal !== 0) {
        const pcAmColonised =
          (headerToCountsMapCnn1.get("AM+") + headerToCountsMapCnn1.get("AD+")) /
          rootTotal;
        const pcDseColonised =
          (headerToCountsMapCnn1.get("D") + headerToCountsMapCnn1.get("AD+")) /
          rootTotal;
        const pcTotalColonised =
          pcAmColonised +
          pcDseColonised -
          headerToCountsMapCnn1.get("AD+") / rootTotal;

        setAmColPc(pcAmColonised);
        setDseColPc(pcDseColonised);
        setTotalColPc(pcTotalColonised);
      } else {
        setAmColPc(0);
        setDseColPc(0);
        setTotalColPc(0);
      }
    } else {
      // Total ignoring background, main root & unreadable
      const rootTotal =
        headerToCountsMapCnn1.get("Bl+") +
        headerToCountsMapCnn1.get("Br+") +
        headerToCountsMapCnn1.get("T+") +
        headerToCountsMapCnn1.get("N-") +
        headerToCountsMapCnn1.get("D") +
        headerToCountsMapCnn1.get("E+") +
        headerToCountsMapCnn1.get("ED+");

      if (rootTotal !== 0) {
        const pcBlueCoilsColonised =
          headerToCountsMapCnn1.get("Bl+") / rootTotal;
        const pcBrownCoilsColonised =
          headerToCountsMapCnn1.get("Br+") / rootTotal;
        const pcTypeTwoColonised = headerToCountsMapCnn1.get("T+") / rootTotal;
        const pcDseColonised =
          (headerToCountsMapCnn1.get("D") + headerToCountsMapCnn1.get("ED+")) /
          rootTotal;
        const pcTotalColonised =
          pcBlueCoilsColonised +
          pcBrownCoilsColonised +
          pcTypeTwoColonised +
          headerToCountsMapCnn1.get("E+") / rootTotal;

        setBlueCoilsColPc(pcBlueCoilsColonised);
        setBrownCoilsColPc(pcBrownCoilsColonised);
        setTypeTwoColPc(pcTypeTwoColonised);
        setDseColPc(pcDseColonised);
        setTotalColPc(pcTotalColonised);
      } else {
        setBlueCoilsColPc(0);
        setBrownCoilsColPc(0);
        setTypeTwoColPc(0);
        setDseColPc(0);
        setTotalColPc(0);
      }
    }
  }, [headerToCountsMapCnn1, colonisationType]);

  return (
    <div
      id="AnnotationsRightSidebar"
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100%",
        width: "150px",
        overflowY: "auto",
        borderLeft: "2px solid lightgray",
      }}
    >
      <DrawerHeader>
        <div style={{ fontSize: "12px", color: "rgb(120, 117, 101)" }}>
          Statistics
        </div>
      </DrawerHeader>
      <SidebarDivider />
      {colonisationType === "am" && (
        <div
          id="colonisedSummaryWrapper"
          style={{
            height: "120px",
            display: "flex",
            flexDirection: "column",
            justifyContent: "center",
            alignItems: "center",
          }}
        >
          <div style={{ fontSize: "12px", fontWeight: 600, margin: "5px 0px" }}>
            Colonised summary
          </div>
          <div style={{ display: "flex", margin: "5px 0px" }}>
            <div style={{ width: "45px", fontSize: "12px" }}>AM:</div>
            <div style={{ width: "50px", fontSize: "12px", textAlign: "end" }}>
              {((amColPc + Number.EPSILON) * 100).toFixed(2)}%
            </div>
          </div>
          <div style={{ display: "flex", margin: "5px 0px" }}>
            <div style={{ width: "45px", fontSize: "12px" }}>DSE:</div>
            <div style={{ width: "50px", fontSize: "12px", textAlign: "end" }}>
              {((dseColPc + Number.EPSILON) * 100).toFixed(2)}%
            </div>
          </div>
          <div style={{ display: "flex", margin: "5px 0px" }}>
            <div style={{ width: "45px", fontSize: "12px" }}>Total:</div>
            <div style={{ width: "50px", fontSize: "12px", textAlign: "end" }}>
              {((totalColPc + Number.EPSILON) * 100).toFixed(2)}%
            </div>
          </div>
        </div>
      )}
      {colonisationType === "erm" && (
        <div
          id="colonisedSummaryWrapper"
          style={{
            height: "200px",
            display: "flex",
            flexDirection: "column",
            justifyContent: "center",
            alignItems: "center",
          }}
        >
          <div style={{ fontSize: "12px", fontWeight: 600, margin: "5px 0px" }}>
            Colonised summary
          </div>
          <div style={{ display: "flex", margin: "5px 0px" }}>
            <div style={{ width: "65px", fontSize: "12px" }}>Blue Coils:</div>
            <div style={{ width: "50px", fontSize: "12px", textAlign: "end" }}>
              {((blueCoilsColPc + Number.EPSILON) * 100).toFixed(2)}%
            </div>
          </div>
          <div style={{ display: "flex", margin: "5px 0px" }}>
            <div style={{ width: "65px", fontSize: "12px" }}>Brown Coils:</div>
            <div style={{ width: "50px", fontSize: "12px", textAlign: "end" }}>
              {((brownCoilsColPc + Number.EPSILON) * 100).toFixed(2)}%
            </div>
          </div>
          <div style={{ display: "flex", margin: "5px 0px" }}>
            <div style={{ width: "65px", fontSize: "12px" }}>Type Two:</div>
            <div style={{ width: "50px", fontSize: "12px", textAlign: "end" }}>
              {((typeTwoCoilsColPc + Number.EPSILON) * 100).toFixed(2)}%
            </div>
          </div>
          <div style={{ display: "flex", margin: "5px 0px" }}>
            <div style={{ width: "65px", fontSize: "12px" }}>DSE:</div>
            <div style={{ width: "50px", fontSize: "12px", textAlign: "end" }}>
              {((dseColPc + Number.EPSILON) * 100).toFixed(2)}%
            </div>
          </div>
          <div style={{ display: "flex", margin: "5px 0px" }}>
            <div style={{ width: "65px", fontSize: "12px" }}>Total:</div>
            <div style={{ width: "50px", fontSize: "12px", textAlign: "end" }}>
              {((totalColPc + Number.EPSILON) * 100).toFixed(2)}%
            </div>
          </div>
        </div>
      )}

      <SidebarDivider />
      <div
        style={{
          width: "100%",
          height: "150px",
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          alignItems: "center",
        }}
      >
        <div style={{ fontSize: "12px", fontWeight: 600, margin: "5px 0px" }}>
          Tile size
        </div>
        <div style={{ fontSize: "12px", margin: "5px 0px" }}>{tileEdge}</div>
      </div>
      <SidebarDivider />
      <div
        style={{
          width: "100%",
          height: "150px",
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          alignItems: "center",
        }}
      >
        <div style={{ fontSize: "12px", fontWeight: 600, margin: "5px 0px" }}>
          Coverage
        </div>
        <div style={{ marginTop: "10px" }}>
          <CircularProgressWithLabel
            value={Math.floor((cnn1Annotations.size / totalTiles) * 100)}
          />
        </div>
      </div>

      <SidebarDivider />
    </div>
  );
};

export default AnnotationsRightSidebar;
