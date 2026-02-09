import Divider from "@mui/material/Divider";
import List from "@mui/material/List";
import ListItemButton from "@mui/material/ListItemButton";
import ListItemText from "@mui/material/ListItemText";
import TextField from "@mui/material/TextField";
import { useEffect, useState } from "react";

import "./styles/existing.css";

const ImageNameContainer = ({
  imageNames,
  fetchPredictionsAndAnnotations,
  colonisationType,
  selectedImageName,
  setSelectedImageName,
}) => {
  const [filter, setFilter] = useState("");

  const handleListItemClick = (imageName, index) => {
    setSelectedImageName(imageName);
    fetchPredictionsAndAnnotations(imageName);
  };

  useEffect(() => {
    if (selectedImageName !== null) {
      fetchPredictionsAndAnnotations(selectedImageName);
    }
  }, [colonisationType]);

  return (
    <div
      id="imageNameContainer"
      className="fullHeight flexColumn"
      style={{
        minHeight: "250px",
        width: "760px",
        minWidth: "760px",
      }}
    >
      <div
        style={{
          marginBottom: "10px",
          display: "flex",
          height: "56px",
        }} /*className="image-name-title"*/
      >
        <TextField
          id="standard-basic"
          label="Search Images"
          variant="outlined"
          sx={{ width: "350px", marginRight: "20px" }}
          onChange={(event) => {
            setFilter(event.currentTarget.value);
          }}
        />
        <div
          className="selectedImageNameWrapper"
          style={{
            width: "350px",
            height: "48px",
            marginLeft: "20px",
            display: "flex",
            alignItems: "center",
            fontWeight: selectedImageName ? 600 : 400,
            fontSize: "15px",
            color: selectedImageName ? "rgb(12, 110, 207)" : "lightgrey",
          }}
        >
          {selectedImageName ?? "Please select image"}
        </div>
      </div>
      <div
        style={{
          height: "calc(100% - 66px)", // 66px = 56px height of "Search images" + 10px  margin
          width: "50%",
          display: "flex",
          paddingBottom: "10px",
          flexDirection: "column",
        }}
      >
        <List
          sx={{
            width: 350,
            marginRight: "20px",
            bgcolor: "white",
            overflow: "auto",
            padding: 0,
          }}
        >
          {/* We could potentially put secondary text to list items if we wanted to include additional information, e.g. number of annotations and predictions? */}
          <Divider variant="fullWidth" /*component="li"*/ />
          {imageNames
            ?.sort()
            .filter((value) =>
              value[0].toLowerCase().includes(filter.toLowerCase()),
            )
            .map((imageName, index) => (
              <ListItemButton
                key={imageName}
                sx={{ paddingTop: "4px", paddingBottom: "4px" }}
                selected={imageName === selectedImageName}
                onClick={() => handleListItemClick(imageName, index)}
                divider
              >
                <ListItemText
                  sx={{ fontFamily: "Franklin Gothic" }}
                  primary={imageName} /*secondary="Jan 9, 2014"*/
                />
              </ListItemButton>
            ))}
        </List>
      </div>
    </div>
  );
};

export default ImageNameContainer;
