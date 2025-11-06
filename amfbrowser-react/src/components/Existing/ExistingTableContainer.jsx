import React from "react";
import Paper from "@mui/material/Paper";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableContainer from "@mui/material/TableContainer";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";
import Checkbox from "@mui/material/Checkbox";
import { FaTrashAlt } from "react-icons/fa";

const columns = [
  {
    id: "enabled",
    label: "Enabled",
    minWidth: 90,
    align: "center",
    format: "checkbox",
  },
  {
    id: "cnn1_annotations_exist",
    label: "Annotations",
    minWidth: 150,
    align: "center",
    format: "downloadButton",
    modelNumber: 1,
    analysis: "annotations",
  },
  {
    id: "cnn1_predictions_exist",
    label: "Predictions",
    minWidth: 150,
    align: "center",
    format: "downloadButton",
    modelNumber: 1,
    analysis: "predictions",
  },
  { id: "tileEdge", label: "Tile", minWidth: 65, align: "center" },
  {
    id: "timestamp",
    label: "Upload",
    minWidth: 115,
    align: "center",
    format: (value) => new Date(value).toLocaleString("en-GB"),
  },
  {
    id: "updatedAt",
    label: "Last Updated",
    minWidth: 115,
    align: "center",
    format: (value) => new Date(value).toLocaleString("en-GB"),
  },
  {
    id: "delete",
    label: "Delete",
    minWidth: 90,
    align: "center",
  },
];

const ExistingTableContainer = ({
  data,
  onDownload,
  handleSetToEnabled,
  setPendingDeletionId,
  colonisationType,
  imageName,
}) => {
  if (Object.entries(data).length === 0) {
    return (
      <div>
        No saved data for {imageName} for{" "}
        {colonisationType === "am" ? "AM" : "ErM"} colonisation.
      </div>
    );
  }

  return (
    <Paper
      sx={{
        width: "90%",
        minWidth: "1000px",
        maxWidth: "1500px",
        overflow: "hidden",
        borderStyle: "solid",
        borderWidth: "1px",
        borderColor: "lightGray",
      }}
    >
      <TableContainer sx={{ height: "100%", minWidth: 1000, maxWidth: 1500 }}>
        <Table stickyHeader aria-label="sticky table">
          <TableHead>
            <TableRow sx={{ height: "50px" }}>
              {columns.map((column) => (
                <TableCell
                  key={column.id}
                  align={column.align}
                  style={{ minWidth: column.minWidth, padding: "2px 10px" }}
                >
                  {column.label}
                </TableCell>
              ))}
            </TableRow>
          </TableHead>
          <TableBody>
            {Object.entries(data).map(([key, row]) => {
              return (
                <TableRow key={row.timestamp} selected={row.enabled === true}>
                  {columns.map((column) => {
                    const value = row[column.id];
                    return (
                      <TableCell
                        key={column.id}
                        align={column.align}
                        style={{
                          minWidth: column.minWidth,
                          padding: "8px 10px",
                          textAlign: "center",
                        }}
                      >
                        {/* 'downloadButton applies to the predictions and annotations */}
                        {column.format === "downloadButton" && value && (
                          <div className="fullWidth fullHeight flexRowCenter">
                            <div
                              onClick={() =>
                                onDownload(
                                  key,
                                  column.modelNumber,
                                  column.analysis,
                                  row["timestamp"],
                                )
                              }
                              className="existingTableDownloadButton"
                            >
                              Download
                            </div>
                          </div>
                        )}

                        {column.id === "tileEdge" && (
                          <div style={{ fontSize: "13px" }}>{value}</div>
                        )}
                        {column.id === "timestamp" && (
                          <div style={{ fontSize: "13px" }}>
                            {new Date(value).toLocaleString("en-GB")}
                          </div>
                        )}
                        {column.id === "updatedAt" && (
                          <div style={{ fontSize: "13px" }}>
                            {new Date(value).toLocaleString("en-GB")}
                          </div>
                        )}

                        {column.id === "enabled" && (
                          <Checkbox
                            color="primary"
                            checked={value}
                            onChange={(e) => handleSetToEnabled(e, key)}
                          />
                        )}

                        {column.id === "delete" && (
                          <FaTrashAlt
                            className="deleteIcon"
                            style={{
                              height: "20px",
                              width: "20px",
                              cursor: "pointer",
                            }}
                            onClick={() => setPendingDeletionId(key)}
                          />
                        )}
                      </TableCell>
                    );
                  })}
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </TableContainer>
    </Paper>
  );
};

export default ExistingTableContainer;
