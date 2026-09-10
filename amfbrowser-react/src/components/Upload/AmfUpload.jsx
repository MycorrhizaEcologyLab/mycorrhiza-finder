import ErrorOutlineOutlinedIcon from "@mui/icons-material/ErrorOutlineOutlined";
import InfoOutlinedIcon from "@mui/icons-material/InfoOutlined";
import FormControlLabel from "@mui/material/FormControlLabel";
import FormLabel from "@mui/material/FormLabel";
import Radio from "@mui/material/Radio";
import RadioGroup from "@mui/material/RadioGroup";
import TextField from "@mui/material/TextField";
import Tooltip from "@mui/material/Tooltip";
import Papa from "papaparse";
import { useContext, useEffect, useRef, useState } from "react";
import { toast } from "react-toastify";

import PredictionsApi from "../../api/amfinderApi";
import { GlobalContextProvider } from "../../contexts/Contexts";
import PrimaryButton from "../Utils/PrimaryButton";
import { isValidFilePath } from "../Utils/utils";
import ValidationIcon from "../Utils/ValidationIcon";
import "./styles/upload.css";

const AmfUpload = () => {
  const { colonisationType, tileEdge, setTileEdge } = useContext(
    GlobalContextProvider,
  );

  const [imageName, setImageName] = useState("");
  const [cnn, setCnn] = useState(1);
  const [currentToast, setCurrentToast] = useState(0);

  const [folderPath, setFolderPath] = useState("");
  const [isImporting, setIsImporting] = useState(false);
  const [importResult, setImportResult] = useState(null);

  const uploadAnnotationsInputRef = useRef(null);
  const uploadPredictionsInputRef = useRef(null);

  useEffect(() => {
    if (colonisationType === "am") {
      setTileEdge(252);
    } else {
      setTileEdge(126);
    }
  }, [colonisationType]);

  const validateHeaders = (headers, mandatory, optional) => {
    // Check missing columns
    const missing = mandatory.filter((col) => !headers.includes(col));
    if (missing.length) {
      return {
        valid: false,
        message: `Missing mandatory column(s): ${missing.join(", ")}.`,
      };
    }

    const allowedCols = new Set([...mandatory, ...optional]);

    // Check for extra columns
    const extra = headers.filter((col) => !allowedCols.has(col));
    if (extra.length) {
      return {
        valid: false,
        message: `Found extra column(s): ${extra.join(", ")}.`,
      };
    }

    return { valid: true };
  };

  const onFileUpload = async (event, isAnnotations) => {
    const toastId = `onFileUpload${currentToast}`;
    setCurrentToast(currentToast + 1);

    if (
      event.target.files[0] !== undefined &&
      !event.target.files[0].name.includes(imageName)
    ) {
      toast.error("CSV name must contain the image name for a valid upload!");

      if (isAnnotations) {
        uploadAnnotationsInputRef.current.value = "";
      } else {
        uploadPredictionsInputRef.current.value = "";
      }

      return;
    }

    if (event.target.files[0] !== undefined) {
      Papa.parse(event.target.files[0], {
        header: true,
        skipEmptyLines: true,
        complete: async function (results) {
          if (!results.data.length) {
            toast.error("No data found in the CSV file");
            if (isAnnotations) {
              uploadAnnotationsInputRef.current.value = "";
            } else {
              uploadPredictionsInputRef.current.value = "";
            }

            return;
          }

          const headers = Object.keys(results.data[0]);
          let mandatoryColumns = [];
          let optionalColumns = [];

          if (colonisationType === "am") {
            mandatoryColumns = [
              "row",
              "col",
              "AMColonised",
              "Uncolonised",
              "Background",
              "Unreadable",
              "DSE",
              "Hybrid",
            ];
          } else {
            mandatoryColumns = [
              "row",
              "col",
              "BlueCoils",
              "BrownCoils",
              "TypeTwo",
              "Uncolonised",
              "Background",
              "MainRoot",
              "Unreadable",
              "DSE",
              "HybridErm",
              "HybridDse",
            ];
          }

          if (isAnnotations) {
            optionalColumns = ["Question", "QuestionComment", "Tags"];
          } else {
            optionalColumns = ["ContextualLabel"];
          }

          // Validate CSV
          const { valid, message } = validateHeaders(
            headers,
            mandatoryColumns,
            optionalColumns,
          );

          if (!valid) {
            toast.error(`CSV validation error: ${message}`, {
              toastId,
              autoClose: 5000,
            });

            if (isAnnotations) {
              uploadAnnotationsInputRef.current.value = "";
            } else {
              uploadPredictionsInputRef.current.value = "";
            }

            return;
          }

          let cnn1Values = [];
          try {
            if (isAnnotations) {
              results.data.forEach((value, key) => {
                // Save CNN1 annotations into a format accepted by DB
                // Full fixed width: [row, col, ...classes, Question,
                // QuestionComment, Tags]. Absent optional columns leave their
                // slot at its default rather than shortening the row.
                if (colonisationType === "am") {
                  const out = new Array(11).fill(0);
                  out[9] = "";
                  out[10] = "";
                  out[0] = parseInt(value["row"]);
                  out[1] = parseInt(value["col"]);
                  out[2] = parseInt(value["AMColonised"]);
                  out[3] = parseInt(value["Uncolonised"]);
                  out[4] = parseInt(value["Background"]);
                  out[5] = parseInt(value["Unreadable"]);
                  out[6] = parseInt(value["DSE"]);
                  out[7] = parseInt(value["Hybrid"]);
                  if ("Question" in value) {
                    out[8] = parseInt(value["Question"]);
                  }
                  if ("QuestionComment" in value) {
                    out[9] = value["QuestionComment"];
                  }
                  if ("Tags" in value) {
                    out[10] = value["Tags"];
                  }
                  cnn1Values.push(out);
                } else {
                  const out = new Array(15).fill(0);
                  out[13] = "";
                  out[14] = "";
                  out[0] = parseInt(value["row"]);
                  out[1] = parseInt(value["col"]);
                  out[2] = parseInt(value["BlueCoils"]);
                  out[3] = parseInt(value["BrownCoils"]);
                  out[4] = parseInt(value["TypeTwo"]);
                  out[5] = parseInt(value["Uncolonised"]);
                  out[6] = parseInt(value["Background"]);
                  out[7] = parseInt(value["MainRoot"]);
                  out[8] = parseInt(value["Unreadable"]);
                  out[9] = parseInt(value["DSE"]);
                  out[10] = parseInt(value["HybridErm"]);
                  out[11] = parseInt(value["HybridDse"]);
                  if ("Question" in value) {
                    out[12] = parseInt(value["Question"]);
                  }
                  if ("QuestionComment" in value) {
                    out[13] = value["QuestionComment"];
                  }
                  if ("Tags" in value) {
                    out[14] = value["Tags"];
                  }
                  cnn1Values.push(out);
                }
              });

              let body = {
                fileName: imageName,
                imageReferenceId: null,
                cnnOneValues: cnn1Values,
                colonisationType: colonisationType,
                tileEdge: tileEdge,
                enabled: true,
              };

              if (imageName !== "") {
                toast.info("Uploading annotations", {
                  toastId,
                  autoClose: false,
                });
                let output = await PredictionsApi.saveAnnotations(body, true);

                if (output === 200) {
                  toast.update(toastId, {
                    type: "success",
                    render: `Annotations succesfully uploaded for ${imageName}`,
                    autoClose: 5000,
                  });
                } else {
                  toast.update(toastId, {
                    type: "error",
                    render: `Failed to upload annotations for ${imageName}`,
                    autoClose: 5000,
                  });
                }
              }
            } else {
              results.data.forEach((value, key) => {
                // Save CNN1 predictions into a format accepted by DB
                if (colonisationType === "am") {
                  const out = new Array(9).fill(0);
                  out[0] = parseInt(value["row"]);
                  out[1] = parseInt(value["col"]);
                  out[2] = parseFloat(value["AMColonised"]);
                  out[3] = parseFloat(value["Uncolonised"]);
                  out[4] = parseFloat(value["Background"]);
                  out[5] = parseFloat(value["Unreadable"]);
                  out[6] = parseFloat(value["DSE"]);
                  out[7] = parseFloat(value["Hybrid"]);
                  if ("ContextualLabel" in value) {
                    out[8] = value["ContextualLabel"];
                  }
                  cnn1Values.push(out);
                } else {
                  const out = new Array(13).fill(0);
                  out[0] = parseInt(value["row"]);
                  out[1] = parseInt(value["col"]);
                  out[2] = parseFloat(value["BlueCoils"]);
                  out[3] = parseFloat(value["BrownCoils"]);
                  out[4] = parseFloat(value["TypeTwo"]);
                  out[5] = parseFloat(value["Uncolonised"]);
                  out[6] = parseFloat(value["Background"]);
                  out[7] = parseFloat(value["MainRoot"]);
                  out[8] = parseFloat(value["Unreadable"]);
                  out[9] = parseFloat(value["DSE"]);
                  out[10] = parseFloat(value["HybridErm"]);
                  out[11] = parseFloat(value["HybridDse"]);
                  if ("ContextualLabel" in value) {
                    out[12] = value["ContextualLabel"];
                  }
                  cnn1Values.push(out);
                }
              });

              let body = {
                fileName: imageName,
                imageReferenceId: null,
                cnnOneValues: cnn1Values,
                colonisationType: colonisationType,
                tileEdge: tileEdge,
              };

              if (imageName !== "") {
                toast.info("Uploading predictions", {
                  toastId,
                  autoClose: false,
                });
                let output = await PredictionsApi.savePredictions(body, true);

                if (output === 200) {
                  toast.update(toastId, {
                    type: "success",
                    render: `Predictions succesfully uploaded for ${imageName}`,
                    autoClose: 5000,
                  });
                } else {
                  toast.update(toastId, {
                    type: "error",
                    render: `Failed to upload predictions for ${imageName}`,
                    autoClose: 5000,
                  });
                }
              }
            }
          } catch (error) {
            toast.error("Failed to upload");
          }

          if (isAnnotations) {
            uploadAnnotationsInputRef.current.value = "";
          } else {
            uploadPredictionsInputRef.current.value = "";
          }

          return;
        },
      });
    }
  };

  const isFolderPathValid = isValidFilePath(folderPath);

  const importFolder = async () => {
    setIsImporting(true);
    setImportResult(null);
    toast.info("Importing folder - this may take a while for large batches", {
      toastId: "importFolder",
      autoClose: false,
    });

    const result = await PredictionsApi.importFolder({
      folderPath,
      colonisationType,
    });

    setIsImporting(false);

    if (!result) {
      toast.update("importFolder", {
        type: "error",
        render: "Failed to import folder",
        autoClose: 5000,
      });
      return;
    }

    setImportResult(result);
    toast.update("importFolder", {
      type: result.errors.length === 0 ? "success" : "warning",
      render: `Imported ${result.imported.length} file(s), ${result.errors.length} error(s)`,
      autoClose: 5000,
    });
  };

  return (
    <div id="amfUploadContainer" className="fullHeight fullWidth flexRowCenter">
      <div
        className="fullHeight fullWidth flexColumn"
        style={{ overflowY: "auto" }}
      >
        <div
          className="fullHeight fullWidth"
          style={{
            minWidth: "500px",
            minHeight: "300px",
          }}
        >
          <div className="fullHeight fullWidth">
            <div className="fullHeight fullWidth flexColumnCenter">
              <div id="batchImportSection" className="flexColumn">
                <FormLabel
                  sx={{
                    display: "flex",
                    justifyContent: "center",
                    color: "black",
                    fontWeight: 600,
                    marginBottom: "10px",
                  }}
                >
                  Batch Import Folder
                </FormLabel>
                <div className="flexRowCenter">
                  <TextField
                    label="Folder path"
                    variant="outlined"
                    error={folderPath !== "" && !isFolderPathValid}
                    value={folderPath}
                    sx={{ width: "350px", marginRight: "10px" }}
                    onChange={(event) => setFolderPath(event.currentTarget.value)}
                  />
                  <ValidationIcon
                    status={
                      folderPath === ""
                        ? "none"
                        : isFolderPathValid
                          ? "success"
                          : "error"
                    }
                  />
                </div>
                <div
                  className="flexRowCenter"
                  style={{ marginTop: "10px", marginBottom: "10px" }}
                >
                  <PrimaryButton
                    disabled={!isFolderPathValid || isImporting}
                    sx={{ minWidth: "200px" }}
                    onClick={importFolder}
                  >
                    Import Folder
                  </PrimaryButton>
                </div>
                {importResult && (
                  <div
                    style={{
                      maxHeight: "150px",
                      overflowY: "auto",
                      width: "500px",
                      fontSize: "0.85em",
                    }}
                  >
                    <div>Imported: {importResult.imported.length}</div>
                    {importResult.errors.length > 0 && (
                      <div>
                        <div>Errors:</div>
                        <ul>
                          {importResult.errors.map((error) => (
                            <li key={error.file}>
                              {error.file}: {error.reason}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                )}
              </div>
              <div id="uploadInputs" className="flexColumn">
                <div className="flexRowCenter">
                  <div className="dummyDiv" style={{ width: "32px" }} />
                  <TextField
                    id="standard-basic"
                    label="Image name"
                    variant="outlined"
                    error={!imageName}
                    sx={{ width: "350px", marginRight: "10px" }}
                    onChange={(event) =>
                      setImageName(event.currentTarget.value)
                    }
                  />
                  {imageName && (
                    <Tooltip
                      id="filePathTooltip"
                      title="Image name must match name as saved down in your file system (without the file path and file extension)"
                    >
                      <InfoOutlinedIcon
                        sx={{
                          color: "black",
                          width: "22px",
                          height: "22px",
                          display: "flex",
                          alignItems: "center",
                        }}
                      />
                    </Tooltip>
                  )}

                  {!imageName && (
                    <Tooltip
                      id="filePathTooltip"
                      title="Image name must match name as saved down in your file system (without the file path and file extension)"
                    >
                      <ErrorOutlineOutlinedIcon
                        color="error"
                        sx={{
                          width: "22px",
                          height: "22px",
                          display: "flex",
                          alignItems: "center",
                        }}
                      />
                    </Tooltip>
                  )}
                </div>

                <div style={{ marginTop: "10px" }}>
                  <div id="cnnRadioGroupWrapper" className="flexRowCenter">
                    <FormLabel
                      sx={{
                        width: "75px",
                        display: "flex",
                        justifyContent: "center",
                        marginRight: "15px",
                        color: "black",
                        fontWeight: 500,
                      }}
                    >
                      CNN:
                    </FormLabel>
                    <RadioGroup
                      row
                      value={cnn}
                      onChange={(e) => setCnn(parseInt(e.target.value))}
                    >
                      <FormControlLabel
                        value={1}
                        control={<Radio />}
                        label="One"
                        sx={{ width: "75px" }}
                      />
                      <FormControlLabel
                        value={2}
                        control={<Radio />}
                        label="Two"
                        disabled={true}
                        sx={{ width: "75px" }}
                      />
                    </RadioGroup>
                  </div>
                </div>

                <div style={{ marginTop: "10px" }}>
                  <div id="tileEdgeRadioGroupWrapper" className="flexRowCenter">
                    <FormLabel
                      sx={{
                        width: "75px",
                        display: "flex",
                        justifyContent: "center",
                        marginRight: "15px",
                        color: "black",
                        fontWeight: 500,
                      }}
                    >
                      Tile edge:
                    </FormLabel>
                    <RadioGroup
                      row
                      value={tileEdge}
                      onChange={(e) => setTileEdge(e.target.value)}
                    >
                      <FormControlLabel
                        value={126}
                        control={<Radio />}
                        label="126"
                        sx={{ width: "75px" }}
                      />
                      <FormControlLabel
                        value={252}
                        control={<Radio />}
                        label="252"
                        sx={{ width: "75px" }}
                      />
                    </RadioGroup>
                  </div>
                </div>
              </div>
              <div
                id="uploadButtons"
                className="buttonRow"
                style={{ minHeight: "100px", alignItems: "end" }}
              >
                <PrimaryButton
                  disabled={!imageName}
                  sx={{ minWidth: "200px" }}
                  onClick={() => uploadAnnotationsInputRef.current.click()}
                >
                  Upload annotations
                </PrimaryButton>
                <input
                  ref={uploadAnnotationsInputRef}
                  type="file"
                  accept={".csv,.tsv"}
                  id="annotations"
                  style={{ display: "none" }}
                  onChange={(e) => onFileUpload(e, true)}
                />
                <PrimaryButton
                  disabled={!imageName}
                  sx={{ minWidth: "200px" }}
                  onClick={() => uploadPredictionsInputRef.current.click()}
                >
                  Upload predictions
                </PrimaryButton>
                <input
                  ref={uploadPredictionsInputRef}
                  type="file"
                  accept={".csv,.tsv"}
                  id="predictions"
                  style={{ display: "none" }}
                  onChange={(e) => onFileUpload(e, false)}
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AmfUpload;
