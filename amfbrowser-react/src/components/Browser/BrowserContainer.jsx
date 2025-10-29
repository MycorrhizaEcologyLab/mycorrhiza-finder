import { useState, useEffect, useRef, useContext } from "react";
import { useNavigate } from "react-router-dom";
import { isEmpty } from "lodash";
import CircularProgress from "@mui/material/CircularProgress";
import ImageSelectionHeader from "./ImageSelectionHeader";
import PredictionAndAnnotationSelection from "./PredictionandAnnotationSelection";
import { removeFileExtension } from "../Utils/utils";
import { GlobalContextProvider } from "../../contexts/Contexts";
import PredictionsApi from "../../api/amfinderApi";
import "./styles/browserStyles.css";

const BrowserContainer = () => {
  const {
    colonisationType,
    setTileEdge,
    selectedImage,
    setSelectedImage,
    selectedImageName,
    setSelectedImageName,
  } = useContext(GlobalContextProvider);

  const navigate = useNavigate();

  const [isLoadingPredsAndAnnots, setIsLoadingPredsAndAnnots] = useState(false);
  const [hasLoadedPredsAndAnnots, setHasLoadedPredsAndAnnots] = useState(false);
  const [existingPredsAndAnnots, setExistingPredsAndAnnots] = useState(null);

  const imageSelectorInputRef = useRef(null);

  useEffect(() => {
    setHasLoadedPredsAndAnnots(false);
    setExistingPredsAndAnnots(null);
    selectedImage && loadAnnotationsAndPredictions(selectedImage);
  }, [colonisationType]);

  const loadAnnotationsAndPredictions = async (image) => {
    const tmpImageName = removeFileExtension(image.name);

    setHasLoadedPredsAndAnnots(false);
    setIsLoadingPredsAndAnnots(true);
    const predictionsAndAnnotations = await PredictionsApi.checkEntriesForImage(
      tmpImageName,
      colonisationType,
    );
    setIsLoadingPredsAndAnnots(false);

    // TODO: Differentiate between successful retrieval of empty object and error during retrieval
    // (The API call should probably be in a try-catch statement)
    setHasLoadedPredsAndAnnots(true);

    // If no preds/annots found, then go straight to annotations screen
    if (predictionsAndAnnotations && !isEmpty(predictionsAndAnnotations)) {
      setExistingPredsAndAnnots(predictionsAndAnnotations);
    } else {
      setExistingPredsAndAnnots(null);
    }
  };

  const navigateToAnnotations = (id, tileEdge) => {
    setTileEdge(tileEdge);

    if (id) {
      let path = `/browser/${selectedImageName}/${colonisationType}/annotations/${id}`;
      navigate(path, { state: { tileEdgeNavigated: tileEdge } }); // Send across a navigated tile edge
    } else {
      let path = `/browser/${selectedImageName}/${colonisationType}/annotations`;
      navigate(path);
    }
  };

  const navigateToPredictions = (id, tileEdge) => {
    if (id == null) return; // This should never happen, but including to be safe
    setTileEdge(tileEdge);

    // Perform the navigation
    navigate(
      `/browser/${selectedImageName}/${colonisationType}/predictions/${id}`,
      { state: { tileEdgeNavigated: tileEdge } },
    );
  };

  return (
    <div id="browserContainer" className="fullWidth fullHeight">
      <ImageSelectionHeader
        inputRef={imageSelectorInputRef}
        isSelectButtonDisabled={false}
        selectedImageName={selectedImageName}
        setSelectedImage={setSelectedImage}
        setSelectedImageName={setSelectedImageName}
        loadAnnotationsAndPredictions={loadAnnotationsAndPredictions}
      />

      <div
        id="browserMainWindow"
        className="fullWidth flexRowCenter"
        style={{
          height: "calc(100% - 60px)",
          overflowY: "auto",
        }}
      >
        <div style={{ height: "100%", padding: "20px", paddingTop: "50px" }}>
          <div className="predictionsAnnotationsWrapper">
            {isLoadingPredsAndAnnots && (
              <div id="circularProgressWrapper">
                <CircularProgress
                  size={120}
                  style={{ color: "rgb(39, 94, 55)" }}
                />
                <p
                  style={{
                    display: "flex",
                    justifyContent: "center",
                    color: "rgb(39, 94, 55)",
                    fontWeight: 600,
                    fontSize: "14px",
                  }}
                >
                  Loading
                </p>
              </div>
            )}
            {hasLoadedPredsAndAnnots && (
              <PredictionAndAnnotationSelection
                existingPredsAndAnnots={existingPredsAndAnnots}
                navigateToAnnotations={navigateToAnnotations}
                navigateToPredictions={navigateToPredictions}
              />
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default BrowserContainer;
