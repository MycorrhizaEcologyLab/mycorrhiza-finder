import { isEmpty } from "lodash";
import { useEffect, useState } from "react";
import PrimaryButton from "../Utils/PrimaryButton";
import PredictionAndAnnotationTable from "./PredictionAndAnnotationTable";

const PredictionAndAnnotationSelection = ({
  existingPredsAndAnnots,
  navigateToAnnotations,
  navigateToPredictions,
}) => {
  const [selectedKey, setSelectedKey] = useState(null);
  const [selectedTileEdge, setSelectedTileEdge] = useState(null);
  const [hasExistingAnnotations, setHasExistingAnnotations] = useState(false);
  const [hasExistingPredictions, setHasExistingPredictions] = useState(false);

  const updateSelectedRow = (key, tileEdge) => {
    setSelectedKey(key);
    setSelectedTileEdge(tileEdge);

    // Update flags for disabling existing annotations and predictions buttons
    if (key && existingPredsAndAnnots) {
      let selectedRow = existingPredsAndAnnots[key];
      if (selectedRow) {
        let hasExistingAnnotations = selectedRow.cnn1_annotations_exist;
        let hasExistingPredictions = selectedRow.cnn1_predictions_exist;

        setHasExistingAnnotations(hasExistingAnnotations);
        setHasExistingPredictions(hasExistingPredictions);
      } else {
        setHasExistingAnnotations(false);
        setHasExistingPredictions(false);
      }
    }
  };

  // Remove selected timestamp when the image changes
  useEffect(() => {
    updateSelectedRow(null, null);
  }, [existingPredsAndAnnots]);

  return (
    <div className="flexColumnCenter predictionsAnnotationsWrapper">
      {!isEmpty(existingPredsAndAnnots) && (
        <PredictionAndAnnotationTable
          data={existingPredsAndAnnots}
          selectedKey={selectedKey}
          updateSelectedRow={updateSelectedRow}
        />
      )}
      {isEmpty(existingPredsAndAnnots) && (
        <div
          style={{ height: "150px", fontSize: "18px" }}
          className="flexColumnCenter"
        >
          No existing annotations or predictions for selected image
        </div>
      )}
      <div
        id="predictionAndAnnotationButtonWrapper"
        style={{ marginTop: "30px", marginBottom: "30px" }}
      >
        {/* Create annotations */}
        <PrimaryButton
          disabled={selectedKey != null}
          sx={{ height: "40px", width: "200px", marginRight: "10px" }}
          onClick={() => navigateToAnnotations(null, selectedTileEdge)}
        >
          Create annotations
        </PrimaryButton>

        {/* Edit annotations */}
        <PrimaryButton
          disabled={selectedKey == null || !hasExistingAnnotations}
          sx={{
            height: "40px",
            width: "200px",
            marginLeft: "5px",
            marginRight: "5px",
          }}
          onClick={() => navigateToAnnotations(selectedKey, selectedTileEdge)}
        >
          Edit annotations
        </PrimaryButton>

        {/* View predictions */}
        <PrimaryButton
          disabled={selectedKey == null || !hasExistingPredictions}
          sx={{ height: "40px", width: "200px", marginLeft: "10px" }}
          onClick={() => navigateToPredictions(selectedKey, selectedTileEdge)}
        >
          View predictions
        </PrimaryButton>
      </div>
    </div>
  );
};

export default PredictionAndAnnotationSelection;
