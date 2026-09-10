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
        className="buttonRow"
        style={{ marginTop: "30px", marginBottom: "30px" }}
      >
        {/* Create annotations */}
        <PrimaryButton
          disabled={selectedKey != null}
          sx={{ minWidth: "200px" }}
          onClick={() => navigateToAnnotations(null, selectedTileEdge)}
        >
          Create annotations
        </PrimaryButton>

        {/* Edit annotations */}
        <PrimaryButton
          disabled={selectedKey == null || !hasExistingAnnotations}
          sx={{ minWidth: "200px" }}
          onClick={() => navigateToAnnotations(selectedKey, selectedTileEdge)}
        >
          Edit annotations
        </PrimaryButton>

        {/* View predictions */}
        <PrimaryButton
          disabled={selectedKey == null || !hasExistingPredictions}
          sx={{ minWidth: "200px" }}
          onClick={() => navigateToPredictions(selectedKey, selectedTileEdge)}
        >
          View predictions
        </PrimaryButton>
      </div>
    </div>
  );
};

export default PredictionAndAnnotationSelection;
