import Divider from "@mui/material/Divider";
import { useContext, useEffect, useState } from "react";
import { toast } from "react-toastify";
import PredictionsApi from "../../api/amfinderApi";
import { GlobalContextProvider } from "../../contexts/Contexts";
import Modal from "../Utils/Modal";
import PrimaryButton from "../Utils/PrimaryButton";
import ExistingTableContainer from "./ExistingTableContainer";
import ImageNameContainer from "./ImageNameContainer";
import "./styles/existing.css";

export default function ExistingContainer() {
  const { colonisationType } = useContext(GlobalContextProvider);

  const [imageNames, setImageNames] = useState([]);
  const [storedEntries, setStoredEntries] = useState(null);
  const [image, setImage] = useState(null);
  const [pendingDeletionId, setPendingDeletionId] = useState(null);
  const [selectedImageName, setSelectedImageName] = useState(null);

  useEffect(() => {
    const fetchImages = async () => {
      // TODO this gets all image names regardless of colonisation type - we could reduce this if we wanted
      const currentImages = await PredictionsApi.getImageNames();
      setImageNames(currentImages);
    };

    fetchImages();
  }, []);

  const fetchPredictionsAndAnnotations = async (name) => {
    const predictionsAndAnnotations = await PredictionsApi.checkEntriesForImage(
      name,
      colonisationType,
    );
    setStoredEntries(predictionsAndAnnotations);
    setImage(name);
  };

  const onDelete = async (id) => {
    await PredictionsApi.deleteImageReference(id);
    const storedEntriesCopy = { ...storedEntries };
    delete storedEntriesCopy[id];
    setStoredEntries(storedEntriesCopy);
  };

  const handleSetToEnabled = (e, id) => {
    const { checked } = e.target;

    if (checked) {
      PredictionsApi.setToEnabled(id);
      toast.success("Enabled sucessfully updated");
      const updatedEntries = Object.entries(storedEntries).reduce(
        (acc, [key, value]) => {
          // Spread the existing value and set the 'enabled' property to true for the current key if it matches 'id'
          acc[key] = { ...value, enabled: key === id };
          return acc;
        },
        {},
      );

      setStoredEntries(updatedEntries);
    }
  };

  const onDownload = async (id, cnn, type, timestamp) => {
    const csvFile = await PredictionsApi.downloadEntries(
      id,
      type,
      colonisationType,
    );
    let csvContent = "data:text/csv;charset=utf-8," + csvFile;

    // Download CSV file
    var encodedUri = encodeURI(csvContent);
    var link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute(
      "download",
      `${image}_${timestamp.toString()}_cnn_${cnn}_${type}.csv`,
    );
    document.body.appendChild(link);

    link.click();
  };

  return (
    <div id="amfToolContainer" className="fullHeight fullWidth flexRowCenter">
      <div
        className="fullHeight fullWidth flexColumn"
        style={{ overflowY: "auto" }}
      >
        <div
          className="fullHeight fullWidth"
          style={{
            minWidth: "1000px",
            minHeight: "590px",
          }}
        >
          <div className="fullHeight fullWidth">
            <div className="fullHeight fullWidth flexColumnCenter">
              <div
                className="imageNameContainer fullWidth"
                style={{
                  height: "35%",
                  minHeight: "270px",
                  display: "flex",
                  justifyContent: "center",
                  paddingTop: "10px",
                  paddingBottom: "10px",
                }}
              >
                <div style={{ display: "flex" }}>
                  <ImageNameContainer
                    fetchPredictionsAndAnnotations={
                      fetchPredictionsAndAnnotations
                    }
                    imageNames={imageNames}
                    colonisationType={colonisationType}
                    selectedImageName={selectedImageName}
                    setSelectedImageName={setSelectedImageName}
                  />
                </div>
              </div>
              <Divider sx={{ width: "100%" }} variant="fullWidth" />
              <div
                id="existingTableContainer"
                className="fullWidth flexColumn"
                style={{
                  height: "65%",
                  minHeight: "320px",
                  padding: "20px 0px",
                  justifyContent: "start",
                  alignItems: "center",
                  minWidth: "1000px",
                }}
              >
                {storedEntries && (
                  <ExistingTableContainer
                    data={storedEntries}
                    onDownload={onDownload}
                    handleSetToEnabled={handleSetToEnabled}
                    setPendingDeletionId={setPendingDeletionId}
                    colonisationType={colonisationType}
                    imageName={selectedImageName}
                  />
                )}
                {!storedEntries && <div>Please select image</div>}
              </div>
            </div>
          </div>
        </div>
      </div>
      {/* Modal that appears when pressing the delete button */}
      <Modal
        isOpen={pendingDeletionId != null}
        overrideStyles={{ width: "800px", height: "200px" }}
        onClose={() => setPendingDeletionId(null)}
      >
        <div className="save-questions-container">
          <div className="save-questions-row">
            Are you sure you want to delete?
          </div>
          <div className="save-questions-row">
            <PrimaryButton
              sx={{ width: "200px", margin: "5px" }}
              onClick={() => {
                onDelete(pendingDeletionId);
                toast.success("Analysis successfully deleted");
                setPendingDeletionId(null);
              }}
            >
              Confirm
            </PrimaryButton>
            <PrimaryButton
              sx={{ width: "200px", margin: "5px" }}
              onClick={() => setPendingDeletionId(null)}
            >
              Cancel
            </PrimaryButton>
          </div>
        </div>
      </Modal>
    </div>
  );
}
