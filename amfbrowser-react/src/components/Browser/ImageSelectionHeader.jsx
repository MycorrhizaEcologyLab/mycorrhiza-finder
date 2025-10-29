import ValidationIcon from "../Utils/ValidationIcon";
import PrimaryButton from "../Utils/PrimaryButton";
import { removeFileExtension } from "../Utils/utils";

const ImageSelectionHeader = ({
  inputRef,
  isSelectButtonDisabled,
  selectedImageName,
  setSelectedImage,
  setSelectedImageName,
  loadAnnotationsAndPredictions,
}) => {
  const handleImageChange = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setSelectedImageName(removeFileExtension(file.name));
    setSelectedImage(file);
    loadAnnotationsAndPredictions(file);
  };

  return (
    <>
      <div id="imageSelectionHeaderContainer">
        <div id="imageSelectionButtonWrapper">
          <PrimaryButton
            disabled={isSelectButtonDisabled}
            className="imageSelectionButton"
            onClick={() => inputRef.current.click()}
          >
            Select image
          </PrimaryButton>
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            id="fungal-image"
            style={{ display: "none" }}
            onChange={handleImageChange}
          />
        </div>
        <div
          id="imageSelectionNameWrapper"
          style={{
            color:
              selectedImageName == null
                ? "rgb(180, 180, 180)"
                : "rgb(0, 60, 80)",
          }}
        >
          {selectedImageName ?? "Please select image"}
        </div>
        <div id="imageSelectionStatusWrapper">
          <ValidationIcon status={selectedImageName ? "success" : "failure"} />
        </div>
      </div>
    </>
  );
};

export default ImageSelectionHeader;
