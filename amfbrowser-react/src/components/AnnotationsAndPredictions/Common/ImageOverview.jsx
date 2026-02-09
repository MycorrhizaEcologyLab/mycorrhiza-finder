import CircularProgress from "@mui/material/CircularProgress";
import { useCallback, useEffect, useRef, useState } from "react";

import PredictionsApi from "../../../api/amfinderApi";
import { indexOfMax } from "../../Utils/utils";

const ImageOverview = ({
  selectedImage,
  numRows,
  numCols,
  gridData,
  colorMapping,
  selectedTile,
  setSelectedTile,
  maxOverviewSize,
  showPredictions,
  headerToValueMapCnn1,
  convertWithContext,
}) => {
  const [isLoadingImage, setIsLoadingImage] = useState(false);
  const [resizedImage, setResizedImage] = useState(null);
  const [error, setError] = useState(false);
  const [isMounted, setIsMounted] = useState(false);

  const canvasRef = useRef(null);

  const getResizedImage = useCallback(async () => {
    let image = resizedImage;
    if (image === null) {
      const resizedImageTmp = await PredictionsApi.getResizedImage(
        selectedImage,
        1500,
      );
      setResizedImage(resizedImageTmp);
      image = resizedImageTmp;
    }
    return image;
  }, [selectedImage, resizedImage]);

  useEffect(() => {
    if (!selectedImage || !canvasRef.current) return;

    const resizeImageAndDrawGrid = (image) => {
      const canvas = canvasRef.current;

      if (!canvas) return;

      const aspectRatio = image.width / image.height;

      let width, height;
      if (image.width > image.height) {
        width = image.width > maxOverviewSize ? maxOverviewSize : image.width;
        height = width / aspectRatio;
      } else {
        height =
          image.height > maxOverviewSize ? maxOverviewSize : image.height;
        width = height * aspectRatio;
      }

      canvas.width = width;
      canvas.height = height;

      const ctx = canvas.getContext("2d");
      ctx.drawImage(image, 0, 0, width, height);

      const rowHeight = height / numRows;
      const colWidth = width / numCols;

      for (let row = 0; row < numRows; row++) {
        for (let col = 0; col < numCols; col++) {
          const key = `${row}.${col}`;
          let value;
          if (showPredictions) {
            value = gridData.get(key)?.annotations;
            // Check if there is a contextual label if no annotation
            if (!value && convertWithContext) {
              value = gridData.get(key)?.contextualLabel;
            }
            // Finally if value does not have an annotation or contextual label,
            // get the max prediction
            if (!value) {
              const copyValue = [...gridData.get(key).predictions];
              const predictions = copyValue.map((s) => s[1]);
              // Get the index of the prediction with the highest value
              const maxIdx = indexOfMax(predictions);
              // Use this index to determine annotations
              const label = copyValue[maxIdx][0];
              value = headerToValueMapCnn1.get(label);
            }
          } else {
            value = gridData.get(key);
          }
          const color = colorMapping.get(value);

          if (color) {
            ctx.fillStyle = color;
            ctx.fillRect(col * colWidth, row * rowHeight, colWidth, rowHeight);
          }

          if (row === selectedTile?.row && col === selectedTile?.col) {
            ctx.strokeStyle = "red";
            ctx.lineWidth = 2;
            ctx.strokeRect(
              col * colWidth,
              row * rowHeight,
              colWidth,
              rowHeight,
            );
          }
        }
      }

      setIsLoadingImage(false);
      setIsMounted(true);
    };

    const processImage = async () => {
      try {
        let image = await getResizedImage();
        const url = URL.createObjectURL(image);
        const img = new Image();

        img.onload = () => {
          resizeImageAndDrawGrid(img);
          URL.revokeObjectURL(url);
        };

        img.onerror = (e) => {
          console.error("Error loading image:", e);
          setIsLoadingImage(false);
          setError(true);
        };

        img.src = url;
      } catch (e) {
        console.error("Failed to fetch and process the image:", e);
        setIsLoadingImage(false);
        setError(true);
      }
    };

    setIsLoadingImage(true);
    processImage();
  }, [
    selectedImage,
    numRows,
    numCols,
    gridData,
    colorMapping,
    selectedTile,
    getResizedImage,
    maxOverviewSize,
    showPredictions,
    headerToValueMapCnn1,
  ]);

  const handleCanvasClick = (event) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    const colWidth = rect.width / numCols;
    const rowHeight = rect.height / numRows;

    const col = Math.floor(x / colWidth);
    const row = Math.floor(y / rowHeight);

    if (setSelectedTile) {
      setSelectedTile({ row, col });
    }
  };

  return (
    <div className="fullHeight fullWidth flexRowCenter">
      <canvas
        ref={canvasRef}
        onClick={handleCanvasClick}
        style={{
          border: "1px solid black",
          cursor: "pointer",
          display: `${!isMounted || error ? "none" : "block"}`,
        }}
      />
      {isLoadingImage && !error && !isMounted && (
        <div id="circularProgressWrapper">
          <CircularProgress size={120} style={{ color: "rgb(39, 94, 55)" }} />
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
      {error && <p>Error in loading image overview.</p>}
    </div>
  );
};

export default ImageOverview;
