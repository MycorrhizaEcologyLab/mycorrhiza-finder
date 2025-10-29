import {
  useState,
  useEffect,
  useContext,
  useCallback,
  useMemo,
  useRef,
} from "react";
import { useLocation, useNavigate } from "react-router-dom";
import CircularProgress from "@mui/material/CircularProgress";
import JSZip from "jszip";
import { debounce } from "lodash";
import html2canvas from "html2canvas";
import { toast } from "react-toastify";
import ImageSelectionHeader from "../../Browser/ImageSelectionHeader";
import AnnotationsAndPredictionsWindow from "./AnnotationsAndPredictionsWindow";
import ImageOverview from "./ImageOverview";
import { GlobalContextProvider } from "../../../contexts/Contexts";
import PredictionsApi from "../../../api/amfinderApi";
import PrimaryButton from "../../Utils/PrimaryButton";
import { calcStdDev, indexOfMax } from "../../Utils/utils";
import Modal from "../../Utils/Modal";
import {
  getHeaderToValueMapCnn1,
  headerToValueMapCnn2,
  getColorMappingCnn1,
  getColorMappingCnn1Transparent,
  colorMappingCnn2,
  getHeaderMapCnn1,
  headerMapCnn2,
  getValueMapCnn1,
  valueMapCnn2,
  getKeyBindingsCnn1,
  headerToKeyBindingsMapCnn2,
} from "../../../config/AnnotationsAndPredictionsConfig";
import "../styles/AnnotationsAndPredictionsStyles.css";

const getImageReferenceIdFromPathname = (pathname) => {
  let imageIdCandidate = pathname.split("/").pop();
  return !isNaN(imageIdCandidate) ? imageIdCandidate : null;
};

const AnnotationsAndPredictionsContainer = () => {
  const {
    colonisationType,
    tileEdge,
    setTileEdge,
    selectedImage,
    selectedImageName,
    settings,
  } = useContext(GlobalContextProvider);

  const navigate = useNavigate();

  const { pathname } = useLocation();
  const imageReferenceId = getImageReferenceIdFromPathname(pathname);

  const isNewAnnotation =
    pathname.includes("annotations") && imageReferenceId == null;

  const showAnnotations = pathname.includes("annotations");
  const showPredictions = pathname.includes("predictions");

  /////////////////////////////////////////////////////////////////////
  // header text + styling colours taken from config
  const headerToValueMapCnn1 = getHeaderToValueMapCnn1(colonisationType);
  const colorMappingCnn1 = getColorMappingCnn1(colonisationType);
  const colorMappingCnn1Transparent =
    getColorMappingCnn1Transparent(colonisationType);

  const headerMapCnn1 = getHeaderMapCnn1(colonisationType);
  const valueMapCnn1 = getValueMapCnn1(colonisationType);
  const keyBindingsCnn1 = getKeyBindingsCnn1(colonisationType);

  /////////////////////////////////////////////////////////////////////
  // useState

  // State impacted by the automated resizing of the components with differing window sizes
  const [gridSize, setGridSize] = useState(450);
  const [gridFontSize, setGridFontSize] = useState(1.2);
  const [iconSize, setIconSize] = useState(85);
  const [iconFontSize, setIconFontSize] = useState(1.6);
  const [selectFontSize, setSelectFontSize] = useState(1.4);
  const [maxGridSize, setMaxGridSize] = useState("20");
  const [maxOverviewSize, setMaxOverviewSize] = useState(1000);
  const [contextualLabelSize, setContextualLabelSize] = useState(25);
  const [contextualLabelFontSize, setContextualLabelFontSize] = useState(10);

  // Flags for the loading of tiles and images
  const [loadingTiles, setLoadingTiles] = useState(false);
  const [loadingImages, setLoadingImages] = useState(false);
  const [areTilesLoaded, setAreTilesLoaded] = useState(false);
  const [areImagesLoaded, setAreImagesLoaded] = useState(false);

  // state from AnnotationsLeftSidebar
  const [mode, setMode] = useState("Select");
  const [level, setLevel] = useState("CNN 1");

  const [showImageOverview, setShowImageOverview] = useState(false); // To potentially move to AnnotationsAndPredictionsCentralPane

  const [areAnnotationsSaving, setAreAnnotationsSaving] = useState(false); // Flag for checking if annotations are saving (
  // Currently as top level as it might make sense to leverage on top level)

  const [selectedTile, setSelectedTile] = useState({ row: 0, col: 0 });
  const [topLeftTile, setTopLeftTile] = useState({ row: 0, col: 0 });
  const [focusedTile, setFocusedTile] = useState({ row: 0, col: 0 });

  const [cnn1Annotations, setCnn1Annotations] = useState(new Map());
  const [cnn2Annotations, setCnn2Annotations] = useState(new Map());
  const [cnn1Predictions, setCnn1Predictions] = useState(new Map());
  const [cnn2Predictions, setCnn2Predictions] = useState(new Map());

  const [selectIcons, setSelectIcons] = useState(true);
  const [selectedOverlay, setSelectedOverlay] = useState(null);

  const [currentQuestions, setCurrentQuestions] = useState([]);
  const [currentQuestionIdx, setCurrentQuestionIdx] = useState(-1);
  const [questionMarkWarning, setQuestionMarkWarning] = useState(false);
  const [questionCommentOpen, setQuestionCommentOpen] = useState(false);
  const [questionMarkComments, setQuestionMarkComments] = useState(new Map());

  const [numRows, setNumRows] = useState(0);
  const [numCols, setNumCols] = useState(0);
  const [totalTiles, setTotalTiles] = useState(0);

  const [dragActivated, setDragActivated] = useState(false);
  const [tileImages, setTileImages] = useState(new Map());

  const [validationMetrics, setValidationMetrics] = useState(null);
  const [validationMetricIdxMap, setValidationMetricIdxMap] = useState(
    new Map(),
  );
  const [currentValidationIdx, setCurrentValidationIdx] = useState(undefined);

  const [confidenceMetrics, setConfidenceMetrics] = useState(null);
  const [confidenceMetricIdxMap, setConfidenceMetricIdxMap] = useState(
    new Map(),
  );
  const [currentConfidenceIdx, setCurrentConfidenceIdx] = useState(undefined);

  const [headerToCountsMapCnn1, setHeaderToCountsMapCnn1] = useState(
    new Map(
      colonisationType === "am"
        ? [
            ["AM+", 0],
            ["N-", 0],
            ["X", 0],
            ["U", 0],
            ["D", 0],
            ["H", 0],
            ["?", 0],
          ]
        : [
            ["Bl+", 0],
            ["Br+", 0],
            ["T+", 0],
            ["N-", 0],
            ["X", 0],
            ["M", 0],
            ["U", 0],
            ["D", 0],
            ["HE+", 0],
            ["HD+", 0],
            ["?", 0],
          ],
    ),
  );

  const [convertWithContext, setConvertWithContext] = useState(true);
  const [disableConvertWithContext, setDisableConvertWithContext] =
    useState(false);

  /////////////////////////////////////////////////////////////////////
  // useRef (in part used to avoid recursive state updates)
  const toCaptureRef = useRef();
  const selectedTileRef = useRef(selectedTile);
  const topLeftTileRef = useRef(topLeftTile);
  const keyHandled = useRef(false);

  /////////////////////////////////////////////////////////////////////
  // useMemo Actions

  const cnn1AnnotationActions = useMemo(
    () => ({
      set: (key, value) =>
        setCnn1Annotations((prevMap) => {
          const nextMap = new Map(prevMap);
          nextMap.set(key, value);
          return nextMap;
        }),
      setBulk: (map) =>
        setCnn1Annotations(() => {
          const nextMap = new Map(map);
          return nextMap;
        }),
      remove: (key) =>
        setCnn1Annotations((prevMap) => {
          const nextMap = new Map(prevMap);
          nextMap.delete(key);
          return nextMap;
        }),
      clear: () => setCnn1Annotations(new Map()),
    }),
    [setCnn1Annotations],
  );

  const cnn1PredictionActions = useMemo(
    () => ({
      set: (key, value) =>
        setCnn1Predictions((prevMap) => {
          const nextMap = new Map(prevMap);
          nextMap.set(key, value);
          return nextMap;
        }),
      setBulk: (map) =>
        setCnn1Predictions(() => {
          const nextMap = new Map(map);
          return nextMap;
        }),
      remove: (key) =>
        setCnn1Predictions((prevMap) => {
          const nextMap = new Map(prevMap);
          nextMap.delete(key);
          return nextMap;
        }),
      clear: () => setCnn1Predictions(new Map()),
    }),
    [setCnn1Predictions],
  );

  const checkAndSetCnn2Annotations = (prevMap, key, values) => {
    const existingValue = prevMap.get(key) || [];
    if (Array.isArray(existingValue)) {
      existingValue.push(...values);
    }
    return new Map(prevMap).set(key, existingValue);
  };

  const cnn2AnnotationActions = useMemo(
    () => ({
      set: (key, value) =>
        setCnn2Annotations((prevMap) => {
          const existingValue = prevMap.get(key) || [];
          if (Array.isArray(existingValue)) {
            existingValue.push(value);
          }
          return new Map(prevMap).set(key, existingValue);
        }),
      setBulk: (map) =>
        setCnn2Annotations(() => {
          const nextMap = new Map(map);
          return nextMap;
        }),
      setValues: (key, value) =>
        setCnn2Annotations((prevMap) => {
          return checkAndSetCnn2Annotations(prevMap, key, value);
        }),
      remove: (key) =>
        setCnn2Annotations((prevMap) => {
          const nextMap = new Map(prevMap);
          nextMap.delete(key);
          return nextMap;
        }),

      removeValue: (key, value) =>
        setCnn2Annotations((prevMap) => {
          const existingValues = prevMap.get(key) || [];
          if (existingValues) {
            const idx = existingValues.indexOf(value);
            if (idx !== -1) {
              existingValues.splice(idx, 1);
            }
            return new Map(prevMap).set(key, existingValues);
          }
          return prevMap;
        }),

      clear: () => setCnn2Annotations(new Map()),
    }),
    [setCnn2Annotations],
  );

  const cnn2PredictionActions = useMemo(
    () => ({
      set: (key, value) =>
        setCnn2Predictions((prevMap) => {
          const existingValue = prevMap.get(key) || [];
          if (Array.isArray(existingValue)) {
            existingValue.push(value);
          }
          return new Map(prevMap).set(key, existingValue);
        }),
      setBulk: (map) =>
        setCnn2Predictions(() => {
          const nextMap = new Map(map);
          return nextMap;
        }),
      remove: (key) =>
        setCnn2Predictions((prevMap) => {
          const nextMap = new Map(prevMap);
          nextMap.delete(key);
          return nextMap;
        }),

      removeValue: (key, value) =>
        setCnn2Predictions((prevMap) => {
          const existingValues = prevMap.get(key) || [];
          if (existingValues) {
            const idx = existingValues.indexOf(value);
            if (idx !== -1) {
              existingValues.splice(idx, 1);
            }
            return new Map(prevMap).set(key, existingValues);
          }
          return prevMap;
        }),

      clear: () => setCnn2Predictions(new Map()),
    }),
    [setCnn2Predictions],
  );

  const tileImageActions = useMemo(
    () => ({
      set: (key, value) =>
        setTileImages((prevMap) => {
          const nextMap = new Map(prevMap);
          nextMap.set(key, value);
          return nextMap;
        }),
      setBulk: (map) =>
        setTileImages(() => {
          const nextMap = new Map(map);
          return nextMap;
        }),
      setMany: (map) =>
        setTileImages((prevMap) => {
          const nextMap = new Map([...prevMap, ...map]);
          return nextMap;
        }),
      remove: (key) =>
        setTileImages((prevMap) => {
          const nextMap = new Map(prevMap);
          nextMap.delete(key);
          return nextMap;
        }),
      clear: () => setTileImages(new Map()),
    }),
    [setTileImages],
  );

  const questionMarkCommentActions = useMemo(
    () => ({
      set: (key, value) =>
        setQuestionMarkComments((prevMap) => {
          const nextMap = new Map(prevMap);
          nextMap.set(key, value);
          return nextMap;
        }),
      remove: (key) =>
        setQuestionMarkComments((prevMap) => {
          const nextMap = new Map(prevMap);
          nextMap.delete(key);
          return nextMap;
        }),
    }),
    [],
  );

  /////////////////////////////////////////////////////////////////////
  // useCallback
  const loadImageTiles = useCallback(
    async (tileEdge, image) => {
      const getZipFiles = async (data) => {
        // Unzip blobs into images
        const zip = await JSZip.loadAsync(data);
        const zipContent = await Promise.all(
          Object.keys(zip.files).map(async (relativePath) => {
            const file = zip.files[relativePath];
            const content = await file.async("blob");
            return { file: relativePath, content };
          }),
        );

        let outputImages = new Map();
        for (const value of zipContent) {
          const url = URL.createObjectURL(value.content);
          outputImages.set(value.file, url);
        }

        tileImageActions.setMany(outputImages);
      };

      const fetchImageTiles = async (numRows, numCols) => {
        setLoadingImages(true);

        // Set black border
        for (let i = 0; i <= numRows; i++) {
          tileImageActions.set(`${i}.${numCols}`, "border");
        }

        for (let j = 0; j <= numCols; j++) {
          tileImageActions.set(`${numRows}.${j}`, "border");
        }

        // Fetch image tiles in batches of batchSize
        const totalImageTiles = numRows * numCols;
        const batchSize = 2000;

        const promises = [];

        for (let i = 0; i < totalImageTiles; i = i + batchSize) {
          const promise = PredictionsApi.getImageTile(i, batchSize).then(
            (response) => getZipFiles(response.blob()),
          );
          promises.push(promise);
        }

        await Promise.all(promises);

        setLoadingImages(false);
        setAreImagesLoaded(true);
      };

      setLoadingTiles(true);

      try {
        await PredictionsApi.setTileEdgeBackend({
          tileEdge: tileEdge,
        });

        // First, send the image to the server side to be split into tiles
        const response = await PredictionsApi.tileImage(
          image !== undefined ? image : selectedImage,
        );

        const numRows = response["num_rows"];
        const numCols = response["num_cols"];

        setNumRows(numRows);
        setNumCols(numCols);

        // Then, fetch the image tiles back as zips
        fetchImageTiles(numRows, numCols);

        setLoadingTiles(false);
        setAreTilesLoaded(true);

        return [numRows, numCols];
      } catch (error) {
        //console.error("Error:", error);
        toast.error("Error loading image tiles");
        //setError(true);
        setLoadingTiles(false);
      }
    },
    [selectedImage, tileImageActions],
  );

  const removeQuestionMark = useCallback(
    (currentKey) => {
      cnn1AnnotationActions.remove(currentKey);
      const idx = currentQuestions.indexOf(currentKey);

      if (idx > -1) {
        // If is in list, remove and reset question idx
        currentQuestions.splice(idx, 1);
        setCurrentQuestions(currentQuestions.sort());
        if (currentQuestionIdx >= currentQuestions.length) {
          setCurrentQuestionIdx(currentQuestionIdx - 1);
        } else if (currentQuestions.length === 0) {
          setCurrentQuestionIdx(-1);
        }
      }

      // If question mark comment exists, then delete
      if (questionMarkComments.has(currentKey)) {
        questionMarkCommentActions.remove(currentKey);
      }
    },
    [
      cnn1AnnotationActions,
      currentQuestions,
      currentQuestionIdx,
      questionMarkComments,
      questionMarkCommentActions,
    ],
  );

  const addQuestionMark = useCallback(
    (currentKey) => {
      cnn1AnnotationActions.set(currentKey, "?");
      const newQuestions = currentQuestions;
      let idx = newQuestions.indexOf(currentKey);

      // If the current key is not already in the list, add it
      if (idx === -1) {
        newQuestions.push(currentKey);

        // Sort questions and update state
        let finalQuestions = newQuestions.sort();
        setCurrentQuestions(finalQuestions);
        setCurrentQuestionIdx(finalQuestions.indexOf(currentKey));
      }
    },
    [cnn1AnnotationActions, currentQuestions],
  );

  const setQuestionTile = useCallback(
    (currentKey) => {
      if (cnn1Annotations.get(currentKey) === "?" && !dragActivated) {
        removeQuestionMark(currentKey);
      } else {
        addQuestionMark(currentKey);
      }
    },
    [addQuestionMark, removeQuestionMark, cnn1Annotations, dragActivated],
  );

  const fillBackgroundTiles = useCallback(() => {
    const tmpMap = new Map(cnn1Annotations);
    for (let i = 0; i < numRows; i++) {
      for (let j = 0; j < numCols; j++) {
        let key = `${i}.${j}`;
        if (!tmpMap.has(key)) {
          tmpMap.set(key, "X");
        }
      }
    }
    setCnn1Annotations(tmpMap);
  }, [numCols, numRows, cnn1Annotations]);

  /////////////////////////////////////////////////////////////////////
  // UseEffect

  // On initial load, set the tile edge to 252 for am and 126 for erm, and fetch annotations / load image tiles
  useEffect(() => {
    // Go back to the browser page if loading fresh page
    // NB there are ways to cache file images (subject to time) https://stackoverflow.com/questions/19119040/how-do-i-save-and-restore-a-file-object-in-local-storage
    if (!selectedImage) {
      navigate("/browser");
    } else {
      // Set correct tile edge to the global context provider
      let correctTileEdge = tileEdge;
      if (!imageReferenceId) {
        if (colonisationType === "am") {
          correctTileEdge = 252;
        } else if (colonisationType === "erm") {
          correctTileEdge = 126;
        }
        setTileEdge(correctTileEdge);
      }

      // Fetch annotations / load image tiles using the correct tile edge
      if (imageReferenceId != null) {
        fetchAnnotations(imageReferenceId, correctTileEdge);
      } else {
        loadImageTiles(correctTileEdge);
      }
    }
  }, []);

  useEffect(() => {
    setTotalTiles(numCols * numRows);
  }, [numRows, numCols]);

  useEffect(() => {
    const countMapCopy = new Map(
      colonisationType === "am"
        ? [
            ["AM+", 0],
            ["N-", 0],
            ["X", 0],
            ["U", 0],
            ["D", 0],
            ["H", 0],
            ["?", 0],
          ]
        : [
            ["Bl+", 0],
            ["Br+", 0],
            ["T+", 0],
            ["N-", 0],
            ["X", 0],
            ["M", 0],
            ["U", 0],
            ["D", 0],
            ["HE+", 0],
            ["HD+", 0],
            ["?", 0],
          ],
    );

    cnn1Annotations.forEach((value, key) => {
      countMapCopy.set(value, countMapCopy.get(value) + 1);
    });

    setHeaderToCountsMapCnn1(countMapCopy);
  }, [cnn1Annotations, colonisationType]);

  // Key down handler for adding annotations
  useEffect(() => {
    const toggleValueCnn1 = (value, currentKey) => {
      if (showAnnotations) {
        if (cnn1Annotations.get(currentKey) === value) {
          cnn1AnnotationActions.remove(currentKey);
        } else {
          if (cnn1Annotations.get(currentKey) === "?") {
            removeQuestionMark(currentKey);
          }
          cnn1AnnotationActions.set(currentKey, value);
        }
      } else if (showPredictions) {
        let preds = cnn1Predictions.get(currentKey);
        if (preds) {
          if (preds?.annotations === value) {
            // Remove annotation
            preds.annotations = null;
            cnn1PredictionActions.set(currentKey, preds);
          } else {
            // Add annotations
            preds.annotations = value;
            cnn1PredictionActions.set(currentKey, preds);
          }
        } else {
          preds = { predictions: null, annotations: value };
          cnn1PredictionActions.set(currentKey, preds);
        }
      }
    };

    const keyDownHandler = (e) => {
      if (keyHandled.current && e.shiftKey === false && !dragActivated) return;
      keyHandled.current = true;

      const currentKey = dragActivated
        ? `${focusedTile.row}.${focusedTile.col}`
        : `${selectedTile.row}.${selectedTile.col}`;

      if (!questionCommentOpen) {
        if (level === "CNN 1") {
          if (colonisationType === "am") {
            switch (e.key.toUpperCase()) {
              case "A":
                e.preventDefault();
                dragActivated
                  ? cnn1AnnotationActions.set(currentKey, "AM+")
                  : toggleValueCnn1("AM+", currentKey);
                break;
              case "N":
                e.preventDefault();
                dragActivated
                  ? cnn1AnnotationActions.set(currentKey, "N-")
                  : toggleValueCnn1("N-", currentKey);
                break;
              case "H":
              case "X":
              case "D":
              case "U":
                e.preventDefault();
                dragActivated
                  ? cnn1AnnotationActions.set(currentKey, e.key.toUpperCase())
                  : toggleValueCnn1(e.key.toUpperCase(), currentKey);
                break;
              case "?":
              case "/":
                e.preventDefault();
                setQuestionTile(currentKey);
                break;
              case "DELETE":
                e.preventDefault();
                if (cnn1Annotations.get(currentKey) === "?") {
                  setQuestionTile(currentKey);
                } else {
                  cnn1AnnotationActions.remove(currentKey);
                }
                break;
              case "*":
                e.preventDefault();
                fillBackgroundTiles();
                break;
              default:
                break;
            }
          } else {
            switch (e.key.toUpperCase()) {
              case "H":
                e.preventDefault();
                dragActivated
                  ? cnn1AnnotationActions.set(currentKey, "HE+")
                  : toggleValueCnn1("HE+", currentKey);
                break;
              case "B":
                e.preventDefault();
                dragActivated
                  ? cnn1AnnotationActions.set(currentKey, "Bl+")
                  : toggleValueCnn1("Bl+", currentKey);
                break;
              case "R":
                e.preventDefault();
                dragActivated
                  ? cnn1AnnotationActions.set(currentKey, "Br+")
                  : toggleValueCnn1("Br+", currentKey);
                break;
              case "Y":
                e.preventDefault();
                dragActivated
                  ? cnn1AnnotationActions.set(currentKey, "HD+")
                  : toggleValueCnn1("HD+", currentKey);
                break;
              case "T":
                e.preventDefault();
                dragActivated
                  ? cnn1AnnotationActions.set(currentKey, "T+")
                  : toggleValueCnn1("T+", currentKey);
                break;
              case "N":
                e.preventDefault();
                dragActivated
                  ? cnn1AnnotationActions.set(currentKey, "N-")
                  : toggleValueCnn1("N-", currentKey);
                break;
              case "M":
              case "X":
              case "D":
              case "U":
                e.preventDefault();
                dragActivated
                  ? cnn1AnnotationActions.set(currentKey, e.key.toUpperCase())
                  : toggleValueCnn1(e.key.toUpperCase(), currentKey);
                break;
              case "?":
              case "/":
                e.preventDefault();
                setQuestionTile(currentKey);
                break;
              case "DELETE":
                e.preventDefault();
                if (cnn1Annotations.get(currentKey) === "?") {
                  setQuestionTile(currentKey);
                } else {
                  cnn1AnnotationActions.remove(currentKey);
                }
                break;
              case "*":
                e.preventDefault();
                fillBackgroundTiles();
                break;
              default:
                break;
            }
          }
        } else if (
          level !== "CNN 1" &&
          cnn1Annotations.get(currentKey) === "AM+"
        ) {
          let currentValueCnn2 = cnn2Annotations.get(currentKey);
          switch (e.key.toUpperCase()) {
            case "I":
              e.preventDefault();
              if (currentValueCnn2?.includes("IH")) {
                cnn2AnnotationActions.removeValue(currentKey, "IH");
              } else {
                cnn2AnnotationActions.set(currentKey, "IH");
              }
              break;
            case "A":
            case "V":
            case "H":
              e.preventDefault();
              if (currentValueCnn2?.includes(e.key.toUpperCase())) {
                cnn2AnnotationActions.removeValue(
                  currentKey,
                  e.key.toUpperCase(),
                );
              } else {
                cnn2AnnotationActions.set(currentKey, e.key.toUpperCase());
              }
              break;
            case "DELETE":
              e.preventDefault();
              cnn2AnnotationActions.remove(currentKey);
              break;
            default:
              break;
          }
        }
      }
    };

    const keyUpHandler = () => {
      keyHandled.current = false;
    };

    window.addEventListener("keydown", keyDownHandler);
    window.addEventListener("keyup", keyUpHandler);

    return () => {
      window.removeEventListener("keydown", keyDownHandler);
      window.removeEventListener("keyup", keyUpHandler);
    };
  }, [
    selectedTile,
    level,
    cnn1Annotations,
    cnn2Annotations,
    cnn1AnnotationActions,
    cnn2AnnotationActions,
    showAnnotations,
    showPredictions,
    cnn1Predictions,
    cnn1PredictionActions,
    setQuestionTile,
    fillBackgroundTiles,
    dragActivated,
    focusedTile,
    colonisationType,
    removeQuestionMark,
    questionCommentOpen,
  ]);

  // Auto resizing of grid based on window height
  useEffect(() => {
    // Change size of grid, icons and font based on width of window
    function handleResize() {
      if (
        window.matchMedia("(max-height: 700px)").matches ||
        window.matchMedia("(max-width: 1000px)").matches
      ) {
        setGridSize(300);
        setGridFontSize(0.7);
        setIconSize(55);
        setIconFontSize(1);
        setSelectFontSize(0.6);
        setMaxGridSize("16");
        setMaxOverviewSize(400);
        setContextualLabelFontSize(7);
        setContextualLabelSize(18);
      } else if (
        window.matchMedia("(max-height: 900px)").matches ||
        window.matchMedia("(max-width: 1150px)").matches
      ) {
        setGridSize(350);
        setGridFontSize(0.8);
        setIconSize(70);
        setIconFontSize(1.1);
        setSelectFontSize(0.7);
        setMaxGridSize("16");
        setMaxOverviewSize(600);
        setContextualLabelFontSize(8);
        setContextualLabelSize(20);
      } else if (
        window.matchMedia("(max-height: 1050px)").matches ||
        window.matchMedia("(max-width: 1400px)").matches
      ) {
        setGridSize(450);
        setGridFontSize(1);
        setIconSize(80);
        setIconFontSize(1.2);
        setSelectFontSize(0.8);
        setMaxGridSize("16");
        setMaxOverviewSize(750);
        setContextualLabelFontSize(10);
        setContextualLabelSize(22);
      } else if (
        window.matchMedia("(max-height: 1200px)").matches ||
        window.matchMedia("(max-width: 1550px)").matches
      ) {
        setGridSize(500);
        setGridFontSize(1.2);
        setIconSize(80);
        setIconFontSize(1.4);
        setSelectFontSize(1);
        setMaxGridSize("18");
        setMaxOverviewSize(900);
        setContextualLabelFontSize(11);
        setContextualLabelSize(26);
      } else {
        setGridSize(600);
        setGridFontSize(1.4);
        setIconSize(80);
        setIconFontSize(1.6);
        setSelectFontSize(1.2);
        setMaxGridSize("18");
        setMaxOverviewSize(1000);
        setContextualLabelFontSize(12);
        setContextualLabelSize(30);
      }
    }

    window.addEventListener("resize", handleResize);
    handleResize();
    return () => {
      window.removeEventListener("resize", handleResize);
    };
  }, []);

  // Update refs when value changes
  useEffect(() => {
    selectedTileRef.current = selectedTile;
  }, [selectedTile]);

  useEffect(() => {
    topLeftTileRef.current = topLeftTile;
  }, [topLeftTile]);

  useEffect(() => {
    const calculateTopLeftTile = (selected, current) => {
      let leftGridSize = 10;
      if (showPredictions) {
        leftGridSize = 3;
      }
      let newTopLeftRow = current.row;
      let newTopLeftCol = current.col;

      if (selected.row === 0) {
        // Set to the first row if selected is at the start.
        newTopLeftRow = 0;
      } else if (selected.row === numRows - 1) {
        // Set to n rows back if selected is at the last row.
        newTopLeftRow = numRows - leftGridSize;
      } else if (selected.row - newTopLeftRow >= leftGridSize) {
        // If moving greater than the grid size, then move that amount, minus the gridsize + 1
        newTopLeftRow += selected.row - newTopLeftRow - leftGridSize + 1;
      } else if (selected.row - newTopLeftRow <= -1) {
        // If moving the other way, then shift grid by the amount moved (no need to subtract grid size for this movement)
        newTopLeftRow += selected.row - newTopLeftRow;
      }

      // Sim comments for moving left & right
      if (selected.col === 0) {
        newTopLeftCol = 0;
      } else if (selected.col === numCols - 1) {
        newTopLeftCol = numCols - leftGridSize;
      } else if (selected.col - newTopLeftCol >= leftGridSize) {
        newTopLeftCol += selected.col - newTopLeftCol - leftGridSize + 1;
      } else if (selected.col - newTopLeftCol <= -1) {
        newTopLeftCol += selected.col - newTopLeftCol;
      }

      return {
        row: (newTopLeftRow + numRows) % numRows,
        col: (newTopLeftCol + numCols) % numCols,
      };
    };

    const newTile = calculateTopLeftTile(selectedTile, topLeftTileRef.current);

    if (
      (!Number.isNaN(newTile.row) &&
        newTile.row !== topLeftTileRef.current.row) ||
      (!Number.isNaN(newTile.col) && newTile.col !== topLeftTileRef.current.col)
    ) {
      setTopLeftTile(newTile);
    }
  }, [selectedTile, numRows, numCols, showPredictions]);

  // Set validation & confidence idx on initial load
  useEffect(() => {
    // If looking at predictions, also update validation idx
    if (showPredictions) {
      const newValidationIdx = validationMetricIdxMap.get(
        `${selectedTile.row}.${selectedTile.col}`,
      );
      setCurrentValidationIdx(newValidationIdx);

      const newConfidenceIdx = confidenceMetricIdxMap.get(
        `${selectedTile.row}.${selectedTile.col}`,
      );
      setCurrentConfidenceIdx(newConfidenceIdx);
    }
  }, []);

  // Key down handler for moving around the grid
  useEffect(() => {
    const setGridPosition = (prev, direction) => {
      const newRow = prev.row + direction.row;
      const newCol = prev.col + direction.col;

      const row = (newRow + numRows) % numRows;
      const col = (newCol + numCols) % numCols;

      return { row, col };
    };

    const handleKeyPress = (event) => {
      // Do not allow tile movement when question comment box is open
      if (!questionCommentOpen) {
        let direction = { row: 0, col: 0 };
        switch (event.key) {
          case "ArrowUp":
            event.preventDefault();
            direction.row = -1;
            break;
          case "ArrowDown":
            event.preventDefault();
            direction.row = 1;
            break;
          case "ArrowLeft":
            event.preventDefault();
            direction.col = -1;
            break;
          case "ArrowRight":
            event.preventDefault();
            direction.col = 1;
            break;
          default:
            return;
        }
        const newSelected = setGridPosition(selectedTileRef.current, direction);

        // If looking at predictions, also update validation idx
        if (showPredictions) {
          const newValidationIdx = validationMetricIdxMap.get(
            `${newSelected["row"]}.${newSelected["col"]}`,
          );
          setCurrentValidationIdx(newValidationIdx);

          const newConfidenceIdx = confidenceMetricIdxMap.get(
            `${newSelected["row"]}.${newSelected["col"]}`,
          );
          setCurrentConfidenceIdx(newConfidenceIdx);
        }

        setSelectedTile(newSelected);
      }
    };

    document.addEventListener("keydown", handleKeyPress);
    return () => document.removeEventListener("keydown", handleKeyPress);
  }, [
    numRows,
    numCols,
    validationMetricIdxMap,
    confidenceMetricIdxMap,
    questionCommentOpen,
    setCurrentValidationIdx,
    showPredictions,
  ]);

  /////////////////////////////////////////////////////////////////////
  // Utils

  const debouncedHandleScroll = useMemo(
    () =>
      // Debounce scroll to avoid maximum depth errors
      debounce((newTile) => {
        // If looking at predictions, also update validation idx
        if (showPredictions) {
          const newValidationIdx = validationMetricIdxMap.get(
            `${newTile["row"]}.${newTile["col"]}`,
          );
          setCurrentValidationIdx(newValidationIdx);

          const newConfidenceIdx = confidenceMetricIdxMap.get(
            `${newTile["row"]}.${newTile["col"]}`,
          );
          setCurrentConfidenceIdx(newConfidenceIdx);
        }

        setSelectedTile(newTile);
      }, 5),
    [
      setSelectedTile,
      showPredictions,
      validationMetricIdxMap,
      confidenceMetricIdxMap,
    ],
  );

  const handleMouseScroll = useCallback(
    (evt) => {
      // Disable scroll if dragging screen
      if (!dragActivated) {
        const newTile = { ...selectedTile };
        const isShift = evt.shiftKey;
        if (evt.deltaY > 0) {
          isShift
            ? (newTile.col = (newTile.col + 1) % numCols)
            : (newTile.row = (newTile.row + 1) % numRows);
        } else if (evt.deltaY < 0) {
          isShift
            ? (newTile.col = (newTile.col - 1 + numCols) % numCols)
            : (newTile.row = (newTile.row - 1 + numRows) % numRows);
        }

        debouncedHandleScroll(newTile);
      }
    },
    [numCols, numRows, debouncedHandleScroll, selectedTile, dragActivated],
  );

  const saveAnnotations = async (overrideQuestions) => {
    if (currentQuestions.length !== 0 && !overrideQuestions) {
      setQuestionMarkWarning(true);
    } else {
      setAreAnnotationsSaving(true);
      let cnn1Values = [];
      cnn1Annotations.forEach((value, key) => {
        let populated_index;
        let out;
        if (colonisationType === "am") {
          // Save CNN1 annotations into a format accepted by DB - AM
          out = new Array(9).fill(0);
          const splitKey = key.split(".");
          // Add rows and cols
          out[0] = parseInt(splitKey[0]);
          out[1] = parseInt(splitKey[1]);
          populated_index = valueMapCnn1.indexOf(value);
          if (populated_index !== -1) {
            // Add to array, skipping row and col indices
            out[populated_index + 2] = 1;
          }
        } else {
          // Save CNN1 annotations into a format accepted by DB - ErM
          out = new Array(13).fill(0);
          const splitKey = key.split(".");
          // Add rows and cols
          out[0] = parseInt(splitKey[0]);
          out[1] = parseInt(splitKey[1]);
          populated_index = valueMapCnn1.indexOf(value);
          if (populated_index !== -1) {
            // Add to array, skipping row and col indices
            out[populated_index + 2] = 1;
          }
        }

        // If question mark then check for comment
        if (populated_index === valueMapCnn1.length - 1) {
          if (questionMarkComments.has(key)) {
            // Add question mark comment
            out.push(questionMarkComments.get(key));
          }
        }

        cnn1Values.push(out);
      });

      let cnn2Values = [];
      cnn2Annotations.forEach((value, key) => {
        //  Only save annotation if key is AM+
        if (cnn1Annotations.get(key) === "AM+") {
          const out = new Array(6).fill(0);
          const splitKey = key.split(".");
          out[0] = parseInt(splitKey[0]);
          out[1] = parseInt(splitKey[1]);
          value.forEach((val, idx) => {
            let populated_index = valueMapCnn2.indexOf(val);
            if (populated_index !== -1) {
              out[populated_index + 2] = 1;
            }
          });
          cnn2Values.push(out);
        }
      });

      let body = {
        imageReferenceId, // Derived from the pathname
        fileName: selectedImageName, // Taken from GlobalContextProvider
        cnnOneValues: cnn1Values,
        cnnTwoValues: cnn2Values,
        colonisationType: colonisationType,
        tileEdge: tileEdge,
        enabled: true,
      };

      // ImageId is returned if the annotations have been successfully saved
      let savedImageReferenceId = await PredictionsApi.saveAnnotations(body);

      if (savedImageReferenceId) {
        setQuestionMarkWarning(false);
        setAreAnnotationsSaving(false);
        // Navigate to the update path if it is a new annotation
        if (isNewAnnotation) {
          navigate(
            `/browser/${selectedImageName}/${colonisationType}/annotations/${savedImageReferenceId}`,
          );
        }

        toast.success("Annotations saved successfully!");
      } else {
        setQuestionMarkWarning(false);
        setAreAnnotationsSaving(false);
        toast.error("Failed to save annotations");
      }
    }
  };

  const setTileValueWithIcon = (icon) => {
    const currentKey =
      selectedTile.row?.toString() + "." + selectedTile.col?.toString();
    if (showPredictions) {
      let preds = cnn1Predictions.get(currentKey);
      if (preds) {
        if (preds.annotations === icon) {
          // Remove annotation
          preds.annotations = null;
          cnn1PredictionActions.set(currentKey, preds);
        } else {
          // Add annotations
          preds.annotations = icon;
          cnn1PredictionActions.set(currentKey, preds);
        }
      } else {
        preds = { predictions: null, annotations: icon };
        cnn1PredictionActions.set(currentKey, preds);
      }
    } else {
      if (!selectIcons) {
        if (selectedOverlay === icon) {
          setSelectedOverlay(null);
        } else {
          setSelectedOverlay(icon);
        }
      } else {
        if (level === "CNN 1") {
          let currentValue = cnn1Annotations.get(currentKey);
          if (icon === "?") {
            setQuestionTile(currentKey);
          } else {
            if (currentValue === "?") {
              removeQuestionMark(currentKey);
            }
            if (currentValue === icon) {
              // If value is already matching pressed icon, then remove
              cnn1AnnotationActions.remove(currentKey);
            } else {
              // Otherwise set
              cnn1AnnotationActions.set(currentKey, icon);
            }
          }
        } else {
          let currentValue = cnn2Annotations.get(currentKey);
          if (cnn1Annotations.get(currentKey) === "AM+") {
            if (currentValue?.includes(icon)) {
              // If values already include pressed icon, remove that value
              selectIcons
                ? cnn2AnnotationActions.removeValue(currentKey, icon)
                : setSelectedOverlay(null);
            } else {
              // Otherwise add to list
              selectIcons
                ? cnn2AnnotationActions.set(currentKey, icon)
                : setSelectedOverlay(icon);
            }
          }
        }
      }
    }
  };

  const onConvert = () => {
    let cnn1AnnotationsMap = new Map();
    cnn1Predictions.forEach((value, key) => {
      // In below flow, prioritise manual label, then context, then max
      // (unless convert with context is false, in which case manual label and then max)
      if (value?.annotations) {
        // If annotation exists, set directly to this
        cnn1AnnotationsMap.set(key, value?.annotations);
      } else if (
        convertWithContext &&
        value?.contextualLabel &&
        value?.predictions &&
        value?.predictions.length > 0
      ) {
        // We still only convert with context if the original prediction is above the threshold
        const annotation = value.contextualLabel;
        // Take prediction of highest value and convert this into annotation
        const copyValue = [...value.predictions];
        const predictions = copyValue.map((s) => s[1]);
        // Get the index of the prediction with the highest value
        const maxIdx = indexOfMax(predictions);
        // Only set annotation if prediction value is above threshold
        if (predictions[maxIdx] >= settings?.threshold) {
          cnn1AnnotationsMap.set(key, annotation);
        }
      } else if (value?.predictions && value?.predictions.length > 0) {
        // Take prediction of highest value and convert this into annotation
        const copyValue = [...value.predictions];
        const predictions = copyValue.map((s) => s[1]);
        // Get the index of the prediction with the highest value
        const maxIdx = indexOfMax(predictions);
        // Only set annotation if prediction value is above threshold
        if (predictions[maxIdx] >= settings?.threshold) {
          const label = copyValue[maxIdx][0];
          cnn1AnnotationsMap.set(key, headerToValueMapCnn1.get(label));
        }
      }
    });

    let cnn2AnnotationsMap = new Map();
    cnn2Predictions.forEach((value, key) => {
      if (value.length > 0) {
        const copyValue = [...value];
        copyValue.shift();
        const predictions = copyValue.map((s) => s[1]);
        const maxIdx = indexOfMax(predictions);
        // Only set annotation if prediction value is above threshold - by definition this can only happen for maximum one label
        if (predictions[maxIdx] >= settings?.threshold) {
          const label = value[maxIdx + 1][0];
          cnn2AnnotationsMap.set(key, headerToValueMapCnn1.get(label));
        }
      }
    });

    cnn1AnnotationActions.setBulk(cnn1AnnotationsMap);
    cnn2AnnotationActions.setBulk(cnn2AnnotationsMap);

    // Navigate to existing annotation page
    navigate(
      `/browser/${selectedImageName}/${colonisationType}/annotations/${imageReferenceId}`,
    );
  };

  const handleSelectedValidationIdxChange = (value) => {
    // When validation idx changes, update the selected tile and the current index
    let parsedValue = parseInt(value);
    if (parsedValue !== null && !isNaN(parsedValue)) {
      let splitKey = validationMetrics[parsedValue][0].split(".");
      const newRow = parseInt(splitKey[0]);
      const newCol = parseInt(splitKey[1]);
      setSelectedTile({
        row: newRow,
        col: newCol,
      });
      setCurrentValidationIdx(parsedValue);

      const newConfidenceIdx = confidenceMetricIdxMap.get(
        `${newRow}.${newCol}`,
      );
      setCurrentConfidenceIdx(newConfidenceIdx);
    } else {
      setCurrentValidationIdx(undefined);
    }
  };

  const handleSelectedConfidenceIdxChange = (value) => {
    // When confidence idx changes, update the selected tile and the current index
    let parsedValue = parseInt(value);
    if (parsedValue !== null && !isNaN(parsedValue)) {
      let splitKey = confidenceMetrics[parsedValue][0].split(".");
      const newRow = parseInt(splitKey[0]);
      const newCol = parseInt(splitKey[1]);
      setSelectedTile({
        row: newRow,
        col: newCol,
      });
      setCurrentConfidenceIdx(parsedValue);

      const newValidationIdx = validationMetricIdxMap.get(
        `${newRow}.${newCol}`,
      );
      setCurrentValidationIdx(newValidationIdx);
    } else {
      setCurrentConfidenceIdx(undefined);
    }
  };

  const handleModeChange = (changeEvent) => {
    let newMode = changeEvent.target.value;
    if (newMode === "Select") setSelectedOverlay(null);
    setSelectIcons(newMode === "Select");
    setMode(newMode);
  };

  const onQuestionClick = () => {
    const currentKey = `${selectedTile.row.toString()}.${selectedTile.col.toString()}`;
    if (level === "CNN 1") {
      if (!selectIcons) {
        if (selectedOverlay === "?") {
          setSelectedOverlay(null);
        } else {
          setSelectedOverlay("?");
        }
      } else {
        setQuestionTile(currentKey);
      }
    }
  };

  const onQuestionNext = () => {
    if (currentQuestions.length > 0) {
      if (currentQuestionIdx !== -1) {
        const questionKey = currentQuestions[currentQuestionIdx];
        if (currentQuestionIdx === currentQuestions.length - 1) {
          setCurrentQuestionIdx(0);
        } else {
          setCurrentQuestionIdx(currentQuestionIdx + 1);
        }
        let splitKey = questionKey?.split(".");
        setSelectedTile({
          row: parseInt(splitKey[0]),
          col: parseInt(splitKey[1]),
        });
      }
    }
  };

  const initialSetAnnotations = (annotations, cnn) => {
    if (annotations) {
      let imageId = Object.keys(annotations)[0];
      let annotationsMap = new Map();
      let values = annotations[imageId];
      values?.forEach((value) => {
        let key = `${value[0].toString()}.${value[1].toString()}`;
        // Ignore row and col
        const labels = [...value].splice(2);
        // Find which indices have value 1 and then set the grid values to these
        const indexes = labels.reduce((r, n, i) => {
          n === 1 && r.push(i);
          return r;
        }, []);
        if (indexes.length > 0) {
          if (cnn === 1) {
            let tileValue = headerToValueMapCnn1.get(headerMapCnn1[indexes[0]]);
            if (tileValue === "?") {
              setQuestionTile(key);
              // In this case, download question comment as well and set if non-empty
              // Check length is 10 for AM or 14 for ErM to make sure we are fetching question comment
              if (
                colonisationType === "am"
                  ? value.length === 10
                  : value.length === 14
              ) {
                const questionComment = value[value.length - 1];
                // Set if non-empty and non null
                if (questionComment !== null && questionComment !== "") {
                  questionMarkCommentActions.set(key, questionComment);
                }
              }
            }
            annotationsMap.set(key, tileValue);
          } else {
            let tileValues = [];
            indexes.forEach((val, idx) =>
              tileValues.push(headerToValueMapCnn2.get(headerMapCnn2[val])),
            );
            annotationsMap = checkAndSetCnn2Annotations(
              annotationsMap,
              key,
              tileValues,
            );
          }
        }
      });

      // Ensure only one state update
      cnn === 1
        ? cnn1AnnotationActions.setBulk(annotationsMap)
        : cnn2AnnotationActions.setBulk(annotationsMap);
    }
  };

  const initialSetPredictions = (predictions, cnn) => {
    if (predictions) {
      // There should be only one imageId returned
      let imageId = Object.keys(predictions)[0];
      let values = predictions[imageId];
      let predictionsMap = new Map();

      // Create map for storing validation metric (std dev)
      let validationMetricMap = new Map();
      let validationMetricIdxMap = new Map();

      // Create map for storing confidence metric (highest confidence value)
      let confidenceMetricMap = new Map();
      let confidenceMetricIdxMap = new Map();

      let contextualLabelCount = 0;

      values?.forEach((value) => {
        let key = `${value[0].toString()}.${value[1].toString()}`;
        // Ignore row and col
        const labels = [...value].splice(2);

        const contextualLabel = labels.pop();
        let convertedContextualLabel = null;
        if (contextualLabel) {
          convertedContextualLabel = headerToValueMapCnn1.get(contextualLabel);
          contextualLabelCount += 1;
        }

        // Calculate standard deviation as a validation metric
        const stdDev = calcStdDev(labels);
        validationMetricMap.set(key, stdDev);

        const highestValue = Math.max(...labels);
        confidenceMetricMap.set(key, highestValue);

        let output = [];
        if (cnn === 1) {
          // Directly add predictions for each class
          labels.forEach((value, idx) => {
            output.push([headerMapCnn1[idx], value]);
          });
          predictionsMap.set(key, {
            predictions: output,
            annotations: null,
            contextualLabel: convertedContextualLabel,
          });
        } else {
          labels.forEach((value, idx) => {
            output.push([headerMapCnn2[idx], value]);
          });
          cnn2PredictionActions.set(key, output);
        }
      });

      // Sort the keys by std dev and store in array
      let sortedValidationMetrics = Array.from(
        validationMetricMap.entries(),
      ).sort((a, b) => {
        return a[1] - b[1];
      });

      // Create a map of key to idx in sorted array
      for (var i = 0; i < sortedValidationMetrics.length; i++) {
        validationMetricIdxMap.set(sortedValidationMetrics[i][0], i);
      }

      // Sort the keys by confidence and store in array
      let sortedConfidenceMetrics = Array.from(
        confidenceMetricMap.entries(),
      ).sort((a, b) => {
        return a[1] - b[1];
      });

      // Create a map of key to idx in sorted array
      for (var j = 0; j < sortedConfidenceMetrics.length; j++) {
        confidenceMetricIdxMap.set(sortedConfidenceMetrics[j][0], j);
      }

      // Set the initial idx to 0.0 tile
      const initialValidationMetricIdx = validationMetricIdxMap.get("0.0");
      const initialConfidenceMetricIdx = confidenceMetricIdxMap.get("0.0");

      // Ensure only one state update
      if (cnn === 1) {
        setCurrentValidationIdx(initialValidationMetricIdx);
        setValidationMetricIdxMap(validationMetricIdxMap);
        setValidationMetrics(sortedValidationMetrics);

        setCurrentConfidenceIdx(initialConfidenceMetricIdx);
        setConfidenceMetricIdxMap(confidenceMetricIdxMap);
        setConfidenceMetrics(sortedConfidenceMetrics);

        if (contextualLabelCount === 0) {
          // If no contextual labels included, disable the toggle
          setConvertWithContext(false);
          setDisableConvertWithContext(true);
        }

        cnn1PredictionActions.setBulk(predictionsMap);
      }
    }
  };

  const fetchAnnotations = async (id, tileEdge, image) => {
    const annotationsCnn1 = await PredictionsApi.fetchAnnotationsById(
      id,
      "1",
      colonisationType,
    );
    let annotationsCnn2 = null;
    if (colonisationType === "am") {
      annotationsCnn2 = await PredictionsApi.fetchAnnotationsById(
        id,
        "2",
        colonisationType,
      );
    }
    const predictionsCnn1 = await PredictionsApi.fetchPredictionsById(
      id,
      "1",
      colonisationType,
    );

    await loadImageTiles(tileEdge, image);

    initialSetAnnotations(annotationsCnn1, 1);
    initialSetPredictions(predictionsCnn1, 1);
    initialSetAnnotations(annotationsCnn2, 2);
  };

  const captureScreenshot = () => {
    let canvasPromise = html2canvas(toCaptureRef.current, {
      useCORS: true, // in case you have images stored in your application
    });

    canvasPromise.then((canvas) => {
      var dataURL = canvas.toDataURL("image/png");
      // Create an image element from the data URL
      var img = new Image();
      img.src = dataURL;
      img.download = dataURL;

      const a = document.createElement("a");
      a.href = img.src;
      const now = new Date();
      const currentTime = now.toISOString();
      a.download = `mfbrowser-screenshot-${currentTime}.jpg`;
      a.click();
    });
  };

  return (
    <div
      id="annotationsAndPredictionsContainer"
      className="fullWidth fullHeight"
    >
      <ImageSelectionHeader
        isSelectButtonDisabled={true}
        selectedImageName={selectedImageName}
        mode={mode}
        setMode={setMode}
      />
      {(loadingTiles || loadingImages) && (
        <div
          className="fullWidth flexRowCenter"
          style={{ height: "calc(100% - 60px)" }}
        >
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
        </div>
      )}

      {areTilesLoaded && areImagesLoaded && (
        <>
          <AnnotationsAndPredictionsWindow
            colorMappingCnn1={colorMappingCnn1}
            colorMappingCnn2={colorMappingCnn2}
            headerToValueMapCnn1={headerToValueMapCnn1}
            headerToValueMapCnn2={headerToValueMapCnn2}
            keyBindingsCnn1={keyBindingsCnn1}
            keyBindingsCnn2={headerToKeyBindingsMapCnn2}
            selectedTile={selectedTile}
            setTileValueWithIcon={setTileValueWithIcon}
            cnn1Annotations={cnn1Annotations}
            cnn2Annotations={cnn2Annotations}
            cnn1Predictions={cnn1Predictions}
            cnn2Predictions={cnn2Predictions}
            iconSize={iconSize}
            selectIcons={selectIcons}
            setSelectedOverlay={setSelectedOverlay}
            handleModeChange={handleModeChange}
            onQuestionClick={onQuestionClick}
            onQuestionNext={onQuestionNext}
            selectFontSize={selectFontSize}
            iconFontSize={iconFontSize}
            headerToCountsMapCnn1={headerToCountsMapCnn1}
            //showAnnotations={showAnnotations}
            mode={mode}
            setMode={setMode}
            level={level}
            setLevel={setLevel}
            maxGridSize={maxGridSize}
            setSelectedTile={setSelectedTile}
            numRows={numRows}
            numCols={numCols}
            tileEdge={tileEdge}
            gridSize={gridSize}
            tileImages={tileImages}
            //showPredictions={showPredictions}
            validationMetricIdxMap={validationMetricIdxMap}
            currentValidationIdx={currentValidationIdx}
            setCurrentValidationIdx={setCurrentValidationIdx}
            handleMouseScroll={handleMouseScroll}
            toCaptureRef={toCaptureRef}
            // AnnotationsViewer additional props
            colorMappingCnn1Transparent={colorMappingCnn1Transparent}
            gridFontSize={gridFontSize}
            topLeftTile={topLeftTile}
            selectedOverlay={selectedOverlay}
            setDragActivated={setDragActivated}
            setFocusedTile={setFocusedTile}
            focusedTile={focusedTile}
            dragActivated={dragActivated}
            setQuestionCommentOpen={setQuestionCommentOpen}
            questionCommentOpen={questionCommentOpen}
            questionMarkComments={questionMarkComments}
            questionMarkCommentActions={questionMarkCommentActions}
            // Action button additional props
            saveAnnotations={saveAnnotations}
            areAnnotationsSaving={areAnnotationsSaving}
            captureScreenshot={captureScreenshot}
            setShowImageOverview={setShowImageOverview}
            // RightSidebar additional props - to potentially be moved within component
            totalTiles={totalTiles}
            onConvert={onConvert}
            handleSelectedValidationIdxChange={
              handleSelectedValidationIdxChange
            }
            validationMetrics={validationMetrics}
            conversionThreshold={settings?.threshold}
            confidenceMetricIdxMap={confidenceMetricIdxMap}
            currentConfidenceIdx={currentConfidenceIdx}
            setCurrentConfidenceIdx={setCurrentConfidenceIdx}
            handleSelectedConfidenceIdxChange={
              handleSelectedConfidenceIdxChange
            }
            confidenceMetrics={confidenceMetrics}
            convertWithContext={convertWithContext}
            disableConvertWithContext={disableConvertWithContext}
            setConvertWithContext={setConvertWithContext}
            contextualLabelFontSize={contextualLabelFontSize}
            contextualLabelSize={contextualLabelSize}
          />

          {/* Warning modal that appears when trying to save with question marks */}
          <Modal
            isOpen={questionMarkWarning}
            onClose={() => setQuestionMarkWarning(false)}
            overrideStyles={{ width: "800px", height: "200px" }}
          >
            <div className="save-questions-container">
              <div className="save-questions-row">
                You still have question marks - do you want to proceed with
                saving?
              </div>
              <div className="save-questions-row">
                <PrimaryButton
                  sx={{ width: "200px", margin: "5px" }}
                  onClick={() => saveAnnotations(true)}
                >
                  Save Annotations
                </PrimaryButton>
                <PrimaryButton
                  sx={{ width: "200px", margin: "5px" }}
                  onClick={() => setQuestionMarkWarning(false)}
                >
                  Back
                </PrimaryButton>
              </div>
            </div>
          </Modal>

          {/* Modal that appears when pressing the Image Overview button */}
          <Modal
            isOpen={showImageOverview}
            onClose={() => setShowImageOverview(false)}
          >
            <ImageOverview
              selectedImage={selectedImage}
              numRows={numRows}
              numCols={numCols}
              colorMapping={
                level === "CNN 1"
                  ? colorMappingCnn1Transparent
                  : colorMappingCnn2
              }
              gridData={showAnnotations ? cnn1Annotations : cnn1Predictions}
              selectedTile={selectedTile}
              setSelectedTile={setSelectedTile}
              maxOverviewSize={maxOverviewSize}
              showPredictions={showPredictions}
              headerToValueMapCnn1={headerToValueMapCnn1}
              convertWithContext={convertWithContext}
            />
          </Modal>
        </>
      )}
    </div>
  );
};

export default AnnotationsAndPredictionsContainer;
