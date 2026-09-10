import CircularProgress from "@mui/material/CircularProgress";
import html2canvas from "html2canvas";
import JSZip from "jszip";
import { debounce } from "lodash";
import {
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { toast } from "react-toastify";
import PredictionsApi from "../../../api/amfinderApi";
import {
  formatTags,
  getColorMappingCnn1,
  getColorMappingCnn1Transparent,
  getHeaderMapCnn1,
  getHeaderToValueMapCnn1,
  getKeyBindingsCnn1,
  getValueMapCnn1,
  parseTags,
} from "../../../config/AnnotationsAndPredictionsConfig";
import { GlobalContextProvider } from "../../../contexts/Contexts";
import ImageSelectionHeader from "../../Browser/ImageSelectionHeader";
import Modal from "../../Utils/Modal";
import PrimaryButton from "../../Utils/PrimaryButton";
import { calcStdDev, indexOfMax } from "../../Utils/utils";
import "../styles/AnnotationsAndPredictionsStyles.css";
import AnnotationsAndPredictionsWindow from "./AnnotationsAndPredictionsWindow";
import ImageOverview from "./ImageOverview";

const getImageReferenceIdFromPathname = (pathname) => {
  // The id segment is a numeric DB id in database mode, or a local CSV
  // filename in local (no-DB) mode. Its absence distinguishes a "new" annotation,
  // the route with no id segment ends in "annotations"/"predictions".
  let imageIdCandidate = pathname.split("/").pop();
  return imageIdCandidate === "annotations" || imageIdCandidate === "predictions"
    ? null
    : imageIdCandidate;
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
  const [tagBadgeSize, setTagBadgeSize] = useState(15);
  const [tagBadgeFontSize, setTagBadgeFontSize] = useState(9);
  const [contextualLabelFontSize, setContextualLabelFontSize] = useState(10);

  // Flags for the loading of tiles and images
  const [loadingTiles, setLoadingTiles] = useState(false);
  const [loadingImages, setLoadingImages] = useState(false);
  const [areTilesLoaded, setAreTilesLoaded] = useState(false);
  const [areImagesLoaded, setAreImagesLoaded] = useState(false);

  // state from AnnotationsLeftSidebar
  const [mode, setMode] = useState("Select");

  const [showImageOverview, setShowImageOverview] = useState(false); // To potentially move to AnnotationsAndPredictionsCentralPane

  const [areAnnotationsSaving, setAreAnnotationsSaving] = useState(false); // Flag for checking if annotations are saving (
  // Currently as top level as it might make sense to leverage on top level)

  const [selectedTile, setSelectedTile] = useState({ row: 0, col: 0 });
  const [topLeftTile, setTopLeftTile] = useState({ row: 0, col: 0 });
  const [focusedTile, setFocusedTile] = useState({ row: 0, col: 0 });

  const [cnn1Annotations, setCnn1Annotations] = useState(new Map());
  const [cnn1Predictions, setCnn1Predictions] = useState(new Map());

  const [selectIcons, setSelectIcons] = useState(true);
  const [selectedOverlay, setSelectedOverlay] = useState(null);

  const [currentQuestions, setCurrentQuestions] = useState([]);
  const [currentQuestionIdx, setCurrentQuestionIdx] = useState(-1);
  const [questionMarkWarning, setQuestionMarkWarning] = useState(false);
  const [questionCommentOpen, setQuestionCommentOpen] = useState(false);
  const [questionMarkComments, setQuestionMarkComments] = useState(new Map());

  // Per-tile user-defined sub-tags, keyed "row.col" like the maps above.
  const [tileTags, setTileTags] = useState(new Map());
  // Tag name -> colour, loaded from the backend palette. Tag *names* come from
  // the annotation data itself, so an empty palette only costs the colours.
  const [tagPalette, setTagPalette] = useState({});
  // Suppresses the annotation hotkeys while the tag name field has focus.
  const [tagInputOpen, setTagInputOpen] = useState(false);

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

  const [autoClassifyBackground, setAutoClassifyBackground] = useState(false);
  const [canUndoBackgroundFill, setCanUndoBackgroundFill] = useState(false);
  const [backgroundThreshold, setBackgroundThreshold] = useState(0.99);

  /////////////////////////////////////////////////////////////////////
  // useRef (in part used to avoid recursive state updates)
  const toCaptureRef = useRef();
  const selectedTileRef = useRef(selectedTile);
  const topLeftTileRef = useRef(topLeftTile);
  const keyHandled = useRef(false);
  const lastFillBackgroundKeysRef = useRef(null);
  const autoBackgroundKeysRef = useRef(null);
  const backgroundThresholdRef = useRef(backgroundThreshold);
  // Kept in sync on every render so debounced/async classification callbacks
  // (whose identities must stay stable - see applyAutoBackgroundClassification
  // below) always read the latest threshold instead of a stale closure.
  backgroundThresholdRef.current = backgroundThreshold;

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

  const tileTagActions = useMemo(
    () => ({
      set: (key, value) =>
        setTileTags((prevMap) => {
          const nextMap = new Map(prevMap);
          if (value && value.length > 0) {
            nextMap.set(key, value);
          } else {
            // Never keep an empty list - "has tags" is a .has() check.
            nextMap.delete(key);
          }
          return nextMap;
        }),
      setBulk: (map) => setTileTags(() => new Map(map)),
      toggle: (key, tag) =>
        setTileTags((prevMap) => {
          const nextMap = new Map(prevMap);
          const current = nextMap.get(key) ?? [];
          const next = current.includes(tag)
            ? current.filter((t) => t !== tag)
            : [...current, tag];
          if (next.length > 0) {
            nextMap.set(key, next);
          } else {
            nextMap.delete(key);
          }
          return nextMap;
        }),
      remove: (key) =>
        setTileTags((prevMap) => {
          const nextMap = new Map(prevMap);
          nextMap.delete(key);
          return nextMap;
        }),
      clear: () => setTileTags(new Map()),
    }),
    [],
  );

  // Every tag the user can pick from: those with a stored colour, plus any
  // found in the loaded annotations (so opening a file authored elsewhere
  // repopulates the list without needing the palette).
  const allTags = useMemo(() => {
    const names = new Set(Object.keys(tagPalette));
    tileTags.forEach((tags) => tags.forEach((tag) => names.add(tag)));
    return Array.from(names).sort((a, b) =>
      a.localeCompare(b, undefined, { sensitivity: "base" }),
    );
  }, [tagPalette, tileTags]);

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

  // Blindly fills every remaining/unlabelled tile as Background ("X"),
  // triggered by the "*" hotkey. Tracks exactly which keys it added so
  // undoFillBackgroundTiles can revert just this action.
  const fillBackgroundTiles = useCallback(() => {
    const tmpMap = new Map(cnn1Annotations);
    const addedKeys = [];
    for (let i = 0; i < numRows; i++) {
      for (let j = 0; j < numCols; j++) {
        let key = `${i}.${j}`;
        if (!tmpMap.has(key)) {
          tmpMap.set(key, "X");
          addedKeys.push(key);
        }
      }
    }
    setCnn1Annotations(tmpMap);
    lastFillBackgroundKeysRef.current = addedKeys;
    setCanUndoBackgroundFill(addedKeys.length > 0);
  }, [numCols, numRows, cnn1Annotations]);

  // Undoes the most recent fillBackgroundTiles() call ("Ctrl+*"), removing
  // only the tiles it added (and only if they haven't since been
  // relabelled by hand).
  const undoFillBackgroundTiles = useCallback(() => {
    const keysToRemove = lastFillBackgroundKeysRef.current;
    if (!keysToRemove || keysToRemove.length === 0) return;

    setCnn1Annotations((prevMap) => {
      const nextMap = new Map(prevMap);
      keysToRemove.forEach((key) => {
        if (nextMap.get(key) === "X") {
          nextMap.delete(key);
        }
      });
      return nextMap;
    });
    lastFillBackgroundKeysRef.current = null;
    setCanUndoBackgroundFill(false);
  }, []);

  // Single entry point for the "*" hotkey and its button: fills remaining
  // tiles as Background if there's nothing pending to undo, otherwise undoes
  // the last fill.
  const toggleBackgroundFill = useCallback(() => {
    if (canUndoBackgroundFill) {
      undoFillBackgroundTiles();
    } else {
      fillBackgroundTiles();
    }
  }, [canUndoBackgroundFill, fillBackgroundTiles, undoFillBackgroundTiles]);

  // Asks the backend to classify the currently loaded tiles as
  // background/root using the same mean-pixel-intensity heuristic as the
  // "Filter Background Tiles" training option, and fills in only the
  // detected background tiles that aren't already labelled. Reads the
  // threshold via a ref (rather than depending on `backgroundThreshold`
  // directly) so this callback's identity stays stable - keeping the
  // debounced reclassify below from being recreated (and racing) on every
  // threshold change.
  const applyAutoBackgroundClassification = useCallback(async () => {
    try {
      const backgroundKeys = await PredictionsApi.getBackgroundTiles(
        backgroundThresholdRef.current,
      );
      if (!backgroundKeys || backgroundKeys.length === 0) {
        autoBackgroundKeysRef.current = [];
        return;
      }

      const addedKeys = [];
      setCnn1Annotations((prevMap) => {
        const nextMap = new Map(prevMap);
        backgroundKeys.forEach((key) => {
          if (!nextMap.has(key)) {
            nextMap.set(key, "X");
            addedKeys.push(key);
          }
        });
        return nextMap;
      });
      autoBackgroundKeysRef.current = addedKeys;
    } catch (error) {
      toast.error("Error auto-classifying background tiles");
    }
  }, []);

  // Reverses applyAutoBackgroundClassification(), removing only the tiles
  // it added (and only if they haven't since been relabelled by hand).
  const undoAutoBackgroundClassification = useCallback(() => {
    const keysToRemove = autoBackgroundKeysRef.current;
    if (!keysToRemove || keysToRemove.length === 0) return;

    setCnn1Annotations((prevMap) => {
      const nextMap = new Map(prevMap);
      keysToRemove.forEach((key) => {
        if (nextMap.get(key) === "X") {
          nextMap.delete(key);
        }
      });
      return nextMap;
    });
    autoBackgroundKeysRef.current = null;
  }, []);

  // Re-runs auto-classification (undo previous + reapply) after the
  // threshold has settled for a moment, so dragging/scrubbing the value
  // doesn't spam the backend on every intermediate value. Only ever
  // triggered explicitly (from a threshold change while checked, below) -
  // NOT from an effect watching autoClassifyBackground, since that raced
  // with handleAutoClassifyBackgroundToggle's own direct apply/undo and
  // could re-add tiles shortly after the checkbox was unticked.
  const debouncedReclassify = useMemo(
    () =>
      debounce(() => {
        undoAutoBackgroundClassification();
        applyAutoBackgroundClassification();
      }, 400),
    [undoAutoBackgroundClassification, applyAutoBackgroundClassification],
  );

  const handleAutoClassifyBackgroundToggle = useCallback(
    (event) => {
      const checked = event.target.checked;
      setAutoClassifyBackground(checked);
      if (checked) {
        applyAutoBackgroundClassification();
      } else {
        // Cancel any reclassify still pending from a recent threshold drag -
        // otherwise it could fire after this undo and re-add the tiles.
        debouncedReclassify.cancel();
        undoAutoBackgroundClassification();
      }
    },
    [
      applyAutoBackgroundClassification,
      undoAutoBackgroundClassification,
      debouncedReclassify,
    ],
  );

  const handleBackgroundThresholdChange = useCallback(
    (value) => {
      setBackgroundThreshold(value);
      if (autoClassifyBackground) {
        debouncedReclassify();
      }
    },
    [autoClassifyBackground, debouncedReclassify],
  );

  // Guarantees the latest threshold is applied the moment the user leaves
  // the control, instead of waiting out the trailing debounce delay.
  const flushBackgroundThresholdChange = useCallback(() => {
    debouncedReclassify.flush();
  }, [debouncedReclassify]);

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

  // Re-run auto-classification whenever a new image's tiles finish loading,
  // provided the "Auto-classify background tiles" checkbox is on.
  useEffect(() => {
    if (areTilesLoaded && autoClassifyBackground) {
      applyAutoBackgroundClassification();
    }
  }, [areTilesLoaded]);

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

      // Suppressed while either per-tile text field has focus, or typing
      // would relabel the selected tile instead of entering text.
      if (!questionCommentOpen && !tagInputOpen) {
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
              toggleBackgroundFill();
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
              toggleBackgroundFill();
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
    cnn1Annotations,
    cnn1AnnotationActions,
    showAnnotations,
    showPredictions,
    cnn1Predictions,
    cnn1PredictionActions,
    setQuestionTile,
    toggleBackgroundFill,
    dragActivated,
    focusedTile,
    colonisationType,
    removeQuestionMark,
    questionCommentOpen,
    tagInputOpen,
  ]);

  // Load the saved tag colours once. Failure is non-fatal - tags still work,
  // they just fall back to the default colour cycle.
  useEffect(() => {
    let cancelled = false;
    PredictionsApi.getTagPalette().then((palette) => {
      if (!cancelled && palette) {
        setTagPalette(palette);
      }
    });
    return () => {
      cancelled = true;
    };
  }, []);

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
        setTagBadgeSize(12);
        setTagBadgeFontSize(8);
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
        setTagBadgeSize(13);
        setTagBadgeFontSize(8);
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
        setTagBadgeSize(14);
        setTagBadgeFontSize(9);
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
        setTagBadgeSize(16);
        setTagBadgeFontSize(10);
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
        setTagBadgeSize(18);
        setTagBadgeFontSize(11);
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
      // Do not allow tile movement when a per-tile text field is open
      if (!questionCommentOpen && !tagInputOpen) {
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
    tagInputOpen,
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
      // Row layout: [row, col, ...class one-hots incl. "?", QuestionComment,
      // Tags]. Emitted at full fixed width so the two trailing text fields sit
      // at stable indices.
      const commentIdx = 2 + valueMapCnn1.length;
      const tagsIdx = commentIdx + 1;
      // Only classified tiles are written. A row with tags but an all-zero
      // one-hot would be skipped on reload and read as the FIRST class by
      // np.argmax during training, so it must never be saved.
      cnn1Annotations.forEach((value, key) => {
        const out = new Array(tagsIdx + 1).fill(0);
        out[commentIdx] = "";
        out[tagsIdx] = "";

        const splitKey = key.split(".");
        out[0] = parseInt(splitKey[0]);
        out[1] = parseInt(splitKey[1]);

        const populated_index = valueMapCnn1.indexOf(value);
        if (populated_index !== -1) {
          // Add to array, skipping row and col indices
          out[populated_index + 2] = 1;
        }

        // If question mark then check for comment
        if (populated_index === valueMapCnn1.length - 1) {
          if (questionMarkComments.has(key)) {
            out[commentIdx] = questionMarkComments.get(key);
          }
        }

        if (tileTags.has(key)) {
          out[tagsIdx] = formatTags(tileTags.get(key));
        }

        cnn1Values.push(out);
      });

      let body = {
        imageReferenceId, // Derived from the pathname
        fileName: selectedImageName, // Taken from GlobalContextProvider
        cnnOneValues: cnn1Values,
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

    cnn1AnnotationActions.setBulk(cnn1AnnotationsMap);

    if (settings?.useDb === false) {
      // Local mode: predictions and annotations are separate files with
      // separate identities (unlike DB mode's one shared ImageReference
      // row), so imageReferenceId here is really the predictions file's own
      // name. Reusing it as the annotations id would make the save
      // overwrite that predictions file in place instead of creating a new
      // annotations CSV, and would make the URL match both "predictions"
      // and "annotations" substrings at once (showing both views
      // simultaneously). Always start a fresh annotation set instead.
      navigate(`/browser/${selectedImageName}/${colonisationType}/annotations`);
    } else {
      // Navigate to existing annotation page
      navigate(
        `/browser/${selectedImageName}/${colonisationType}/annotations/${imageReferenceId}`,
      );
    }
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
    if (!selectIcons) {
      if (selectedOverlay === "?") {
        setSelectedOverlay(null);
      } else {
        setSelectedOverlay("?");
      }
    } else {
      setQuestionTile(currentKey);
    }
  };

  /////////////////////////////////////////////////////////////////////
  // Sub-tags

  const selectedTileKey = `${selectedTile.row}.${selectedTile.col}`;

  // Creates (or recolours) a tag and persists the palette. Tag names may not
  // contain the delimiter that packs them into a single CSV cell.
  const onCreateTag = async (name, colour) => {
    const tag = (name ?? "").trim();
    if (tag === "") {
      return;
    }
    if (tag.includes("|")) {
      toast.error('Tag names cannot contain "|"');
      return;
    }
    const nextPalette = { ...tagPalette, [tag]: colour };
    setTagPalette(nextPalette);
    const saved = await PredictionsApi.saveTagPalette(nextPalette);
    if (saved) {
      setTagPalette(saved);
    }
  };

  // Forgets a tag entirely: removed from the palette and from every tile.
  const onDeleteTag = async (tag) => {
    setTileTags((prevMap) => {
      const nextMap = new Map();
      prevMap.forEach((tags, key) => {
        const kept = tags.filter((t) => t !== tag);
        if (kept.length > 0) {
          nextMap.set(key, kept);
        }
      });
      return nextMap;
    });
    const nextPalette = { ...tagPalette };
    delete nextPalette[tag];
    setTagPalette(nextPalette);
    const saved = await PredictionsApi.saveTagPalette(nextPalette);
    if (saved) {
      setTagPalette(saved);
    }
  };

  const onToggleTagOnSelectedTile = (tag) => {
    tileTagActions.toggle(selectedTileKey, tag);
  };

  const onClearTagsOnSelectedTile = () => {
    tileTagActions.remove(selectedTileKey);
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
      let tagsMap = new Map();
      let values = annotations[imageId];
      // Trailing fields are located by index rather than by row length, so a
      // row carrying tags does not hide the question comment (and a legacy
      // row that stops short of either index simply has neither).
      const commentIdx = 2 + valueMapCnn1.length;
      const tagsIdx = commentIdx + 1;
      values?.forEach((value) => {
        let key = `${value[0].toString()}.${value[1].toString()}`;
        // Ignore row and col, and stop before the trailing text fields so
        // they can never be mistaken for a set class flag.
        const labels = value.slice(2, commentIdx);
        // Find which indices have value 1 and then set the grid values to these
        const indexes = labels.reduce((r, n, i) => {
          n === 1 && r.push(i);
          return r;
        }, []);
        if (indexes.length > 0) {
          let tileValue = headerToValueMapCnn1.get(headerMapCnn1[indexes[0]]);
          if (tileValue === "?") {
            setQuestionTile(key);
            // In this case, download question comment as well and set if non-empty
            if (value.length > commentIdx) {
              const questionComment = value[commentIdx];
              // Set if non-empty and non null
              if (questionComment !== null && questionComment !== "") {
                questionMarkCommentActions.set(key, questionComment);
              }
            }
          }

          // Absent in annotation files written before sub-tags existed.
          if (value.length > tagsIdx) {
            const tags = parseTags(value[tagsIdx]);
            if (tags.length > 0) {
              tagsMap.set(key, tags);
            }
          }

          annotationsMap.set(key, tileValue);
        }
      });

      cnn1AnnotationActions.setBulk(annotationsMap);
      tileTagActions.setBulk(tagsMap);
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
        // Directly add predictions for each class
        labels.forEach((value, idx) => {
          output.push([headerMapCnn1[idx], value]);
        });
        predictionsMap.set(key, {
          predictions: output,
          annotations: null,
          contextualLabel: convertedContextualLabel,
        });
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
    const predictionsCnn1 = await PredictionsApi.fetchPredictionsById(
      id,
      "1",
      colonisationType,
    );

    await loadImageTiles(tileEdge, image);

    initialSetAnnotations(annotationsCnn1, 1);
    initialSetPredictions(predictionsCnn1, 1);
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
      className="fullWidth fullHeight flexColumn"
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
          style={{ flex: 1, minHeight: 0 }}
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
            headerToValueMapCnn1={headerToValueMapCnn1}
            keyBindingsCnn1={keyBindingsCnn1}
            selectedTile={selectedTile}
            setTileValueWithIcon={setTileValueWithIcon}
            cnn1Annotations={cnn1Annotations}
            cnn1Predictions={cnn1Predictions}
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
            // Sub-tags
            tileTags={tileTags}
            allTags={allTags}
            tagPalette={tagPalette}
            selectedTileTags={tileTags.get(selectedTileKey) ?? []}
            hasSelectedClass={cnn1Annotations.has(selectedTileKey)}
            onCreateTag={onCreateTag}
            onDeleteTag={onDeleteTag}
            onToggleTag={onToggleTagOnSelectedTile}
            onClearTags={onClearTagsOnSelectedTile}
            setTagInputOpen={setTagInputOpen}
            tagBadgeSize={tagBadgeSize}
            tagBadgeFontSize={tagBadgeFontSize}
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
            autoClassifyBackground={autoClassifyBackground}
            onAutoClassifyBackgroundToggle={handleAutoClassifyBackgroundToggle}
            canUndoBackgroundFill={canUndoBackgroundFill}
            onToggleBackgroundFill={toggleBackgroundFill}
            backgroundThreshold={backgroundThreshold}
            onBackgroundThresholdChange={handleBackgroundThresholdChange}
            onBackgroundThresholdBlur={flushBackgroundThresholdChange}
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
              <div className="save-questions-row buttonRow">
                <PrimaryButton
                  sx={{ minWidth: "200px" }}
                  onClick={() => saveAnnotations(true)}
                >
                  Save Annotations
                </PrimaryButton>
                <PrimaryButton
                  sx={{ minWidth: "200px" }}
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
              colorMapping={colorMappingCnn1Transparent}
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
