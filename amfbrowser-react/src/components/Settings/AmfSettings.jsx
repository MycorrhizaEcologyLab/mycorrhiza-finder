import { useContext, useEffect, useMemo, useRef, useState } from "react";

import CircularProgress from "@mui/material/CircularProgress";
import FormControlLabel from "@mui/material/FormControlLabel";
import Switch from "@mui/material/Switch";
import Tab from "@mui/material/Tab";
import Tabs from "@mui/material/Tabs";
import { toast } from "react-toastify";

import PredictionsApi from "../../api/amfinderApi";
import { GlobalContextProvider } from "../../contexts/Contexts";
import PrimaryButton from "../Utils/PrimaryButton";
import SettingsField from "./SettingsField";
import SettingsItemSelect from "./SettingsItemSelect";
import "./styles/settings.css";

// Order tabs are shown in; only groups that actually have entries in the
// fetched schema render as a tab.
const TAB_ORDER = [
  { key: "general", label: "General" },
  { key: "train", label: "Train" },
  { key: "activeLearning", label: "Active Learning" },
  { key: "predict", label: "Predict" },
  { key: "test", label: "Test" },
  { key: "convert", label: "Convert" },
  { key: "tifconversion", label: "TIF Conversion" },
  { key: "calibrate", label: "Calibrate" },
  { key: "colonisation", label: "Colonisation" },
];

// A handful of fields depend on each other in ways the schema doesn't
// express (e.g. training uses either the "normal" or "after active
// learning" learning rate/epoch pair, never both).
const DISABLE_RULES = {
  epochs: (settings) => !!settings.trainActiveLearning,
  epochsActiveLearning: (settings) => !settings.trainActiveLearning,
  learningRate: (settings) => !!settings.trainActiveLearning,
  learningRateActiveLearning: (settings) => !settings.trainActiveLearning,
};

const AmfSettings = () => {
  const { colonisationType, tileEdge, setTileEdge, settings, setSettings } =
    useContext(GlobalContextProvider);

  const [schema, setSchema] = useState([]);
  const [trainedModels, setTrainedModels] = useState([]);
  const [activeTab, setActiveTab] = useState("general");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const hasCheckedInitialModelTypeRef = useRef(false);

  useEffect(() => {
    if (colonisationType === "am") {
      setTileEdge(252);
    } else {
      setTileEdge(126);
    }
  }, [colonisationType]);

  const fetchInitialSettings = async () => {
    const initialSettings = await PredictionsApi.getAllSettings();
    setSettings(initialSettings);
  };

  const fetchSchema = async () => {
    const initialSchema = await PredictionsApi.getSettingsSchema();
    setSchema(initialSchema ?? []);
  };

  const fetchTrainedModels = async () => {
    const models = await PredictionsApi.getTrainedModels();
    setTrainedModels(models ?? []);
  };

  useEffect(() => {
    if (
      settings === null ||
      settings === undefined ||
      Object.keys(settings).length === 0
    ) {
      fetchInitialSettings();
    }
    if (schema.length === 0) {
      fetchSchema();
    }
    fetchTrainedModels();
  }, []);

  const specByKey = useMemo(
    () => Object.fromEntries(schema.map((spec) => [spec.key, spec])),
    [schema],
  );

  const availableGroups = useMemo(() => {
    const groupsInSchema = new Set(schema.map((spec) => spec.group));
    return TAB_ORDER.filter((tab) => groupsInSchema.has(tab.key));
  }, [schema]);

  const fieldsForTab = useMemo(
    () =>
      schema.filter(
        (spec) => spec.group === activeTab && (showAdvanced || !spec.advanced),
      ),
    [schema, activeTab, showAdvanced],
  );

  // One-time startup check: if the saved/default Model Type is already a
  // transformer (currently only DeiT3), make sure Resize Dimension reflects
  // that instead of staying blank/stale from before. Guarded to run only
  // once so it doesn't fight the user if they interactively clear Resize
  // Dimension later while still on a transformer model type - that
  // interactive case is already handled by handleInputChange below.
  useEffect(() => {
    if (
      hasCheckedInitialModelTypeRef.current ||
      schema.length === 0 ||
      !settings ||
      Object.keys(settings).length === 0
    ) {
      return;
    }
    hasCheckedInitialModelTypeRef.current = true;

    const modelTypeSpec = specByKey.modelType;
    const currentModelType = settings.modelType ?? modelTypeSpec?.default;
    const chosenChoice = modelTypeSpec?.choices?.find(
      (choice) => choice.value === currentModelType,
    );
    if (
      chosenChoice?.isTransformer &&
      settings.resizeDim !== chosenChoice.transformerResizeDim
    ) {
      setSettings((prevState) => ({
        ...prevState,
        resizeDim: chosenChoice.transformerResizeDim,
      }));
    }
  }, [schema, settings, specByKey, setSettings]);

  // Loads whichever model is actually going to be used (Model Path Override
  // takes priority, same as the backend's own resolution order, else Model
  // for AM/ErM) and inspects its real class to tell whether it's a
  // transformer - so users who never touch the Train tab's Model Type
  // dropdown (e.g. they only set Model Path Override) still get Resize
  // Dimension set correctly.
  const checkModelArchitecture = async () => {
    const resolvedPath =
      settings.modelPath ||
      (colonisationType === "am" ? settings.model : settings.modelErm);
    if (!resolvedPath) {
      toast.error(
        "No model path configured to check (Model Path Override, Model for AM, or Model for ErM).",
      );
      return;
    }

    const result = await PredictionsApi.checkModelArchitecture(resolvedPath);
    if (!result || !result.exists) {
      toast.error(`Could not find or load model: ${resolvedPath}`);
      return;
    }

    const transformerChoice = specByKey.modelType?.choices?.find(
      (choice) => choice.isTransformer,
    );
    const transformerResizeDim = transformerChoice?.transformerResizeDim ?? 224;

    if (result.isTransformer) {
      setSettings((prevState) => ({
        ...prevState,
        resizeDim: transformerResizeDim,
      }));
      toast.success(
        `Detected ${result.modelClassName} (transformer) - Resize Dimension set to ${transformerResizeDim}.`,
      );
    } else {
      toast.info(
        `Detected ${result.modelClassName ?? "unknown"} architecture - no Resize Dimension change needed.`,
      );
    }
  };

  const saveSettings = async () => {
    const response = await PredictionsApi.saveSettings(settings);
    if (response === 200) {
      toast.success("Successfully saved settings!");
    } else {
      toast.error("Failed to save settings");
    }
  };

  const resetToDefault = async (name) => {
    const response = await PredictionsApi.revertSettingToDefault({
      key: name,
      value: "",
    });
    if (response) {
      setSettings((prevState) => ({
        ...prevState,
        [response["key"]]: response["value"],
      }));
      toast.success(`Successfully reset ${response["key"]} to default value`);
    } else {
      toast.error("Failed to reset to default value");
    }
  };

  const resetAllToDefault = async () => {
    const response = await PredictionsApi.revertAllSettingsToDefault(settings);
    if (response === 200) {
      const settings = await PredictionsApi.getAllSettings();
      setSettings(settings);
      toast.success("All settings reset to default");
    } else {
      toast.error("Error resetting all to default");
    }
  };

  const handleInputChange = (event) => {
    const { name, type } = event.target;
    const spec = specByKey[name];
    let outputValue;
    if (type === "checkbox") {
      outputValue = event.target.checked;
    } else {
      const { value } = event.target;
      if (spec?.type === "float") {
        outputValue = value === "" ? null : parseFloat(value);
      } else if (spec?.type === "integer") {
        outputValue = value === "" ? null : parseInt(value, 10);
      } else {
        outputValue = value;
      }
    }

    // Transformer model types (currently only DeiT3) expect a fixed input
    // resolution, so keep Resize Dimension in sync with the chosen model
    // type instead of leaving it stale from whatever was picked before.
    if (name === "modelType") {
      const choice = spec?.choices?.find((option) => option.value === outputValue);
      setSettings((prevState) => ({
        ...prevState,
        modelType: outputValue,
        resizeDim: choice?.isTransformer ? choice.transformerResizeDim : null,
      }));
      return;
    }

    setSettings((prevState) => ({
      ...prevState,
      [name]: outputValue,
    }));
  };

  const isLoading =
    settings === null ||
    settings === undefined ||
    Object.keys(settings).length === 0 ||
    schema.length === 0;

  return (
    <div className="settings-grid">
      {isLoading && (
        <div
          style={{
            width: "100%",
            height: "calc(100% - 60px)",
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
          }}
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
      {!isLoading && (
        <div>
          <div className="settings-toolbar">
            <FormControlLabel
              control={
                <Switch
                  checked={showAdvanced}
                  onChange={(event) => setShowAdvanced(event.target.checked)}
                />
              }
              label="Show advanced settings"
            />
          </div>
          <Tabs
            className="settings-tabs"
            value={activeTab}
            onChange={(_event, value) => setActiveTab(value)}
          >
            {availableGroups.map((tab) => (
              <Tab key={tab.key} value={tab.key} label={tab.label} />
            ))}
          </Tabs>

          {activeTab === "general" && (
            <SettingsItemSelect
              name="tileEdge"
              displayName="Tile Edge"
              defaultValue={252}
              value={tileEdge}
              options={[
                { value: 252, label: "252" },
                { value: 126, label: "126" },
              ]}
              handleInputChange={(e) => setTileEdge(parseInt(e.target.value, 10))}
              resetToDefault={() => setTileEdge(colonisationType === "am" ? 252 : 126)}
            />
          )}

          {activeTab === "general" && (
            <div className="settings-item">
              <label>
                Checks whichever model is actually active (Model Path
                Override, else Model for AM/ErM) and sets Resize Dimension
                automatically if it's a transformer.
              </label>
              <PrimaryButton
                sx={{ flexShrink: 0 }}
                onClick={checkModelArchitecture}
              >
                Check Model Architecture
              </PrimaryButton>
            </div>
          )}

          {fieldsForTab.map((spec) => (
            <SettingsField
              key={spec.key}
              spec={spec}
              value={settings[spec.key] ?? spec.default}
              handleInputChange={handleInputChange}
              resetToDefault={resetToDefault}
              disabled={DISABLE_RULES[spec.key]?.(settings) ?? false}
              modelOptions={trainedModels}
            />
          ))}
        </div>
      )}
      {/* SAVE */}
      <div className="full-width-sticky settings-save">
        <PrimaryButton className="settingsSaveButton" onClick={saveSettings}>
          Save Settings
        </PrimaryButton>
        <PrimaryButton
          className="settingsSaveButton"
          onClick={resetAllToDefault}
        >
          Reset to defaults
        </PrimaryButton>
      </div>
    </div>
  );
};

export default AmfSettings;
