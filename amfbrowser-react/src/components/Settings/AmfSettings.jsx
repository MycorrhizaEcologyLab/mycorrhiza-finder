import React, { useContext, useEffect } from "react";

import { toast } from "react-toastify";
import CircularProgress from "@mui/material/CircularProgress";

import "./styles/settings.css";
import { GlobalContextProvider } from "../../contexts/Contexts";
import PrimaryButton from "../Utils/PrimaryButton";
import SettingsItemText from "./SettingsItemText";
import SettingsItemSelect from "./SettingsItemSelect";
import SettingsItemCheckbox from "./SettingsItemCheckbox";
import SettingsItemFloat from "./SettingsItemFloat";
import SettingsItemInteger from "./SettingsItemInteger";
import PredictionsApi from "../../api/amfinderApi";

const AmfSettings = () => {
  const { colonisationType, tileEdge, setTileEdge, settings, setSettings } =
    useContext(GlobalContextProvider);

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

  useEffect(() => {
    if (
      settings === null ||
      settings === undefined ||
      Object.keys(settings).length === 0
    ) {
      fetchInitialSettings();
    }
  }, []);

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
    let outputValue;
    if (type === "checkbox") {
      const { checked } = event.target;
      outputValue = checked;
    } else {
      const { value } = event.target;
      if (
        name === "learningRate" ||
        name === "threshold" ||
        name === "adamBeta1" ||
        name === "adamBeta2" ||
        name === "vfrac" ||
        name === "balanceFactor" ||
        name === "learningRateActiveLearning" ||
        name === "dropoutRate"
      ) {
        outputValue = parseFloat(value);
      } else if (
        name === "batchSize" ||
        name === "epochs" ||
        name === "vfrac" ||
        name === "patienceE" ||
        name === "patienceR" ||
        name === "numSamplesForLabelling" ||
        name === "mcSamples" ||
        name === "epochsActiveLearning"
      ) {
        outputValue = parseInt(value);
      } else {
        outputValue = value;
      }
    }

    setSettings((prevState) => ({
      ...prevState,
      [name]: outputValue,
    }));
  };

  return (
    <div className="settings-grid">
      {(settings === null ||
        settings === undefined ||
        Object.keys(settings).length === 0) && (
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
      {settings !== null &&
        settings !== undefined &&
        Object.keys(settings).length !== 0 && (
          <div>
            {/* GENERAL */}
            <div className="settings-heading" id="general">
              General Settings
            </div>
            <SettingsItemText
              name="outdir"
              displayName="Output directory"
              defaultValue="images folder"
              value={settings.outdir}
              handleInputChange={handleInputChange}
              resetToDefault={resetToDefault}
            />
            <SettingsItemText
              name="model"
              displayName="Model for AM"
              defaultValue="efficientnet_252_6class_D2.pth"
              value={settings.model}
              handleInputChange={handleInputChange}
              resetToDefault={resetToDefault}
            />
            <SettingsItemText
              name="modelErm"
              displayName="Model for ErM"
              defaultValue="250411_126_ErM_EfficientNet_82.pth"
              value={settings.modelErm}
              handleInputChange={handleInputChange}
              resetToDefault={resetToDefault}
            />
            <SettingsItemText
              name="temperatureFactorPath"
              displayName="Model Temperature Factor Path for AM"
              defaultValue="efficientnet_252_6class_D2_temperature_value.txt"
              value={settings.temperatureFactorPath}
              handleInputChange={handleInputChange}
              resetToDefault={resetToDefault}
            />
            <SettingsItemText
              name="temperatureFactorPathErm"
              displayName="Model Temperature Factor Path for ErM"
              defaultValue="250411_126_ErM_EfficientNet_82_temperature_value.txt"
              value={settings.temperatureFactorPathErm}
              handleInputChange={handleInputChange}
              resetToDefault={resetToDefault}
            />
            <SettingsItemSelect
              name="device"
              displayName="Device"
              defaultValue="Automatic"
              value={settings.device}
              options={[
                { value: "automatic", label: "Automatic" },
                { value: "cpu", label: "CPU" },
                { value: "cuda:0", label: "Cuda (GPU)" },
              ]}
              handleInputChange={handleInputChange}
              resetToDefault={resetToDefault}
            />
            <SettingsItemSelect
              name="tileEdge"
              displayName="Tile Edge"
              defaultValue={252}
              value={tileEdge}
              options={[
                { value: 252, label: "252" },
                { value: 126, label: "126" },
              ]}
              handleInputChange={(e) => setTileEdge(parseInt(e.target.value))}
            />
            <SettingsItemCheckbox
              name="useDb"
              displayName="Use local database"
              defaultValue={true}
              checked={settings.useDb}
              handleInputChange={handleInputChange}
              resetToDefault={resetToDefault}
            />
            <SettingsItemCheckbox
              name="useContextualConfidence"
              displayName="Use Contextual Confidence"
              defaultValue={true}
              checked={settings.useContextualConfidence}
              handleInputChange={handleInputChange}
              resetToDefault={resetToDefault}
            />
            <SettingsItemFloat
              name="contextualConfidenceThreshold"
              displayName="Contextual Confidence Threshold"
              defaultValue={1.0}
              value={settings.contextualConfidenceThreshold}
              handleInputChange={handleInputChange}
              resetToDefault={resetToDefault}
            />
            {/* TRAINING */}
            <div className="settings-heading" id="train">
              Train
            </div>
            <SettingsItemSelect
              name="modelType"
              displayName="Model Type"
              defaultValue="EfficientNet"
              value={settings.modelType}
              options={[
                { value: "cnn1", label: "CNN1" },
                { value: "resnet", label: "ResNet" },
                { value: "resnext", label: "ResNeXt" },
                { value: "efficientnet", label: "EfficientNet" },
                { value: "efficientnetv2", label: "EfficientNetV2" },
              ]}
              handleInputChange={handleInputChange}
              resetToDefault={resetToDefault}
            />
            <SettingsItemCheckbox
              name="trainActiveLearning"
              displayName="Train model after active learning"
              defaultValue={false}
              checked={settings.trainActiveLearning}
              handleInputChange={handleInputChange}
              resetToDefault={resetToDefault}
            />
            <SettingsItemInteger
              name="batchSize"
              displayName="Batch Size"
              defaultValue={32}
              value={settings.batchSize}
              handleInputChange={handleInputChange}
              resetToDefault={resetToDefault}
              min={0}
            />
            <SettingsItemInteger
              name="epochs"
              displayName="Epochs"
              defaultValue={50}
              value={settings.epochs}
              handleInputChange={handleInputChange}
              resetToDefault={resetToDefault}
              disabled={settings.trainActiveLearning}
              min={0}
            />
            <SettingsItemInteger
              name="epochsActiveLearning"
              displayName="Epochs after active learning"
              defaultValue={5}
              value={settings.epochsActiveLearning}
              handleInputChange={handleInputChange}
              resetToDefault={resetToDefault}
              disabled={!settings.trainActiveLearning}
              min={0}
            />
            <SettingsItemFloat
              name="vfrac"
              displayName="VFrac"
              defaultValue={0.2}
              value={settings.vfrac}
              handleInputChange={handleInputChange}
              resetToDefault={resetToDefault}
            />
            <SettingsItemInteger
              name="patienceE"
              displayName="Patience - Early Stopping"
              defaultValue={10}
              value={settings.patienceE}
              handleInputChange={handleInputChange}
              resetToDefault={resetToDefault}
            />
            <SettingsItemInteger
              name="patienceR"
              displayName="Patience - Learning Rate Reduction"
              defaultValue={5}
              value={settings.patienceR}
              handleInputChange={handleInputChange}
              resetToDefault={resetToDefault}
            />
            <SettingsItemCheckbox
              name="dataAugm"
              displayName="Data augmentation"
              defaultValue={false}
              checked={settings.dataAugm}
              handleInputChange={handleInputChange}
              resetToDefault={resetToDefault}
            />
            <SettingsItemCheckbox
              name="summary"
              displayName="Summary"
              defaultValue={false}
              checked={settings.summary}
              handleInputChange={handleInputChange}
              resetToDefault={resetToDefault}
            />
            <SettingsItemCheckbox
              name="semiSupervised"
              displayName="Semi-supervised"
              defaultValue={false}
              checked={settings.semiSupervised}
              handleInputChange={handleInputChange}
              resetToDefault={resetToDefault}
            />
            <SettingsItemFloat
              name="learningRate"
              displayName="Learning Rate"
              defaultValue={0.000004218361045}
              value={settings.learningRate}
              handleInputChange={handleInputChange}
              resetToDefault={resetToDefault}
              disabled={settings.trainActiveLearning}
            />
            <SettingsItemFloat
              name="learningRateActiveLearning"
              displayName="Learning Rate after active learning"
              defaultValue={0.0000004}
              value={settings.learningRateActiveLearning}
              handleInputChange={handleInputChange}
              resetToDefault={resetToDefault}
              disabled={!settings.trainActiveLearning}
            />
            <SettingsItemFloat
              name="adamBeta1"
              displayName="Adam Beta 1"
              defaultValue={0.906450740008503}
              value={settings.adamBeta1}
              handleInputChange={handleInputChange}
              resetToDefault={resetToDefault}
            />
            <SettingsItemFloat
              name="adamBeta2"
              displayName="Adam Beta 2"
              defaultValue={0.986390310777448}
              value={settings.adamBeta2}
              handleInputChange={handleInputChange}
              resetToDefault={resetToDefault}
            />
            <SettingsItemFloat
              name="balanceFactor"
              displayName="Balance Factor"
              defaultValue={1.24895925434138}
              value={settings.balanceFactor}
              handleInputChange={handleInputChange}
              resetToDefault={resetToDefault}
            />
            <SettingsItemCheckbox
              name="mlFlowFlag"
              displayName="ML Flow flag"
              defaultValue={false}
              checked={settings.mlFlowFlag}
              handleInputChange={handleInputChange}
              resetToDefault={resetToDefault}
            />
            <SettingsItemCheckbox
              name="preTrained"
              displayName="Use pre-trained model"
              defaultValue={false}
              checked={settings.preTrained}
              handleInputChange={handleInputChange}
              resetToDefault={resetToDefault}
            />
            {/* ACTIVE LEARNING */}
            <div className="settings-heading" id="active">
              Active Learning
            </div>
            <SettingsItemCheckbox
              name="getTilesForLabellingUsingActiveLearning"
              displayName="Output tiles for labelling"
              defaultValue={false}
              checked={settings.getTilesForLabellingUsingActiveLearning}
              handleInputChange={handleInputChange}
              resetToDefault={resetToDefault}
            />
            <SettingsItemSelect
              name="activeLearningMethod"
              displayName="Active Learning Method"
              defaultValue="bald"
              value={settings.activeLearningMethod}
              options={[
                { value: "bald", label: "BALD" },
                { value: "batchbald", label: "BatchBALD" },
              ]}
              handleInputChange={handleInputChange}
              resetToDefault={resetToDefault}
            />
            <SettingsItemInteger
              name="numSamplesForLabelling"
              displayName="Number of samples for labelling"
              defaultValue={10}
              value={settings.numSamplesForLabelling}
              handleInputChange={handleInputChange}
              resetToDefault={resetToDefault}
            />
            <SettingsItemInteger
              name="mcSamples"
              displayName="Number of Monte Carlo Samples"
              defaultValue={50}
              value={settings.mcSamples}
              handleInputChange={handleInputChange}
              resetToDefault={resetToDefault}
            />
            {/* CONVERT */}
            <div className="settings-heading" id="convert">
              Convert
            </div>
            <SettingsItemFloat
              name="threshold"
              displayName="Threshold"
              defaultValue={0.5}
              value={settings.threshold}
              handleInputChange={handleInputChange}
              resetToDefault={resetToDefault}
            />
            <SettingsItemCheckbox
              name="aggregateTiles"
              displayName="Aggregate Tiles"
              defaultValue={false}
              checked={settings.aggregateTiles}
              handleInputChange={handleInputChange}
              resetToDefault={resetToDefault}
            />
            <div className="settings-heading" id="convert">
              TIF Conversion
            </div>
            <SettingsItemSelect
              name="convertImageFileType"
              displayName="TIF Conversion File Type"
              defaultValue="jpg"
              value={settings.convertImageFileType}
              options={[
                { value: "jpg", label: "JPG" },
                { value: "png", label: "PNG" },
              ]}
              handleInputChange={handleInputChange}
              resetToDefault={resetToDefault}
            />
            {/* TEST */}
            <div className="settings-heading" id="test">
              Test
            </div>
            <SettingsItemCheckbox
              name="semiSupervised"
              displayName="Semi-supervised"
              defaultValue={false}
              checked={settings.semiSupervised}
              handleInputChange={handleInputChange}
              resetToDefault={resetToDefault}
            />
            <SettingsItemText
              name="fixmatchResultsDirectory"
              displayName="Semi-supervised results directory"
              defaultValue="images folder"
              value={settings.fixmatchResultsDirectory}
              handleInputChange={handleInputChange}
              resetToDefault={resetToDefault}
              disabled={!settings.semiSupervised}
            />
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
