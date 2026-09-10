import SettingsItemCheckbox from "./SettingsItemCheckbox";
import SettingsItemFloat from "./SettingsItemFloat";
import SettingsItemInteger from "./SettingsItemInteger";
import SettingsItemSelect from "./SettingsItemSelect";
import SettingsItemText from "./SettingsItemText";

const formatDefault = (value) => {
  if (value === null || value === undefined || value === "") {
    return "not set";
  }
  return String(value);
};

// Renders the right SettingsItem* component for a settings-schema entry, so
// adding a setting to the backend schema is enough for it to show up here.
const SettingsField = ({
  spec,
  value,
  handleInputChange,
  resetToDefault,
  disabled,
  modelOptions,
}) => {
  const defaultValue = formatDefault(spec.default);

  switch (spec.type) {
    case "modelSelect": {
      // Defensively keep the currently-configured value in the list even if
      // it's momentarily missing from the live directory listing (e.g. the
      // models folder hasn't loaded yet, or the file was renamed/removed),
      // so the select never silently loses track of what's actually set.
      const knownOptions = modelOptions ?? [];
      const availableNames = knownOptions.includes(value)
        ? knownOptions
        : [...knownOptions, value].filter(Boolean);
      return (
        <SettingsItemSelect
          name={spec.key}
          displayName={spec.label}
          defaultValue={defaultValue}
          help={spec.help}
          value={value}
          options={availableNames.map((name) => ({ value: name, label: name }))}
          handleInputChange={handleInputChange}
          resetToDefault={resetToDefault}
          disabled={disabled}
        />
      );
    }
    case "boolean":
      return (
        <SettingsItemCheckbox
          name={spec.key}
          displayName={spec.label}
          defaultValue={defaultValue}
          help={spec.help}
          checked={value}
          handleInputChange={handleInputChange}
          resetToDefault={resetToDefault}
          disabled={disabled}
        />
      );
    case "integer":
      return (
        <SettingsItemInteger
          name={spec.key}
          displayName={spec.label}
          defaultValue={defaultValue}
          help={spec.help}
          value={value}
          handleInputChange={handleInputChange}
          resetToDefault={resetToDefault}
          disabled={disabled}
        />
      );
    case "float":
      return (
        <SettingsItemFloat
          name={spec.key}
          displayName={spec.label}
          defaultValue={defaultValue}
          help={spec.help}
          value={value}
          handleInputChange={handleInputChange}
          resetToDefault={resetToDefault}
          disabled={disabled}
        />
      );
    case "select":
      return (
        <SettingsItemSelect
          name={spec.key}
          displayName={spec.label}
          defaultValue={defaultValue}
          help={spec.help}
          value={value}
          options={spec.choices ?? []}
          handleInputChange={handleInputChange}
          resetToDefault={resetToDefault}
          disabled={disabled}
        />
      );
    default:
      return (
        <SettingsItemText
          name={spec.key}
          displayName={spec.label}
          defaultValue={defaultValue}
          help={spec.help}
          value={value}
          handleInputChange={handleInputChange}
          resetToDefault={resetToDefault}
          disabled={disabled}
        />
      );
  }
};

export default SettingsField;
