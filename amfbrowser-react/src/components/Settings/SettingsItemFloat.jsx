import TextField from "@mui/material/TextField";
import { useState } from "react";
import { FaRedo } from "react-icons/fa";

const SettingsItemFloat = ({
  name,
  displayName,
  defaultValue,
  help,
  value,
  handleInputChange,
  step = "any",
  disabled = false,
  resetToDefault,
}) => {
  const [isPulsing, setIsPulsing] = useState(false);

  const handleClick = () => {
    setIsPulsing(true);
    setTimeout(() => {
      setIsPulsing(false);
    }, 500);
  };

  return (
    <div className="settings-item">
      <label
        htmlFor={name}
        title={`${help ? help + " " : ""}(type: float, default: ${defaultValue})`}
      >
        {displayName}
      </label>
      <TextField
        className="text-input"
        name={name}
        value={value ?? ""}
        onChange={handleInputChange}
        disabled={disabled}
        step={step}
        type="number"
      />
      <FaRedo
        className={`reset-setting ${isPulsing ? "pulse" : ""}`}
        onClick={() => {
          resetToDefault(name);
          handleClick();
        }}
      />
    </div>
  );
};

export default SettingsItemFloat;
