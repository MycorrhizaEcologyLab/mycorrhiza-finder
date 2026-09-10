import TextField from "@mui/material/TextField";
import { useState } from "react";
import { FaRedo } from "react-icons/fa";

const SettingsItemInteger = ({
  name,
  displayName,
  defaultValue,
  help,
  value,
  handleInputChange,
  disabled = false,
  min = 0,
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
        title={`${help ? help + " " : ""}(type: int, default: ${defaultValue})`}
      >
        {displayName}
      </label>
      <TextField
        className="text-input"
        name={name}
        value={value ?? ""}
        onChange={handleInputChange}
        disabled={disabled}
        min={min}
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

export default SettingsItemInteger;
