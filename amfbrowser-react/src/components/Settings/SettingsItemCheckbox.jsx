import Checkbox from "@mui/material/Checkbox";
import { useState } from "react";
import { FaRedo } from "react-icons/fa";

const SettingsItemCheckbox = ({
  name,
  displayName,
  defaultValue,
  help,
  checked,
  handleInputChange,
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
    <div className="settings-item settings-item-checkbox">
      <label
        htmlFor={name}
        title={`${help ? help + " " : ""}(type: boolean, default: ${defaultValue})`}
      >
        {displayName}
      </label>
      <Checkbox
        name={name}
        color="primary"
        checked={checked ?? false}
        onChange={handleInputChange}
        disabled={disabled}
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

export default SettingsItemCheckbox;
