import React, { useState } from "react";
import { FaRedo } from "react-icons/fa";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";

const SettingsItemSelect = ({
  name,
  displayName,
  defaultValue,
  value,
  options,
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
    <div className="settings-item">
      <label htmlFor={name} title={`(type: string, default: ${defaultValue})`}>
        {displayName}
      </label>
      <Select
        className="select"
        name={name}
        value={value}
        onChange={handleInputChange}
      >
        {options.map((option, index) => (
          <MenuItem key={index} value={option.value}>
            {option.label}
          </MenuItem>
        ))}
      </Select>
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

export default SettingsItemSelect;
