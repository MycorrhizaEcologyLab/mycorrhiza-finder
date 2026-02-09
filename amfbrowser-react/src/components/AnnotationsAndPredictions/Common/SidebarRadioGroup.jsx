import FormControlLabel from "@mui/material/FormControlLabel";
import FormLabel from "@mui/material/FormLabel";
import Radio from "@mui/material/Radio";
import RadioGroup from "@mui/material/RadioGroup";

const SidebarRadioGroup = ({
  label,
  value,
  onChange,
  options,
  disabledOptions = [],
}) => {
  return (
    <div id="sidebarRadioGroupContainer" style={{ padding: "20px 10px" }}>
      <FormLabel
        sx={{
          display: "flex",
          paddingLeft: "8px",
          paddingBottom: "10px",
          color: "black",
          fontWeight: 600,
          fontSize: "12px",
        }}
      >
        {label}
      </FormLabel>
      <RadioGroup value={value} onChange={(e) => onChange(e)}>
        {options.map((option) => (
          <FormControlLabel
            key={option}
            value={option}
            disabled={disabledOptions.includes(option)}
            control={<Radio sx={{ padding: "5px" }} />}
            sx={{ margin: 0 }}
            label={
              <p style={{ margin: 0, fontSize: "12px", fontWeight: 500 }}>
                {option}
              </p>
            }
          />
        ))}
      </RadioGroup>
    </div>
  );
};

export default SidebarRadioGroup;
