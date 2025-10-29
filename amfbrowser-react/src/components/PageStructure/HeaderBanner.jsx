import { useLocation } from "react-router-dom";
import Radio from "@mui/material/Radio";
import RadioGroup from "@mui/material/RadioGroup";
import FormControlLabel from "@mui/material/FormControlLabel";
import FormLabel from "@mui/material/FormLabel";

const HeaderBanner = ({ colonisationType, setColonisationType }) => {
  const { pathname } = useLocation();

  // Logic for when the "Fungus type" radio group should be disabled:
  // 1) In the annotations or predictions pages
  // 2) in the "About" or "Settings" pages
  const isFungusTypeRadioDisabled =
    pathname.includes("/annotations") ||
    pathname.includes("/predictions") ||
    pathname.includes("/settings") ||
    pathname.includes("/about");

  return (
    <div
      style={{
        display: "flex",
        width: "100%",
        height: "40px",
        backgroundColor: "rgb(232, 233, 236)", // "rgb(236, 232, 228)",
        borderBottomStyle: "solid",
        borderBottomWidth: "1px",
        borderBottomColor: "lightgray",
      }}
    >
      {/* Commenting out the back button (THIS COULD BE RE-INTRODUCED) */}
      {/* <div
        style={{
          width: "80px",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          borderRightStyle: "solid",
          borderRightWidth: "2px",
          borderRightColor: "lightGray",
        }}
      >
        <IconButton onClick={navigateBack}>
          <ArrowBackOutlinedIcon />
        </IconButton>
      </div> */}
      <div
        style={{
          width: "300px",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <FormLabel
            sx={{
              display: "flex",
              marginRight: "15px",
              color: "black",
              fontWeight: 500,
            }}
          >
            Fungus type:
          </FormLabel>
          <RadioGroup
            row
            value={colonisationType}
            onChange={(e) => setColonisationType(e.target.value)}
          >
            <FormControlLabel
              value="am"
              control={<Radio />}
              label="AM"
              disabled={isFungusTypeRadioDisabled}
            />
            <FormControlLabel
              value="erm"
              control={<Radio />}
              label="ErM"
              disabled={isFungusTypeRadioDisabled}
            />
          </RadioGroup>
        </div>
      </div>
      <div
        className="flexRowCenter"
        style={{
          flex: 1,
          backgroundColor:
            colonisationType === "am" ? "rgb(39, 94, 55)" : "rgb(102, 153, 0)",
          fontWeight: 16,
          fontSize: "20px",
          color: "white",
          transition: "background-color 200ms linear",
        }}
      >
        {colonisationType === "am"
          ? "Arbuscular Mycorrhiza"
          : "Ericoid Mycorrhiza"}
      </div>
    </div>
  );
};

export default HeaderBanner;
