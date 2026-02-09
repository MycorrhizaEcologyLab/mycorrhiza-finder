import ClearIcon from "@mui/icons-material/Clear";
import DoneIcon from "@mui/icons-material/Done";
import CircularProgress from "@mui/material/CircularProgress";
import "./styles/validationStyles.css";

const ValidationIcon = ({ status }) => {
  let wrapperBackgroundColor;
  switch (status) {
    case "success":
      wrapperBackgroundColor = "rgb(30, 155, 215)";
      break;
    case "error":
      wrapperBackgroundColor = "rgb(255, 91, 91)";
      break;
    case "submitting":
      wrapperBackgroundColor = "rgb(239, 238, 237)";
      break;
    case "none":
      wrapperBackgroundColor = "rgb(239, 238, 237)";
      break;
    default:
      wrapperBackgroundColor = "rgb(239, 238, 237)";
      break;
  }

  const iconOpacity = status === "none" ? 0 : 1;

  return (
    <div
      id="validationIconWrapper"
      style={{
        backgroundColor: wrapperBackgroundColor,
      }}
    >
      {(status === "success" || status === "none") && (
        <DoneIcon
          className="validationIcon"
          sx={{
            opacity: iconOpacity,
          }}
        />
      )}
      {status === "error" && (
        <ClearIcon
          className="validationIcon"
          sx={{
            opacity: iconOpacity,
          }}
        />
      )}
      {status === "submitting" && <CircularProgress />}
    </div>
  );
};

export default ValidationIcon;
