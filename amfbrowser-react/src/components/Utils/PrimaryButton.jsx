import { styled } from "@mui/material/styles";
import Button from "@mui/material/Button";

const PrimaryButton = styled(Button)(() => ({
  color: "white",
  height: "40px",
  lineHeight: 0,
  fontWeight: 600,
  borderRadius: "1.75rem",
  borderStyle: "solid",
  borderWidth: "2px",
  borderColor: "rgb(174, 171, 158)",
  backgroundColor: "rgb(0, 130, 133)",
  "&:hover": {
    backgroundColor: "rgb(1, 107, 112)",
  },
  "&.Mui-disabled": {
    background: "rgb(227, 226, 221)",
    color: "rgb(120, 117, 101)",
  },
}));

export default PrimaryButton;
