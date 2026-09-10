import { styled } from "@mui/material/styles";
import Button from "@mui/material/Button";

const PrimaryButton = styled(Button)(() => ({
  color: "white",
  // minHeight rather than a fixed height so a button can grow instead of
  // clipping its label, and a real lineHeight so that if a label ever does
  // wrap, the second line sits below the first instead of on top of it.
  minHeight: "40px",
  lineHeight: 1.35,
  // Labels stay on one line; call sites use minWidth (not width) so the
  // button grows to fit its text rather than overflowing it.
  whiteSpace: "nowrap",
  // Never let a flex row squeeze a button below its label width.
  flexShrink: 0,
  padding: "6px 16px",
  // Shrink the label slightly on small viewports instead of demanding
  // more width than the row can give.
  fontSize: "clamp(0.75rem, 0.7rem + 0.25vw, 0.875rem)",
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
