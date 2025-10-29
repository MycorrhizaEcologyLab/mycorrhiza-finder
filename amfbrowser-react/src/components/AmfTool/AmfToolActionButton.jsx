import Button from "@mui/material/Button";

const AmfToolActionButton = ({ isSelected, disabled, onClick, children }) => (
  <Button
    disabled={disabled}
    onClick={onClick}
    sx={{
      width: "125px",
      height: "125px",
      minWidth: "125px",
      minHeight: "125px",
      marginLeft: "10px",
      marginRight: "10px",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      justifyContent: "center",
      color: isSelected ? "white" : "rgb(60, 59, 50)",
      fontWeight: 600,
      borderRadius: "10px",
      borderStyle: "solid",
      borderWidth: "2px",
      borderColor: "rgb(174, 171, 158)",
      backgroundColor: isSelected ? "rgb(0, 130, 133)" : "rgb(232, 231, 228)",
      "&:hover": {
        backgroundColor: "rgb(1, 107, 112)",
        color: "white",
      },
      "&.Mui-disabled": {
        background: "rgb(227, 226, 221)",
        color: "rgb(120, 117, 101)",
      },
    }}
  >
    {children}
  </Button>
);

export default AmfToolActionButton;
