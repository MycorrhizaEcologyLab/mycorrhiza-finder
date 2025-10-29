import { NavLink } from "react-router-dom";
import { styled } from "@mui/material/styles";
import MuiDrawer from "@mui/material/Drawer";
import Divider from "@mui/material/Divider";
import IconButton from "@mui/material/IconButton";
import ChevronLeftIcon from "@mui/icons-material/ChevronLeft";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";
import ListItem from "@mui/material/ListItem";
import ListItemButton from "@mui/material/ListItemButton";
import ListItemIcon from "@mui/material/ListItemIcon";
import ListItemText from "@mui/material/ListItemText";
import InfoOutlinedIcon from "@mui/icons-material/InfoOutlined";
import VisibilityOutlinedIcon from "@mui/icons-material/VisibilityOutlined";
import SettingsOutlinedIcon from "@mui/icons-material/SettingsOutlined";
import BuildOutlinedIcon from "@mui/icons-material/BuildOutlined";
import OnlinePredictionOutlinedIcon from "@mui/icons-material/OnlinePredictionOutlined";
import FileUploadOutlinedIcon from "@mui/icons-material/FileUploadOutlined";

import combinedLogo from "../../assets/combined-logo.png";

import "./styles/Sidebar.css";

const drawerWidth = 200;

const openedMixin = (theme) => ({
  width: drawerWidth,
  transition: theme.transitions.create("width", {
    easing: theme.transitions.easing.sharp,
    duration: theme.transitions.duration.enteringScreen,
  }),
  overflowX: "hidden",
});

const closedMixin = (theme) => ({
  transition: theme.transitions.create("width", {
    easing: theme.transitions.easing.sharp,
    duration: theme.transitions.duration.leavingScreen,
  }),
  overflowX: "hidden",
  width: `calc(${theme.spacing(7)} + 1px)`,
  [theme.breakpoints.up("sm")]: {
    width: `calc(${theme.spacing(8)} + 1px)`,
  },
});

const DrawerHeader = styled("div")(({ theme }) => ({
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  padding: theme.spacing(0, 1),
}));

const Drawer = styled(MuiDrawer, {
  shouldForwardProp: (prop) => prop !== "open",
})(({ theme }) => ({
  width: drawerWidth,
  flexShrink: 0,
  whiteSpace: "nowrap",
  boxSizing: "border-box",
  variants: [
    {
      props: ({ open }) => open,
      style: {
        ...openedMixin(theme),
        "& .MuiDrawer-paper": openedMixin(theme),
      },
    },
    {
      props: ({ open }) => !open,
      style: {
        ...closedMixin(theme),
        "& .MuiDrawer-paper": closedMixin(theme),
      },
    },
  ],
}));

const DrawerListItem = ({ text, linkPath, open, children }) => (
  <NavLink
    className="sidebarLink"
    to={linkPath}
    style={{
      textDecoration: "none",
      color: "black",
    }}
  >
    <ListItem key={text} disablePadding sx={{ display: "block" }}>
      <ListItemButton
        sx={[
          {
            minHeight: 48,
            px: 2.5,
          },
          open
            ? {
                justifyContent: "initial",
              }
            : {
                justifyContent: "center",
              },
        ]}
      >
        <ListItemIcon
          sx={[
            {
              minWidth: 0,
              justifyContent: "center",
            },
            open
              ? {
                  mr: 3,
                }
              : {
                  mr: "auto",
                },
          ]}
        >
          {children}
        </ListItemIcon>
        <ListItemText
          primary={text}
          sx={[
            open
              ? {
                  opacity: 1,
                }
              : {
                  opacity: 0,
                },
          ]}
        />
      </ListItemButton>
    </ListItem>
  </NavLink>
);

const Sidebar = ({ open, handleSidebarChange }) => {
  return (
    <Drawer id="drawer" variant="permanent" open={open}>
      <DrawerHeader>
        <IconButton onClick={handleSidebarChange}>
          {open ? <ChevronLeftIcon /> : <ChevronRightIcon />}
        </IconButton>
      </DrawerHeader>
      <Divider />

      <div className="sidebarHeader">Analytics</div>

      {/* Analytics */}
      <DrawerListItem text="Browser" linkPath="/browser" open={open}>
        <VisibilityOutlinedIcon />
      </DrawerListItem>
      <Divider style={{ marginTop: "10px" }} />

      {/* Finder */}
      <div className="sidebarHeader">Finder</div>
      <DrawerListItem text="MF Tool" linkPath="/amfTool" open={open}>
        <BuildOutlinedIcon />
      </DrawerListItem>
      <DrawerListItem text="Existing" linkPath="/existing" open={open}>
        <OnlinePredictionOutlinedIcon />
      </DrawerListItem>
      <DrawerListItem text="Upload" linkPath="/upload" open={open}>
        <FileUploadOutlinedIcon />
      </DrawerListItem>
      <Divider style={{ marginTop: "10px" }} />

      {/* Other */}
      <div className="sidebarHeader">Other</div>
      <DrawerListItem text="About" linkPath="/about" open={open}>
        <InfoOutlinedIcon />
      </DrawerListItem>
      <DrawerListItem text="Settings" linkPath="/settings" open={open}>
        <SettingsOutlinedIcon />
      </DrawerListItem>
      <Divider />

      <div
        id="lowerDrawerWrapper flexRow"
        style={{
          flexGrow: 1, // Div takes up remaining height of the drawer
          justifyContent: "center",
          alignItems: "end",
        }}
      >
        <div
          className="fullHeight fullWidth flexColumn"
          style={{
            justifyContent: "end",
            alignItems: "center",
            paddingTop: "20px",
          }}
        >
          <div
            className="flexColumn"
            style={{
              height: "89px",
              alignItems: "center",
              justifyContent: "end",
              paddingTop: "50px",
            }}
          >
            <img
              alt={"Combined logo"}
              src={combinedLogo}
              style={{
                width: "200px",
                height: "44px",
                opacity: open ? 1 : 0,
              }}
              className="noselect"
            />
            <div
              style={{
                marginTop: "20px",
                marginBottom: "10px",
                fontSize: "10px",
              }}
            >
              v5.0.0
            </div>
          </div>
        </div>
      </div>

      <Divider />
    </Drawer>
  );
};

export default Sidebar;
