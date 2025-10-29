import { useState, useEffect } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { ToastContainer, Slide } from "react-toastify";
import Box from "@mui/material/Box";
import CssBaseline from "@mui/material/CssBaseline";
import { GlobalContextProvider } from "./contexts/Contexts";

import Sidebar from "./components/PageStructure/Sidebar";
import HeaderBanner from "./components/PageStructure/HeaderBanner";
import HomePage from "./components/Home/HomePage";
import SettingsContainer from "./components/Settings/SettingsContainer";
import AmfToolContainer from "./components/AmfTool/AmfToolContainer";
import ExistingContainer from "./components/Existing/ExistingContainer";
import AmfUpload from "./components/Upload/AmfUpload";
import BrowserContainer from "./components/Browser/BrowserContainer";
import AnnotationsAndPredictionsContainer from "./components/AnnotationsAndPredictions/Common/AnnotationsAndPredictionsContainer";
import AboutContainer from "./components/About/AboutContainer";
import PredictionsApi from "./api/amfinderApi";

import "./App.css";
import "./components/Utils/styles/utilStyles.css";

export default function App() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [colonisationType, setColonisationType] = useState("am");
  const [tileEdge, setTileEdge] = useState(
    colonisationType === "am" ? 252 : 126,
  );
  const [settings, setSettings] = useState({});
  const [amfToolFilePath, setAmfToolFilePath] = useState("");

  const [selectedImage, setSelectedImage] = useState(null);
  const [selectedImageName, setSelectedImageName] = useState(null);

  const handleSidebarChange = () => {
    setIsSidebarOpen((isOpen) => !isOpen);
  };

  const fetchInitialSettings = async () => {
    const initialSettings = await PredictionsApi.getAllSettings();
    setSettings(initialSettings);
  };

  useEffect(() => {
    fetchInitialSettings();
  }, []);

  return (
    <GlobalContextProvider.Provider
      value={{
        colonisationType,
        setColonisationType,
        tileEdge,
        setTileEdge,
        settings,
        setSettings,
        amfToolFilePath,
        setAmfToolFilePath,
        selectedImage,
        setSelectedImage,
        selectedImageName,
        setSelectedImageName,
      }}
    >
      <Box sx={{ display: "flex" }}>
        <CssBaseline />
        <Router>
          <Sidebar
            open={isSidebarOpen}
            handleSidebarChange={handleSidebarChange}
          />

          <Box
            component="main"
            sx={{
              flexGrow: 1,
              display: "flex",
              flexDirection: "column",
              justifyContent: "center",
              overflowX: "auto",
            }}
          >
            <HeaderBanner
              colonisationType={colonisationType}
              setColonisationType={setColonisationType}
            />
            <div id="mainWindow">
              <Routes>
                <Route path="/" element={<HomePage />} />

                {/* Analytics */}
                <Route path="/browser" element={<BrowserContainer />} />
                <Route
                  path="/browser/:imageName/:colonisationType/annotations"
                  caseSensitive
                  element={<AnnotationsAndPredictionsContainer />}
                />
                <Route
                  path="/browser/:imageName/:colonisationType/annotations/:id"
                  caseSensitive
                  element={<AnnotationsAndPredictionsContainer />}
                />
                <Route
                  path="/browser/:imageName/:colonisationType/predictions"
                  caseSensitive
                  element={<AnnotationsAndPredictionsContainer />}
                />
                <Route
                  path="/browser/:imageName/:colonisationType/predictions/:id"
                  caseSensitive
                  element={<AnnotationsAndPredictionsContainer />}
                />

                {/* Finder */}
                <Route
                  path="/amfTool"
                  caseSensitive
                  element={<AmfToolContainer />}
                />
                <Route
                  path="/existing"
                  caseSensitive
                  element={<ExistingContainer />}
                />
                <Route path="/upload" caseSensitive element={<AmfUpload />} />

                {/* Other */}
                <Route
                  path="/about"
                  caseSensitive
                  element={<AboutContainer />}
                />
                <Route
                  path="/settings"
                  caseSensitive
                  element={<SettingsContainer />}
                />
              </Routes>
            </div>
          </Box>
        </Router>
      </Box>
      {/* ToastContainer (for default notification settings) */}
      <ToastContainer
        position="bottom-right"
        theme="colored"
        transition={Slide}
        draggable
      />
    </GlobalContextProvider.Provider>
  );
}
