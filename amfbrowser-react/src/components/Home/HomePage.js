import { useNavigate } from "react-router-dom";
import PrimaryButton from "../Utils/PrimaryButton";
import combinedLogo from "../../assets/combined-logo.png";

export default function HomePage() {
  const navigate = useNavigate();
  const selectImageRoute = () => {
    let path = "browser";
    navigate(path);
  };

  const amfRoute = () => {
    let path = "amfTool";
    navigate(path);
  };

  return (
    <div id="homePage" className="fullHeight fullWidth flexRowCenter">
      <div
        className="fullHeight fullWidth flexColumnCenter"
        style={{ overflowY: "auto" }}
      >
        <div
          className="fullHeight fullWidth flexColumnCenter"
          style={{
            minHeight: "250px",
          }}
        >
          <div className="fullHeight fullWidth flexColumnCenter">
            <div
              className="flexRow"
              style={{ height: "45px", fontSize: "17px" }}
            >
              Select a page to get started:
            </div>
            <div className="flexRow">
              <PrimaryButton
                sx={{ width: "150px", marginRight: "15px" }}
                onClick={selectImageRoute}
              >
                Browser
              </PrimaryButton>
              <PrimaryButton
                sx={{ width: "150px", marginLeft: "15px" }}
                onClick={amfRoute}
              >
                MF Tool
              </PrimaryButton>
            </div>
            <img
              style={{
                width: "150px",
                marginTop: "40px",
                marginBottom: "0px",
              }}
              alt={"Combined logo"}
              src={combinedLogo}
              className="noselect about-logo"
            />
          </div>
        </div>
      </div>
    </div>
  );
}
