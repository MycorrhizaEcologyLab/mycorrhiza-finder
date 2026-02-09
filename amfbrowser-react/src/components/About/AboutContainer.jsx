import combinedLogo from "../../assets/combined-logo.png";
import "./styles/about.css";

const AboutContainer = () => {
  return (
    <div id="aboutContainer" className="fullHeight fullWidth flexRowCenter">
      <div
        className="fullHeight fullWidth flexColumnCenter"
        style={{ overflowY: "auto" }}
      >
        <div
          className="fullHeight fullWidth"
          style={{
            minWidth: "750px",
            maxWidth: "1500px",
            minHeight: "350px",
          }}
        >
          <div className="fullHeight fullWidth">
            <div
              id="topHalf"
              className="flexColumnCenter"
              style={{
                width: "100%",
                height: "80%",
                minHeight: "420px",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              <div className="fullHeight fullWidth flexColumnCenter">
                <div
                  className="flexColumnCenter"
                  style={{
                    width: "70%",
                    minWidth: "600px",
                    maxWidth: "1200px",
                    textAlign: "center",
                    fontSize: "16px",
                  }}
                >
                  <img
                    style={{ width: "400px" }}
                    alt={"Combined logo"}
                    src={combinedLogo}
                    className="noselect about-logo"
                  />
                  <p>
                    {`This version of the MycorrhizaFinder tool has been developed by the Royal
                      Botanical Gardens Kew through the `}
                    <a
                      href="https://www.gov.uk/government/publications/natural-capital-and-ecosystem-assessment-programme/natural-capital-and-ecosystem-assessment-programme"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      Natural Capital and Ecosystem Assessment (NCEA)
                    </a>{" "}
                    programme.{" "}
                  </p>
                  <p>
                    The NCEA is Defra's largest research and development
                    programme, it is generating a robust evidence base of the
                    location, extent and condition of our natural capital and
                    ecosystems, and how the state of nature is changing over
                    time.
                  </p>
                  <p>
                    {" "}
                    The original tool was developed by the Sainsbury Laboratory,
                    University of Cambridge.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AboutContainer;
