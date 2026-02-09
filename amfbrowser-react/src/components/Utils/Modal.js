import Fade from "@mui/material/Fade";
import { FaTimes } from "react-icons/fa";

const Modal = ({ isOpen, onClose, children, overrideStyles }) => {
  if (!isOpen) return null;

  let styles = overrideStyles ? { ...overrideStyles } : {};

  return (
    <Fade in={true}>
      <div className="modal-overlay" onClick={onClose}>
        <div
          className="modal-content"
          style={styles}
          onClick={(e) => e.stopPropagation()}
        >
          <FaTimes onClick={onClose} className="close-icon" />
          {children}
        </div>
      </div>
    </Fade>
  );
};

export default Modal;
