import AddIcon from "@mui/icons-material/Add";
import CloseIcon from "@mui/icons-material/Close";
import FormLabel from "@mui/material/FormLabel";
import IconButton from "@mui/material/IconButton";
import { useState } from "react";
import {
  DEFAULT_TAG_COLOURS,
  getTagColour,
} from "../../../config/AnnotationsAndPredictionsConfig";

/**
 * Sub-tag panel for the annotations left sidebar.
 *
 * Tags are additional to a tile's single class label - a user can attach any
 * number of them to mark features they want to find again later.
 * Clicking a tag toggles it on the currently selected tile.
 */
const AnnotationsTagPanel = ({
  allTags,
  tagPalette,
  selectedTileTags,
  hasSelectedClass,
  onCreateTag,
  onDeleteTag,
  onToggleTag,
  onClearTags,
  setTagInputOpen,
  isOpen,
}) => {
  const [newTagName, setNewTagName] = useState("");
  // Seeded from the default cycle so a fresh tag is not always the same colour.
  const [newTagColour, setNewTagColour] = useState(
    DEFAULT_TAG_COLOURS[0],
  );

  const handleAdd = () => {
    const name = newTagName.trim();
    if (name === "") {
      return;
    }
    onCreateTag(name, newTagColour);
    setNewTagName("");
    setNewTagColour(
      DEFAULT_TAG_COLOURS[(allTags.length + 1) % DEFAULT_TAG_COLOURS.length],
    );
  };

  return (
    <div
      id="sidebarTags"
      style={{ padding: "20px 18px", flexShrink: 0, minWidth: 0 }}
    >
      <FormLabel
        sx={{
          display: "flex",
          paddingBottom: "12px",
          color: "black",
          fontWeight: 600,
          fontSize: "12px",
        }}
      >
        Tags
      </FormLabel>

      {/* Create a new tag: name + colour */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "6px",
          marginBottom: "10px",
        }}
      >
        <input
          type="text"
          value={newTagName}
          placeholder="New tag"
          title="Name a new tag, then pick its colour and press +"
          // Annotation hotkeys are suppressed while this has focus, otherwise
          // typing a name would relabel the selected tile.
          onFocus={() => setTagInputOpen(true)}
          onBlur={() => setTagInputOpen(false)}
          onChange={(e) => setNewTagName(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              handleAdd();
            }
          }}
          style={{
            flex: 1,
            minWidth: 0,
            fontSize: "12px",
            padding: "4px 6px",
            borderRadius: "6px",
            border: "1px solid rgb(174, 171, 158)",
          }}
        />
        <input
          type="color"
          value={newTagColour}
          title="Colour for this tag"
          onChange={(e) => setNewTagColour(e.target.value)}
          style={{
            width: "26px",
            height: "26px",
            padding: 0,
            border: "1px solid rgb(174, 171, 158)",
            borderRadius: "6px",
            background: "none",
            cursor: "pointer",
            flexShrink: 0,
          }}
        />
        <IconButton
          size="small"
          onClick={handleAdd}
          disabled={newTagName.trim() === ""}
          title="Add tag"
          sx={{ padding: "2px", flexShrink: 0 }}
        >
          <AddIcon sx={{ fontSize: "18px" }} />
        </IconButton>
      </div>

      {allTags.length === 0 && (
        <div style={{ fontSize: "11px", color: "rgb(120, 117, 101)" }}>
          No tags yet.
        </div>
      )}

      {/* Existing tags - click to toggle on the selected tile */}
      {allTags.map((tag) => {
        const colour = getTagColour(tag, tagPalette, allTags);
        const active = selectedTileTags.includes(tag);
        return (
          <div
            key={tag}
            style={{
              display: "flex",
              alignItems: "center",
              gap: "4px",
              marginBottom: "5px",
              minWidth: 0,
            }}
          >
            <div
              className="noselect"
              onClick={() => hasSelectedClass && onToggleTag(tag)}
              title={
                hasSelectedClass
                  ? `${active ? "Remove" : "Add"} "${tag}" on the selected tile`
                  : "Label this tile with a class before tagging it"
              }
              style={{
                flex: 1,
                minWidth: 0,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                height: "26px",
                padding: "0 8px",
                borderRadius: "13px",
                fontSize: "12px",
                color: "white",
                backgroundColor: colour,
                opacity: hasSelectedClass ? (active ? 1 : 0.55) : 0.3,
                outline: active ? "2px solid red" : "1px solid darkgray",
                cursor: hasSelectedClass ? "pointer" : "not-allowed",
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
              }}
            >
              {tag}
            </div>
            {isOpen && (
              <IconButton
                size="small"
                onClick={() => onDeleteTag(tag)}
                title={`Delete "${tag}" everywhere`}
                sx={{ padding: "2px", flexShrink: 0 }}
              >
                <CloseIcon sx={{ fontSize: "14px" }} />
              </IconButton>
            )}
          </div>
        );
      })}

      {allTags.length > 0 && (
        <div
          className="noselect"
          onClick={() => selectedTileTags.length > 0 && onClearTags()}
          title="Remove every tag from the selected tile"
          style={{
            marginTop: "10px",
            height: "26px",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            borderRadius: "13px",
            fontSize: "12px",
            border: "1px solid rgb(174, 171, 158)",
            color:
              selectedTileTags.length > 0
                ? "rgb(60, 59, 50)"
                : "rgb(160, 157, 145)",
            backgroundColor:
              selectedTileTags.length > 0 ? "rgb(232, 231, 228)" : "transparent",
            cursor: selectedTileTags.length > 0 ? "pointer" : "not-allowed",
          }}
        >
          Clear tile tags
        </div>
      )}
    </div>
  );
};

export default AnnotationsTagPanel;
