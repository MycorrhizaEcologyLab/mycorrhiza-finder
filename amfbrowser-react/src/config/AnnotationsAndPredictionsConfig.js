// headerToValueMapCnn1
export const getHeaderToValueMapCnn1 = (colonisationType) => {
  return colonisationType === "am"
    ? headerToValueMapCnn1Am
    : headerToValueMapCnn1Erm;
};

const headerToValueMapCnn1Am = new Map([
  ["AMColonised", "AM+"],
  ["Uncolonised", "N-"],
  ["Background", "X"],
  ["Unreadable", "U"],
  ["DSE", "D"],
  ["Hybrid", "H"],
  ["Question", "?"],
]);

const headerToValueMapCnn1Erm = new Map([
  ["BlueCoils", "Bl+"],
  ["BrownCoils", "Br+"],
  ["TypeTwo", "T+"],
  ["Uncolonised", "N-"],
  ["Background", "X"],
  ["MainRoot", "M"],
  ["Unreadable", "U"],
  ["DSE", "D"],
  ["HybridErm", "HE+"],
  ["HybridDse", "HD+"],
  ["Question", "?"],
]);

// colorMappingCnn1
export const getColorMappingCnn1 = (colonisationType) => {
  return colonisationType === "am" ? colorMappingCnn1Am : colorMappingCnn1Erm;
};

const colorMappingCnn1Am = new Map([
  ["AM+", "#002060"],
  ["N-", "#e8c775"],
  ["X", "black"],
  ["U", "#03F0FC"],
  ["D", "#F84646"],
  ["H", "#7030A0"],
  ["?", "#FFA500"],
]);

const colorMappingCnn1Erm = new Map([
  ["Bl+", "#002060"],
  ["Br+", "#824201"],
  ["T+", "#04bd4e"],
  ["N-", "#e8c775"],
  ["X", "black"],
  ["M", "#c40041"],
  ["U", "#03F0FC"],
  ["D", "#F84646"],
  ["HE+", "#c1d602"],
  ["HD+", "#7030A0"],
  ["?", "#FFA500"],
]);

// colorMappingCnn1Transparent
export const getColorMappingCnn1Transparent = (colonisationType) => {
  return colonisationType === "am"
    ? colorMappingCnn1TransparentAm
    : colorMappingCnn1TransparentErm;
};
const colorMappingCnn1TransparentAm = new Map([
  ["AM+", "rgb(0, 32, 96, 0.6)"],
  ["N-", "rgb(232, 199, 117, 0.6)"],
  ["X", "rgb(0, 0, 0, 0.6)"],
  ["U", "rgb(3, 240, 252, 0.6)"],
  ["D", "rgb(248, 70, 70, 0.6)"],
  ["H", "rgb(112, 48, 160, 0.6)"],
  ["?", "rgb(255,165,0, 0.6)"],
]);

const colorMappingCnn1TransparentErm = new Map([
  ["Bl+", "rgb(0, 32, 96, 0.6)"],
  ["Br+", "rgb(130, 66, 1, 0.6)"],
  ["T+", "rgb(4, 189, 78, 0.6)"],
  ["N-", "rgb(232, 199, 117, 0.6)"],
  ["X", "rgb(0, 0, 0, 0.6)"],
  ["M", "rgb(196, 0, 65, 0.6)"],
  ["U", "rgb(3, 240, 252, 0.6)"],
  ["D", "rgb(248, 70, 70, 0.6)"],
  ["HE+", "rgb(193, 214, 2, 0.6)"],
  ["HD+", "rgb(112, 48, 160, 0.6)"],
  ["?", "rgb(255,165,0, 0.6)"],
]);

// headerMapCnn1
export const getHeaderMapCnn1 = (colonisationType) => {
  return colonisationType === "am" ? headerMapCnn1Am : headerMapCnn1Erm;
};
const headerMapCnn1Am = [
  "AMColonised",
  "Uncolonised",
  "Background",
  "Unreadable",
  "DSE",
  "Hybrid",
  "Question",
];

const headerMapCnn1Erm = [
  "BlueCoils",
  "BrownCoils",
  "TypeTwo",
  "Uncolonised",
  "Background",
  "MainRoot",
  "Unreadable",
  "DSE",
  "HybridErm",
  "HybridDse",
  "Question",
];

// valueMapCnn1
export const getValueMapCnn1 = (colonisationType) => {
  return colonisationType === "am" ? valueMapCnn1Am : valueMapCnn1Erm;
};
const valueMapCnn1Am = ["AM+", "N-", "X", "U", "D", "H", "?"];
const valueMapCnn1Erm = [
  "Bl+",
  "Br+",
  "T+",
  "N-",
  "X",
  "M",
  "U",
  "D",
  "HE+",
  "HD+",
  "?",
];

export const getKeyBindingsCnn1 = (colonisationType) => {
  return colonisationType === "am"
    ? headerToKeyBindingsMapCnn1Am
    : headerToKeyBindingsMapCnn1Erm;
};

const headerToKeyBindingsMapCnn1Am = new Map([
  ["AM+", "A"],
  ["N-", "N"],
  ["X", "X"],
  ["U", "U"],
  ["D", "D"],
  ["H", "H"],
  ["?", "? or /"],
]);

const headerToKeyBindingsMapCnn1Erm = new Map([
  ["Bl+", "B"],
  ["Br+", "R"],
  ["T+", "T"],
  ["N-", "N"],
  ["X", "X"],
  ["M", "M"],
  ["U", "U"],
  ["D", "D"],
  ["HE+", "H"],
  ["HD+", "Y"],
  ["?", "? or /"],
]);

/* Sub-tags -------------------------------------------------------------- */

// Separator packing multiple sub-tags into one CSV cell. Must match
// TAG_DELIMITER in the backend's config.py.
export const TAG_DELIMITER = "|";

// Fallback colours, handed out in order to tags that have no stored colour
// (e.g. an annotation file opened on a machine with no saved palette). Chosen
// to stay distinguishable from the class colours above.
export const DEFAULT_TAG_COLOURS = [
  "#1b998b",
  "#e07a5f",
  "#8367c7",
  "#dd6e42",
  "#2a9d8f",
  "#c9184a",
  "#457b9d",
  "#b56576",
  "#6a994e",
  "#9c6644",
];

export const parseTags = (value) => {
  if (value === null || value === undefined) {
    return [];
  }
  const text = String(value).trim();
  if (text === "" || text.toLowerCase() === "nan") {
    return [];
  }
  const tags = [];
  text.split(TAG_DELIMITER).forEach((part) => {
    const tag = part.trim();
    if (tag !== "" && !tags.includes(tag)) {
      tags.push(tag);
    }
  });
  return tags;
};

// Packs sub-tags into a single delimited cell. Round-tripping through
// parseTags trims, de-duplicates and drops empties; the trailing join is what
// turns the result back into a string. Must stay equivalent to format_tags in
// the backend's api_utils.py - a row element has to be a string, not a list.
export const formatTags = (tags) =>
  parseTags((tags ?? []).join(TAG_DELIMITER)).join(TAG_DELIMITER);

// Resolves a tag's colour: the stored palette wins, otherwise a stable slot in
// the default cycle based on the tag's position in the known tag list.
export const getTagColour = (tag, palette, allTags) => {
  if (palette && palette[tag]) {
    return palette[tag];
  }
  const idx = Math.max(0, (allTags ?? []).indexOf(tag));
  return DEFAULT_TAG_COLOURS[idx % DEFAULT_TAG_COLOURS.length];
};
