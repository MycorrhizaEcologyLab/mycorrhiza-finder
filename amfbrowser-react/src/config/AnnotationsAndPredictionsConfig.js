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
  ["AM_DSE", "AD+"],
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
  ["ErM", "E+"],
  ["ErM_DSE", "ED+"],
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
  ["AD+", "#7030A0"],
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
  ["E+", "#c1d602"],
  ["ED+", "#7030A0"],
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
  ["AD+", "rgb(112, 48, 160, 0.6)"],
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
  ["E+", "rgb(193, 214, 2, 0.6)"],
  ["ED+", "rgb(112, 48, 160, 0.6)"],
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
  "AM_DSE",
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
  "ErM",
  "ErM_DSE",
  "Question",
];

// valueMapCnn1
export const getValueMapCnn1 = (colonisationType) => {
  return colonisationType === "am" ? valueMapCnn1Am : valueMapCnn1Erm;
};
const valueMapCnn1Am = ["AM+", "N-", "X", "U", "D", "AD+", "?"];
const valueMapCnn1Erm = [
  "Bl+",
  "Br+",
  "T+",
  "N-",
  "X",
  "M",
  "U",
  "D",
  "E+",
  "ED+",
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
  ["AD+", "H"],
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
  ["E+", "E"],
  ["ED+", "H"],
  ["?", "? or /"],
]);
