const calcStdDev = (array) => {
  const n = array.length;
  if (n === 0) {
    return 0;
  }

  const mean = array.reduce((a, b) => a + b) / n;
  return Math.sqrt(
    array.map((x) => Math.pow(x - mean, 2)).reduce((a, b) => a + b) / n,
  );
};

// Helper method to give back index of max value in a list
const indexOfMax = (arr) => {
  if (arr.length === 0) {
    return -1;
  }

  var max = arr[0];
  var maxIndex = 0;

  for (var i = 1; i < arr.length; i++) {
    if (arr[i] > max) {
      maxIndex = i;
      max = arr[i];
    }
  }

  return maxIndex;
};

const isValidFilePath = (path) => {
  // Regular expression for a Windows file path with both forward and backward slashes
  let regex_windows = /^[a-zA-Z]:[\\/](?:[^\\/*<>?|"]+[\\/])*[^\\/*<>?|"]*$/;

  // Regular expression for a Linux file path
  let regex_linux = /^\/([^\0*<>?|"]+\/)*[^\0*<>?|"]*$/;

  return path !== "" && (regex_windows.test(path) || regex_linux.test(path));
};

const removeFileExtension = (filename) => {
  // Remove file extension to get name of file
  const lastDotIndex = filename.lastIndexOf(".");
  return lastDotIndex === -1 ? filename : filename.slice(0, lastDotIndex);
};

export { calcStdDev, indexOfMax, isValidFilePath, removeFileExtension };
