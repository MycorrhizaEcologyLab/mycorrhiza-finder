export default class PredictionsApi {
  // TODO clean these up, they're all very similar
  static async getImageFromBackend(url) {
    return await fetch(url).catch((error) => {
      console.error(error);
      return null;
    });
  }

  static async getFromBackend(url) {
    return await fetch(url)
      .then((response) => response.json())
      .catch((error) => {
        console.error(error);
        return null;
      });
  }

  static async postToBackend(url, body) {
    const req = new Request(url, {
      method: "POST",
      body: JSON.stringify(body),
      headers: {
        "Content-Type": "application/json",
      },
    });
    return await fetch(req)
      .then((response) => response.status)
      .catch((error) => {
        console.error(error);
        return null;
      });
  }

  static async postToBackendWithResponse(url, body) {
    const req = new Request(url, {
      method: "POST",
      body: JSON.stringify(body),
      headers: {
        "Content-Type": "application/json",
      },
    });
    return await fetch(req)
      .then((response) => response.json())
      .catch((error) => {
        console.error(error);
        return null;
      });
  }

  static async patchToBackend(url) {
    return await fetch(url, {
      method: "PATCH",
    })
      .then((response) => response.status)
      .catch((error) => {
        console.error(error);
        return null;
      });
  }

  static async deleteFromBackend(url) {
    return await fetch(url, {
      method: "DELETE",
    })
      .then((response) => response.status)
      .catch((error) => {
        console.error(error);
        return null;
      });
  }

  static async postImageToBackend(url, file) {
    var formData = new FormData();
    formData.append("file", file);
    try {
      const response = await fetch(url, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error("Error in postImageToBackend:", error);
      return null;
    }
  }

  static async postResizeToBackend(url, file, maxSize) {
    var formData = new FormData();
    formData.append("file", file);
    formData.append("maxSize", maxSize);

    try {
      const response = await fetch(url, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return response.blob();
    } catch (error) {
      console.error("Error in postResizeToBackend:", error);
      return null;
    }
  }

  ////// Predictions & annotations //////

  static async fetchAnnotationsByName(name, cnn, colonisation_type) {
    const fetchAnnotationsUrl = `http://127.0.0.1:8001/fetch-annotations-cnn-${cnn}?name=${name}&colonisation_type=${colonisation_type}`;
    return await this.getFromBackend(fetchAnnotationsUrl);
  }

  static async fetchPredictionsByName(name, cnn, colonisation_type) {
    const fetchPredictionsUrl = `http://127.0.0.1:8001/fetch-predictions-cnn-${cnn}?name=${name}&colonisation_type=${colonisation_type}`;
    return await this.getFromBackend(fetchPredictionsUrl);
  }

  static async fetchAnnotationsById(id, cnn, colonisation_type) {
    const fetchAnnotationsUrl = `http://127.0.0.1:8001/fetch-annotations-cnn-${cnn}?id_=${id}&colonisation_type=${colonisation_type}`;
    return await this.getFromBackend(fetchAnnotationsUrl);
  }

  static async fetchPredictionsById(id, cnn, colonisation_type) {
    const fetchPredictionsUrl = `http://127.0.0.1:8001/fetch-predictions-cnn-${cnn}?id_=${id}&colonisation_type=${colonisation_type}`;
    return await this.getFromBackend(fetchPredictionsUrl);
  }

  static async importFolder(request_body) {
    const importFolderUrl = `http://127.0.0.1:8001/import-folder`;
    return this.postToBackendWithResponse(importFolderUrl, request_body);
  }

  static async saveAnnotations(request_body, return_status = false) {
    const saveAnnotationsUrl = `http://127.0.0.1:8001/save-annotations`;
    if (return_status) {
      return this.postToBackend(saveAnnotationsUrl, request_body);
    }
    return this.postToBackendWithResponse(saveAnnotationsUrl, request_body);
  }

  static async savePredictions(request_body, return_status = false) {
    const savePredictionsUrl = `http://127.0.0.1:8001/save-predictions`;
    if (return_status) {
      return this.postToBackend(savePredictionsUrl, request_body);
    }
    return this.postToBackendWithResponse(savePredictionsUrl, request_body);
  }

  static async deleteImageReference(id) {
    const deleteImageReferenceUrl = `http://127.0.0.1:8001/delete-image-reference/${id}`;
    return this.deleteFromBackend(deleteImageReferenceUrl);
  }

  static async setToEnabled(id) {
    const setToEnabledUrl = `http://127.0.0.1:8001/set-to-enabled/${id}`;
    return this.patchToBackend(setToEnabledUrl);
  }

  ////// AMFinder tools //////

  static async calculatePredictions(request_body) {
    const calculatePredictionsUrl = `http://127.0.0.1:8001/calculate-predictions`;
    return await this.postToBackend(calculatePredictionsUrl, request_body);
  }

  static async trainModel(request_body) {
    const trainModelUrl = `http://127.0.0.1:8001/train-model`;
    return await this.postToBackend(trainModelUrl, request_body);
  }

  static async testModel(request_body) {
    const testModelUrl = `http://127.0.0.1:8001/test-model`;
    return await this.postToBackend(testModelUrl, request_body);
  }

  static async calculateColonisation(request_body) {
    const colonisationUrl = `http://127.0.0.1:8001/calculate-colonisation`;
    return await this.postToBackend(colonisationUrl, request_body);
  }

  static async convertImages(request_body) {
    const conversionUrl = `http://127.0.0.1:8001/convert-images`;
    return await this.postToBackend(conversionUrl, request_body);
  }

  static async tifConversion(request_body) {
    const tifConversion = `http://127.0.0.1:8001/tif-conversion`;
    return await this.postToBackend(tifConversion, request_body);
  }

  static async calibrateModel(request_body) {
    const calibrationUrl = `http://127.0.0.1:8001/calibrate-model`;
    return await this.postToBackend(calibrationUrl, request_body);
  }

  static async saveSettings(request_body) {
    const saveUrl = "http://127.0.0.1:8001/save-settings";
    return await this.postToBackend(saveUrl, request_body);
  }

  static async getAllSettings() {
    const getAllSettingsUrl = "http://127.0.0.1:8001/get-all-settings";
    return await this.getFromBackend(getAllSettingsUrl);
  }

  static async getSettingsSchema() {
    const getSettingsSchemaUrl = "http://127.0.0.1:8001/settings-schema";
    return await this.getFromBackend(getSettingsSchemaUrl);
  }

  static async revertSettingToDefault(request_body) {
    const revertUrl = "http://127.0.0.1:8001/revert-setting-to-default";
    return await this.postToBackendWithResponse(revertUrl, request_body);
  }

  static async revertAllSettingsToDefault(request_body) {
    const revertUrl = "http://127.0.0.1:8001/revert-all-settings-to-default";
    return await this.postToBackend(revertUrl, request_body);
  }

  ////// Sub-tag palette //////

  static async getTagPalette() {
    const tagPaletteUrl = "http://127.0.0.1:8001/tag-palette";
    return await this.getFromBackend(tagPaletteUrl);
  }

  static async saveTagPalette(colours) {
    const tagPaletteUrl = "http://127.0.0.1:8001/tag-palette";
    return await this.postToBackendWithResponse(tagPaletteUrl, { colours });
  }

  ////// Download //////
  static async getImageNames() {
    const getImageNamesUrl = "http://127.0.0.1:8001/get-image-names";
    return await this.getFromBackend(getImageNamesUrl);
  }

  static async checkEntriesForImage(name, colonisation_type) {
    const checkEntriesForImageUrl = `http://127.0.0.1:8001/check-entries-for-image?name=${name}&colonisation_type=${colonisation_type}`;
    return await this.getFromBackend(checkEntriesForImageUrl);
  }

  static async checkEntriesForId(id, colonisation_type) {
    const checkEntriesForIdUrl = `http://127.0.0.1:8001/check-entries-for-id?id_=${id}&colonisation_type=${colonisation_type}`;
    return await this.getFromBackend(checkEntriesForIdUrl);
  }

  static async downloadEntries(id, type, colonisation_type) {
    const downloadEntriesUrl = `http://127.0.0.1:8001/download-entries?id_=${id}&type_=${type}&colonisation_type=${colonisation_type}`;
    return await this.getFromBackend(downloadEntriesUrl);
  }

  ////// Image fetching //////
  static async tileImage(file) {
    const uploadImageUrl = "http://127.0.0.1:8001/tile-image";
    return await this.postImageToBackend(uploadImageUrl, file);
  }

  static async getImageTile(startIndex, batchSize) {
    const getImageTileUrl = `http://127.0.0.1:8001/get-image-tile?startIndex=${startIndex}&batchSize=${batchSize}`;
    return await this.getImageFromBackend(getImageTileUrl);
  }

  static async getResizedImage(file, maxSize) {
    const getResizedImageUrl = `http://127.0.0.1:8001/resize-image`;
    return await this.postResizeToBackend(getResizedImageUrl, file, maxSize);
  }

  static async setTileEdgeBackend(body) {
    const setTileEdgeUrl = `http://127.0.0.1:8001/set-tile-edge`;
    return await this.postToBackend(setTileEdgeUrl, body);
  }

  static async getBackgroundTiles(threshold) {
    const getBackgroundTilesUrl = `http://127.0.0.1:8001/background-tiles?threshold=${threshold}`;
    return await this.getFromBackend(getBackgroundTilesUrl);
  }

  static async checkModelArchitecture(path) {
    const checkModelArchitectureUrl = `http://127.0.0.1:8001/model-architecture?path=${encodeURIComponent(path)}`;
    return await this.getFromBackend(checkModelArchitectureUrl);
  }

  static async getTrainedModels() {
    const getTrainedModelsUrl = "http://127.0.0.1:8001/trained-models";
    return await this.getFromBackend(getTrainedModelsUrl);
  }
}
