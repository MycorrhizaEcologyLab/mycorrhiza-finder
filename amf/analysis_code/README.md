# AMFinder v1 analysis scripts

## Summary

The scripts and jupyter notebooks in this folder were used to carry out analysis on the first trained iteration of AMFinder, which can be found under legacy_trained_networks/252_tile.h5.

Below is a high-level description of each analysis file. You can leverage the accompanying jupyter notebooks to run these analysis scripts and produce meaningful outputs.

# visualize_UMAP.py

This script extracts and analyzes activations from AMFinder to study patterns in its classification of stained root images. It processes dataset activations, applies dimensionality reduction using UMAP, and clusters embeddings with K-Means. Additionally, it includes an interactive UMAP plot to click on datapoints to see the input image and its Grad-CAM.

# visualize_activations.py

This script provides tools to visualizeactivations from AMFinder. It allows users to plot static filter activations for specific layers, compare RGB channel responses. Additionally, it allows for visualization of the activations of a specific layer for a specific input image. These visualizations help in understanding feature extraction and the role of different filters in model predictions.

# pure_purple_analysis.py

This script analyzes the activations of convolutional layers in AMFinder, focusing on identifying "pure purple" filters (uniform activations). It includes functions for separating correctly and incorrectly classified samples, performing statistical tests on activation distributions, and visualizing the frequency of non-informative filters.
