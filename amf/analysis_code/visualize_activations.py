# AMFinder - visualize_activations.py
#
# MIT License
# Copyright (c) 2021 Edouard Evangelisti, Carl Turner
#               2024-2025 Royal Botanic Gardens, Kew
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.

"""
Compute visualized activations

Global
------------
i dont know what to put here

Functions
------------
:function visualize_layer_activations: plots all filter activations for a given model, layer, input file, indexed image
:function static_filter_activation: plot the R, G, B, RGB filter activations of all filters in a given model, layer.
:function return_layer_activations: returns activations for each tile
"""

from tensorflow.keras.models import Model, load_model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import glob

def visualize_layer_activations(model, image, index, layer_name):

    # Create a sub-model up to the specified layer
    layer_output_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

    # Get activations
    activations = layer_output_model.predict(image)
    activations = activations[index]

    num_filters = activations.shape[-1]
    cols = 8
    rows = num_filters // cols + (1 if num_filters % cols != 0 else 0)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    if rows > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    pure_purple_count = 0

    for i in range(num_filters):
        ax = axes[i]
        act_map = activations[:, :, i]

        # Check if all values are identical (pure purple)
        act_min, act_max = act_map.min(), act_map.max()
        if act_max == act_min:
            # This means the filter has no variation
            pure_purple_count += 1

        # Normalize for visualization
        if act_max != act_min:
            act_map = (act_map - act_min) / (act_max - act_min)

        ax.imshow(act_map, cmap='viridis')
        ax.axis('off')
        ax.set_title(f"Filter {i}")

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    print(f"Number of filters with no informative activation (pure purple): {pure_purple_count}")

    plt.tight_layout()
    plt.show()


def static_filter_activations(model,layer_name):
    filters, biases = model.get_layer(name=layer_name).get_weights()
    num_filters = filters.shape[3]
    num_channels = filters.shape[2]  # Typically 3 for RGB

    rows = num_filters
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))

    for i in range(num_filters):
        f = filters[:, :, :, i]
        f_min, f_max = f.min(), f.max()
        f_norm = (f - f_min) / (f_max - f_min) if f_max != f_min else f

        axes[i, 0].imshow(f_norm[:, :, 0], cmap='Reds')
        axes[i, 0].axis('off')
        axes[i, 0].set_title(f"Filter {i}: R")

        axes[i, 1].imshow(f_norm[:, :, 1], cmap='Greens')
        axes[i, 1].axis('off')
        axes[i, 1].set_title("G")

        axes[i, 2].imshow(f_norm[:, :, 2], cmap='Blues')
        axes[i, 2].axis('off')
        axes[i, 2].set_title("B")

        axes[i, 3].imshow(f_norm)
        axes[i, 3].axis('off')
        axes[i, 3].set_title("RGB")

    fig.suptitle(f"Filter Visualizations for Layer: {layer_name}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()


def return_GAP_layer_activations(model, image, index, layer_name):
    # Create a sub-model up to the specified layer
    layer_output_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

    # Get activations for the batch of images
    activations = layer_output_model.predict(image)

    # Extract the activation for the given 'index'
    # activations[index] should now be a 3D array: (H, W, num_filters)
    single_activation = activations[index]

    # Global average pool the activations to produce a single vector per image
    # This will result in a 1D array of length = num_filters
    act_vector = np.mean(single_activation, axis=(0, 1))

    # Return the flattened activation vector suitable for t-SNE
    return act_vector


import os
import numpy as np

def return_layer_activations_preserve_spatial_test(model, images, layer_name, layer_model):

    # Get activations for the entire batch at once
    activations = layer_model.predict(images)  # shape: (batch_size, H,W,Filters)
    flattened_activation = np.array([img.flatten() for img in activations])
    return flattened_activation

import os
import numpy as np
import umap
import matplotlib.pyplot as plt

import umap

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.widgets import Button
import cv2
from mpl_toolkits.axes_grid1 import make_axes_locatable
#
def normalize(x):
    """Normalize image data to [0, 1]."""
    return x / 255.0

def umap_embeddings_from_disk(activations_file_name, n_components=2, random_state=100):
    """
    Load all activations from a disk file, fit UMAP, and transform.

    Note: This reads all data into memory at once. Ensure you have
    sufficient RAM for the entire activation dataset.
    """
    # Step 1 & 2: Read and concatenate all stored activations
    all_activations = []
    with open(activations_file_name, 'rb') as f:
        while True:
            try:
                batch = np.load(f)
                all_activations.append(batch)
            except EOFError:
                break
    all_activations = np.vstack(all_activations)

    # Step 3 & 4: Create UMAP, fit, and transform
    reducer = umap.UMAP(n_components=n_components, random_state=random_state)
    embeddings = reducer.fit_transform(all_activations)

    return reducer, embeddings


def plot_clustered_umap(
    embeddings,
    true_labels,
    predicted_labels,
    class_names,
    images=None,
    cluster_by='true_labels',
    n_clusters=3,
    highlight_clusters=None):


    from sklearn.cluster import KMeans
    from collections import Counter
    from matplotlib.patches import Rectangle

    """
    1. KMeans on 2D embeddings.
    2. Sort clusters left->right by center.x (Cluster 0,1,2).
    3. For each cluster, derive its 'major label' from either:
       - True labels (if cluster_by='true_labels')
       - Predicted labels (if cluster_by='predicted_labels')
    4. Plot:
       - Points colored by cluster (saddlebrown, cornflowerblue, teal)
       - Red 'X' at each center, labeled with the major label above it
    5. Stats at the bottom (side by side):
       - Vertical listing of all True Labels & Predicted Labels
    6. Legend in top-left, no numeric ticks, axes labeled 'UMAP 1', 'UMAP 2'
    """

    # ------------------------
    # 1. KMeans on embeddings
    # ------------------------
    kmeans = KMeans(n_clusters=n_clusters, random_state=100)
    original_cluster_labels = kmeans.fit_predict(embeddings)
    centers = kmeans.cluster_centers_

    unique_clusters = np.unique(original_cluster_labels)

    # ------------------------------------------------
    # 2. Sort clusters left->right by center.x
    #    => rename them 0..1..2 in that order
    # ------------------------------------------------
    cluster_ids_sorted_by_x = sorted(unique_clusters, key=lambda c: centers[c, 0])

    # Map old cluster IDs -> new (0..1..2)
    new_label_map = {}
    for new_id, old_id in enumerate(cluster_ids_sorted_by_x):
        new_label_map[old_id] = new_id

    # Re-labeled cluster assignments
    cluster_labels = np.array([new_label_map[old_lbl]
                               for old_lbl in original_cluster_labels])

    # Reorder centers to match new labeling
    sorted_centers = centers[cluster_ids_sorted_by_x]

    # ------------------------------------------------
    # 3. Compute label distributions & major label
    # ------------------------------------------------
    cluster_distributions = {}
    major_labels = {}  # e.g. {0: 'Colonized', 1: 'No-root', etc.}

    for new_id in range(n_clusters):
        old_id = cluster_ids_sorted_by_x[new_id]
        mask = (original_cluster_labels == old_id)

        # Tally the true/pred labels for points in this old cluster
        true_counts = Counter(true_labels[mask])
        pred_counts = Counter(predicted_labels[mask])

        cluster_distributions[new_id] = {
            'true_label_counts': dict(true_counts),
            'pred_label_counts': dict(pred_counts),
        }

        # Decide which distribution to use for the "major label"
        # based on cluster_by
        if cluster_by == 'true_labels':
            ref_counts = true_counts
        else:
            ref_counts = pred_counts

        if len(ref_counts) > 0:
            major_label_idx = max(ref_counts, key=ref_counts.get)
            major_label_name = class_names[major_label_idx]
        else:
            major_label_name = "N/A"
        major_labels[new_id] = major_label_name

    # ------------------------------------------------
    # 4. Plot points with custom colors
    # ------------------------------------------------
    # Cluster 0 => saddlebrown
    # Cluster 1 => cornflowerblue (lighter blue)
    # Cluster 2 => teal
    custom_colors = ["saddlebrown", "cornflowerblue", "teal"]
    point_colors = [custom_colors[new_lbl] for new_lbl in cluster_labels]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.scatter(
        embeddings[:, 0],
        embeddings[:, 1],
        c=point_colors,
        marker='o',
        s=80,
        alpha=0.6,
        edgecolors='none'
    )

    # ------------------------------------------------
    # 5. Plot centers & label with the major label
    # ------------------------------------------------
    for new_id in range(n_clusters):
        cx, cy = sorted_centers[new_id]
        ax.scatter(cx, cy, c='red', marker='X', s=200, edgecolors='black')
        ax.text(
            cx,
            cy + 0.15,  # small offset
            major_labels[new_id],  # e.g. 'No-root'
            ha='center',
            va='bottom',
            fontsize=9,
            color='black'
        )

    # ------------------------------------------------
    # 6. Optionally highlight clusters
    # ------------------------------------------------
    if highlight_clusters is not None:
        for hc in highlight_clusters:
            mask = (cluster_labels == hc)
            c_pts = embeddings[mask]
            if c_pts.size == 0:
                continue
            x_min, y_min = c_pts.min(axis=0)
            x_max, y_max = c_pts.max(axis=0)
            rect = Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=2,
                edgecolor='black',
                facecolor='none'
            )
            ax.add_patch(rect)

    # ------------------------------------------------
    # 7. Place cluster stats at bottom, side by side
    # ------------------------------------------------
    # Increase bottom margin to make space
    plt.subplots_adjust(bottom=0.35)

    # x positions for n_clusters
    x_positions = np.linspace(0.20, 0.80, n_clusters)

    for new_id, x_pos in zip(range(n_clusters), x_positions):
        dist_info = cluster_distributions[new_id]
        t_counts = dist_info['true_label_counts']
        p_counts = dist_info['pred_label_counts']

        lines = [
            f"Cluster {new_id} (Major: {major_labels[new_id]})",
            "True Labels:"
        ]
        for lbl, cnt in sorted(t_counts.items()):
            lines.append(f"  {class_names[lbl]}: {cnt}")

        lines.append("Predicted Labels:")
        for lbl, cnt in sorted(p_counts.items()):
            lines.append(f"  {class_names[lbl]}: {cnt}")

        info_text = "\n".join(lines)

        fig.text(
            x_pos, 0.05,
            info_text,
            ha='center',
            va='bottom',
            fontsize=8
        )

    # ------------------------------------------------
    # 8. Remove numeric ticks, label axes
    # ------------------------------------------------
    ax.set_xlabel("UMAP 1", fontsize=10)
    ax.set_ylabel("UMAP 2", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    # ------------------------------------------------
    # 9. Legend in top-left
    # ------------------------------------------------
    legend_handles = []
    for i in range(n_clusters):
        legend_handles.append(
            plt.Line2D(
                [0], [0],
                color=custom_colors[i],
                marker='o',
                linestyle='None',
                markersize=8,
                label=f"Cluster {i} (Major: {major_labels[i]})"
            )
        )
    ax.legend(handles=legend_handles, loc='upper left', fontsize=8)

    ax.set_title(f"UMAP Clustering (by '{cluster_by}')")

    plt.show()

    return cluster_labels, cluster_distributions

def predict_labels(model):
    base_dir = "Preprocessed_data/Data"
    output_dir = "Preprocessed_data/Labels/Predicted"
    os.makedirs(output_dir, exist_ok=True)

    # Recursive glob search for x_*.npy files
    file_paths = glob.glob(os.path.join(base_dir, "**", "x_*.npy"), recursive=True)

    for file_path in sorted(file_paths):
        # Load data
        print(f"Processing: {file_path}")
        x = np.load(file_path)

        # Predict labels
        prediction = model.predict(x, batch_size=32)

        # Generate the output file path
        file_name = os.path.basename(file_path)  # Extract the file name
        predicted_name = file_name.replace("x_", "y_").replace(".npy", "_predicted.npy")
        output_path = os.path.join(output_dir, predicted_name)

        # Save predictions
        np.save(output_path, prediction)
        print(f"Saved: {output_path}")

        # Free memory
        del x
        del prediction

def test_save_activations_to_disk(
    data_list,
    labels_list,
    model,
    layer_name,
    layer_model,
    save_name,
    batch_size=100,
    reduce_non_col=1,
    reduce_no_root=1,
    random_seed = 100
):
    """
    Processes multiple datasets as a single dataset, trims classes, and saves
    activations + indices to disk. By unifying the data first, the number of
    colonized samples remains consistent.

    Parameters:
        data_list (list of numpy.ndarray): List of input data arrays.
        labels_list (list of numpy.ndarray): List of true labels (1D arrays).
        model: The complete model.
        layer_name (str): Name of the layer for activations.
        layer_model: Layer-specific model (if needed for activation computation).
        save_name (str): Base name for saving activations and indices.
        batch_size (int): Number of samples per batch.
        reduce_non_col (float): Fraction of non-colonized class to keep [0,1].
        reduce_no_root (float): Fraction of no-root class to keep [0,1].
    """

    # 1. Combine all data/labels into unified arrays
    combined_data = np.concatenate(data_list, axis=0)
    combined_labels = np.concatenate(labels_list, axis=0)

    # Class definitions
    COLONIZED_CLASS = 0
    NON_COLONIZED_CLASS = 1
    NO_ROOT_CLASS = 2

    # 2. Separate indices by class (unshuffled)
    idx_colonized = np.where(combined_labels == COLONIZED_CLASS)[0]
    idx_non_col   = np.where(combined_labels == NON_COLONIZED_CLASS)[0]
    idx_no_root   = np.where(combined_labels == NO_ROOT_CLASS)[0]

    # 3. Shuffle only if we’re reducing those classes
    #    to get a random subset
    np.random.seed(random_seed)
    if reduce_non_col < 1.0:
        np.random.shuffle(idx_non_col)
    if reduce_no_root < 1.0:
        np.random.shuffle(idx_no_root)
    # Optional: shuffle colonized if you want to randomly subsample it
    # np.random.shuffle(idx_colonized)

    # 4. Keep specified fraction of each class
    keep_non_col = idx_non_col[: int(reduce_non_col * len(idx_non_col))]
    keep_no_root = idx_no_root[: int(reduce_no_root * len(idx_no_root))]

    # 5. Combine the reduced sets (still unshuffled at this point)
    reduced_indices = np.concatenate([idx_colonized, keep_non_col, keep_no_root])

    # 6. Save a copy of these combined reduced indices in their original order
    #    Sort them to strictly preserve ascending order from the original dataset
    #    (Alternatively, skip sorting if you prefer the concatenation order.)
    reduced_indices = np.sort(reduced_indices)
    np.save(
        f"Preprocessed_data/activations/Test_activations/{save_name}_indices.npy",
        reduced_indices
    )

    # 8. Filter the data based on the shuffled reduced indices
    filtered_data = combined_data[reduced_indices]

    # 9. Extract layer activations in batches
    all_activations = []
    num_samples = len(filtered_data)
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_data = filtered_data[start_idx:end_idx]

        # Compute layer activations
        activations = return_layer_activations_preserve_spatial_test(
            model=model,
            images=batch_data,
            layer_name=layer_name,
            layer_model=layer_model
        )

        # Flatten activations and accumulate
        flattened_activation = np.array([act.flatten() for act in activations])
        all_activations.append(flattened_activation)

    all_activations = np.vstack(all_activations)

    # 10. Save final activations and their shuffled indices
    #     (Use 'reduced_indices' if you want to know the order used in the final UMAP)
    np.save(
        f"Preprocessed_data/activations/Test_activations/{save_name}_activations.npy",
        all_activations
    )

    print(
        f"Activations saved to Preprocessed_data/activations/Test_activations/{save_name}_activations.npy"
    )
    print(
        f"Original reduced indices saved to Preprocessed_data/activations/Test_activations/{save_name}_original_reduced_indices.npy")

from scipy.stats import ttest_ind


def analyze_no_root_filter_statistics(model, layer_name, data_1, data_2, data_1_title, data_2_title, bins, title):
    """
    Analyze the number of "pure purple" (identical activation) filters predicting "no-root"
    for a given layer and perform statistical tests to compare correct vs misclassified samples.

    Parameters:
        model: The model containing the layer to analyze.
        layer_name: The name of the layer to analyze.
        correct_data (numpy.ndarray): Input data for correctly classified samples.
                                      Shape: (num_correct_samples, height, width, channels).
        misclassified_data (numpy.ndarray): Input data for misclassified samples.
                                            Shape: (num_misclassified_samples, height, width, channels).

    Returns:
        dict: A dictionary containing:
              - "t_stat": t-statistic of the t-test
              - "p_value": p-value of the t-test
              - "statistical_significance": Whether the difference is statistically significant
              - "correct_purple_counts": List of (raw or ratio) purple counts for correct samples
              - "misclassified_purple_counts": List of (raw or ratio) purple counts for misclassified samples
              - "correct_mean": Mean number/ratio of pure purple filters for correct samples
              - "misclassified_mean": Mean number/ratio of pure purple filters for misclassified samples
              - "correct_std": Standard deviation for correct samples
              - "misclassified_std": Standard deviation for misclassified samples
    """

    # Helper to get activations from the specified layer
    def get_layer_activations(data):
        layer_output_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
        return layer_output_model.predict(data)

    # Count how many filters in each sample are "pure purple" (all activations identical)
    def count_pure_purple_filters(activations):
        """
        activations shape: (num_samples, H, W, num_filters)
        For each sample, we check each filter (H x W slice).
        If min == max, that filter is "pure purple" (same activation everywhere).
        """
        purple_counts = []
        for sample_activations in activations:
            count = 0
            # Shape is (H, W, num_filters), so transpose to (num_filters, H, W)
            for filter_activation in sample_activations.transpose(2, 0, 1):
                if np.min(filter_activation) == np.max(filter_activation):
                    count += 1
            purple_counts.append(count)
        return purple_counts

    # 1) Get activations for correct and misclassified data
    correct_activations = get_layer_activations(data_1)
    misclassified_activations = get_layer_activations(data_2)

    # 2) Count pure purple filters for both correct and misclassified samples
    correct_purple_counts = count_pure_purple_filters(correct_activations)
    misclassified_purple_counts = count_pure_purple_filters(misclassified_activations)

    # 3) (Optional) Convert raw counts to fraction of filters
    #    This ensures we compare "fraction of pure purple filters" across samples
    #    Instead of raw counts. Note: correct_activations.shape[-1] is total filters.
    #total_filters = correct_activations.shape[-1]  # or you can keep them as raw counts
    #correct_purple_counts = [c / total_filters for c in correct_purple_counts]
    #misclassified_purple_counts = [c / total_filters for c in misclassified_purple_counts]

    # 4) Compute statistics
    correct_mean = np.mean(correct_purple_counts)
    correct_std  = np.std(correct_purple_counts)
    misclassified_mean = np.mean(misclassified_purple_counts)
    misclassified_std  = np.std(misclassified_purple_counts)

    t_stat, p_value = ttest_ind(correct_purple_counts, misclassified_purple_counts)

    significance_threshold = 0.05
    statistical_significance = p_value < significance_threshold

    print("\nStatistical Analysis of No-Root Filters:")
    print(f"Correctly Classified - Mean: {correct_mean:.2f}, Std: {correct_std:.2f}")
    print(f"Misclassified       - Mean: {misclassified_mean:.2f}, Std: {misclassified_std:.2f}")
    print(f"T-test - t-statistic: {t_stat:.2f}, p-value: {p_value:.4f}")
    if statistical_significance:
        print("The difference is statistically significant (p < 0.05).")
    else:
        print("The difference is not statistically significant (p >= 0.05).")

    # 5) Plot histograms.
    #    By default, hist() shows raw counts, but `density=True` will normalize the area to 1.
    #    The x-axis is the fraction of pure purple filters per sample, y-axis is frequency (or density).
    plt.figure()
    plt.hist(correct_purple_counts, bins=bins, alpha=0.7, label=data_1_title, density=True)
    plt.hist(misclassified_purple_counts, bins=bins, alpha=0.7, label=data_2_title, density=True)
    plt.xlabel("Fraction of Pure Purple Filters")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.savefig(f'{title}.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # 6) Return results in a dictionary
    return {
        "t_stat": t_stat,
        "p_value": p_value,
        "statistical_significance": statistical_significance,
        "correct_purple_counts": correct_purple_counts,
        "misclassified_purple_counts": misclassified_purple_counts,
        "correct_mean": correct_mean,
        "misclassified_mean": misclassified_mean,
        "correct_std": correct_std,
        "misclassified_std": misclassified_std
    }

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from scipy.stats import ttest_ind


def find_pure_purple_filters(data, model, layer_name, threshold):
    """
    Find filters that activate as "pure purple" (all activations identical) above a given threshold.

    Parameters:
        - data: Input data to pass through the model.
        - model: The trained model.
        - layer_name: The name of the layer to inspect.
        - threshold: A float between 0 and 1 representing the activation threshold.

    Returns:
        A dictionary where keys are filter indices and values are the percentage of images
        where the filter activates as "pure purple" above the threshold.
    """
    # Step 1: Get activations for the specified layer
    layer_output_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    activations = layer_output_model.predict(data)

    # Step 2: Initialize a counter array for pure purple activations
    num_filters = activations.shape[-1]
    pure_purple_counts = np.zeros(num_filters)

    # Step 3: Iterate through each sample's activations and count pure purple filters
    for sample_activations in activations:
        # Transpose to (num_filters, H, W)
        for filter_idx, filter_activation in enumerate(sample_activations.transpose(2, 0, 1)):
            if np.min(filter_activation) == np.max(filter_activation):  # Check for "pure purple"
                pure_purple_counts[filter_idx] += 1

    # Step 4: Calculate the percentage of "pure purple" activations per filter
    total_samples = activations.shape[0]
    pure_purple_percentages = pure_purple_counts / total_samples

    # Step 5: Identify filters that exceed the threshold
    result = {filter_idx: pct for filter_idx, pct in enumerate(pure_purple_percentages) if pct >= threshold}

    return result

    """
    Plots histograms of how many 'pure-purple' filters
    (max == min across spatial dimensions) are present in
    the specified convolution layer outputs for two datasets.

    :param model: Keras model
    :param layer_name: Name of the layer from which to extract outputs
    :param data1: NumPy array of shape (N, H, W, C) for the first dataset
    :param data2: (Optional) NumPy array of shape (M, H, W, C) for the second dataset
                  If provided, we'll plot two histograms and compare them statistically.
    :return: A dictionary with statistics about the two distributions.
    """

    # 1) Build a sub-model that outputs the given layer's activations
    layer_output_model = Model(inputs=model.input,
                               outputs=model.get_layer(layer_name).output)

    # 2) Get activations for dataset 1
    activations_data1 = layer_output_model.predict(data1)
    # activations_data1 shape -> (N, outH, outW, num_filters)

    # Count "pure-purple" filters for each image in data1
    # A "pure-purple" filter means np.max == np.min for that filter's 2D slice
    # For example: for filter_idx in range(num_filters):
    #    we check if all values in (outH x outW) are the same.
    num_images_1 = activations_data1.shape[0]
    num_filters = activations_data1.shape[-1]

    pp_counts_1 = []  # will hold integer count of pure-purple filters for each image
    for i in range(num_images_1):
        image_activations = activations_data1[i]  # shape: (outH, outW, num_filters)
        count_pure_purple = 0
        for f in range(num_filters):
            filter_activation = image_activations[..., f]
            if np.max(filter_activation) == np.min(filter_activation):
                count_pure_purple += 1
        pp_counts_1.append(count_pure_purple)

    pp_counts_1 = np.array(pp_counts_1)

    # (Optional) repeat for dataset 2
    pp_counts_2 = None
    if data2 is not None:
        activations_data2 = layer_output_model.predict(data2)
        num_images_2 = activations_data2.shape[0]

        pp_counts_2 = []
        for i in range(num_images_2):
            image_activations = activations_data2[i]  # shape: (outH, outW, num_filters)
            count_pure_purple = 0
            for f in range(num_filters):
                filter_activation = image_activations[..., f]
                if np.max(filter_activation) == np.min(filter_activation):
                    count_pure_purple += 1
            pp_counts_2.append(count_pure_purple)
        pp_counts_2 = np.array(pp_counts_2)

    # 3) Plot histograms
    plt.figure(figsize=(8, 4))
    plt.hist(pp_counts_1, bins=range(0, num_filters + 1), alpha=0.5,
             label='Dataset 1', edgecolor='black')

    if pp_counts_2 is not None:
        plt.hist(pp_counts_2, bins=range(0, num_filters + 1), alpha=0.5,
                 label='Dataset 2', edgecolor='black')

    plt.xlabel('Number of Pure-Purple Filters per Image')
    plt.ylabel('Frequency')
    plt.title(f'Pure-Purple Filter Distribution in Layer: {layer_name}')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 4) Compute and print stats
    stats_dict = {}
    stats_dict['mean_data1'] = np.mean(pp_counts_1).item()

    if pp_counts_2 is not None:
        stats_dict['mean_data2'] = np.mean(pp_counts_2).item()
        # Simple t-test
        t_stat, p_val = ttest_ind(pp_counts_1, pp_counts_2, equal_var=False)
        stats_dict['t_stat'] = t_stat.item()
        stats_dict['p_val'] = p_val.item()

    return stats_dict