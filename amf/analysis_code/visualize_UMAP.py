import numpy as np
import os 
from sklearn.cluster import KMeans
from collections import Counter
from matplotlib.patches import Rectangle
from tensorflow.keras.models import Model, load_model
import tensorflow as tf
import matplotlib.pyplot as plt
import umap
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.widgets import Button
import cv2 
from mpl_toolkits.axes_grid1 import make_axes_locatable


def return_layer_activations_preserve_spatial_test(model, images, layer_name, layer_model):

    # Get activations for the entire batch at once
    activations = layer_model.predict(images)  # shape: (batch_size, H,W,Filters)
    flattened_activation = np.array([img.flatten() for img in activations])
    return flattened_activation

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
    n_clusters=3):  

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
    # 6. Place cluster stats at bottom, side by side
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
    # 7. Remove numeric ticks, label axes
    # ------------------------------------------------
    ax.set_xlabel("UMAP 1", fontsize=10)
    ax.set_ylabel("UMAP 2", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    # ------------------------------------------------
    # 8. Legend in top-left
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


# ------------------------------------------------
#HELPER GRAD-CAM FUNCTIONS
# ------------------------------------------------


def normalize(x):
    """Normalize image data to [0, 1]."""
    return x / 255.0

def make_gradcam_heatmap(img_array, model, layer_name="C4", pred_index=None):
    """
    Generate Grad-CAM heatmap for a single image.
    Expects img_array of shape (1, H, W, C) in float32.
    """
    # Create a model that maps the input to the activations of layer_name + final output
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])  # Use top predicted class if none provided
        class_channel = preds[:, pred_index]

    # Gradient of the class output w.r.t. feature map
    grads = tape.gradient(class_channel, conv_outputs)
    # Global average pooling on the gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    # Normalize between 0 and 1 for visualization
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap + 1e-16)
    return heatmap.numpy()

def overlay_gradcam(original_img, heatmap, alpha=0.0001):
    """
    Overlay a Grad-CAM heatmap onto the original image.
    - original_img: uint8 image, shape (H, W, 3) with values [0..255].
    - heatmap: float32 array, shape (H, W) with values [0..1].
    - alpha: blending factor for the heatmap.
    Returns a uint8 image of the same shape as original_img.
    """
    # Rescale heatmap to [0..255] and apply a colormap
    heatmap_255 = (heatmap * 255).astype('uint8')
    heatmap_colored = cv2.applyColorMap(heatmap_255, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Combine the heatmap with the original image
    overlay = cv2.addWeighted(original_img, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay



# -----------------------------------------------------
# Main UMAP + interactive function
# -----------------------------------------------------
def umap_plot(
    embeddings, 
    true_labels, 
    indices, 
    images, 
    index_labels, 
    model,
    predicted_labels=None, 
    p_index_labels=None
):
    """
    Plots a UMAP scatter plot for embeddings. Clicking a point shows:
      - The original image
      - The Grad-CAM heatmap
      - An overlay of Grad-CAM on the original image
    """

    classifications = ['Colonized', 'Non-colonized', 'No root']  # Adjust as needed
    color_map = {2: 'saddlebrown', 1: 'cornflowerblue', 0: 'teal'}
    point_colors = [color_map[label] for label in true_labels]

    # --- 1) Create UMAP scatter plot ---
    fig, ax = plt.subplots(figsize=(6, 6))
    scatter = ax.scatter(
        embeddings[:, 0], embeddings[:, 1], 
        c=point_colors, alpha=0.6, edgecolors='none'
    )
    highlighted_point = ax.scatter([], [], s=40, edgecolor='black', facecolor='none', label="Selected Point")

    # Legends
    color_legend = [
        mpatches.Patch(color=color, label=f"True Label: {classifications[i]}")
        for i, color in color_map.items()
    ]

    ax.legend(handles=color_legend, loc='best', title="Legend")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("UMAP Embeddings by True Labels")

    # --- 2) Create a figure with 3 subplots for the images ---
    fig_image, (ax_image, ax_grad, ax_overlay) = plt.subplots(1, 3, figsize=(9, 3))

    # Initialize left subplot (raw image)
    img_plot = ax_image.imshow(images[0].astype('uint8'))
    ax_image.axis('off')
    ax_image.set_title(f"True = {classifications[true_labels[0]]}")

    # Initialize center subplot (grad-cam heatmap)
    blank_heatmap = np.zeros((images[0].shape[0], images[0].shape[1]))
    grad_plot = ax_grad.imshow(blank_heatmap, cmap='jet', vmin=0, vmax=1)
    ax_grad.axis('off')
    ax_grad.set_title("Grad-CAM")

    # Initialize right subplot (overlay)
    overlay_plot = ax_overlay.imshow(images[0].astype('uint8'))
    ax_overlay.axis('off')
    ax_overlay.set_title("Overlay")

    gradcam_cbar = None  # Keep track of the existing colorbar
    # --- 3) Click handler ---
    def on_click(event):
        nonlocal gradcam_cbar  # Access the colorbar variable
        if event.xdata is None or event.ydata is None:
            return
        x, y = event.xdata, event.ydata

        # Find the closest point in embeddings
        distances = np.sqrt((embeddings[:, 0] - x) ** 2 + (embeddings[:, 1] - y) ** 2)
        closest_idx = np.argmin(distances)
        display_index = indices[closest_idx]

        # Update highlighted point on the scatter
        highlighted_point.set_offsets([[embeddings[closest_idx, 0], embeddings[closest_idx, 1]]])
        highlighted_point.set_visible(True)

        # --- 4) Update the raw image display ---
# --- 4) Update the raw image display ---
        tile = images[display_index].astype('uint8')
        true_label_text = f"True = {classifications[true_labels[display_index]]}"
        if predicted_labels is not None:
            pred_label_text = f"Predicted = {classifications[predicted_labels[display_index]]}"
            ax_image.set_title(f"{true_label_text}\n{pred_label_text}")  # Newline for better formatting
        else:
            ax_image.set_title(true_label_text)

        img_plot.set_data(tile)

        # --- 5) Generate Grad-CAM heatmap for this tile ---
        normalized_tile = normalize(np.array([tile], np.float32))  # shape (1, H, W, 3)
        heatmap = make_gradcam_heatmap(normalized_tile, model, layer_name="C4")

        # Update the Grad-CAM subplot
        grad_plot.set_data(heatmap)
        grad_plot.set_clim(vmin=0, vmax=1)  # Normalize
        ax_grad.set_title(f"Grad-CAM")

        # --- 6) Create and show the overlay image ---
        resized_heatmap = cv2.resize(heatmap, (tile.shape[1], tile.shape[0]))
        heatmap_255 = (resized_heatmap * 255).astype('uint8')
        heatmap_colored = cv2.applyColorMap(heatmap_255, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        overlaid_img = cv2.addWeighted(tile, 0.6, heatmap_colored, 0.4, 0)
        overlay_plot.set_data(overlaid_img)
        ax_overlay.set_title(f"Overlay")

        # Redraw both figures
        fig.canvas.draw()
        fig_image.canvas.draw()

    # Attach click event
    fig.canvas.mpl_connect('button_press_event', on_click)

    return fig, fig_image