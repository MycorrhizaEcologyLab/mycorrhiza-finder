import numpy as np
from scipy.stats import ttest_ind
from tensorflow.keras.models import Model, load_model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



def split_data_by_classification(true_labels, predicted_labels, data_array):
    # Initialize lists for correct and misclassified indices
    cor_indexes = []
    mis_indexes = []

    # Find correct and misclassified indices
    for i, (true, pred) in enumerate(zip(true_labels, predicted_labels)):
        if true == pred:
            cor_indexes.append(i)  # Append correct index
        else:
            mis_indexes.append(i)  # Append incorrect index

    # Split the data array
    cor_data = data_array[cor_indexes]
    mis_data = data_array[mis_indexes]
    cor_label = true_labels[cor_indexes]
    mis_true_label = true_labels[mis_indexes]
    mis_pred_label = predicted_labels[mis_indexes]
    del data_array
    return cor_data, mis_data, cor_label, mis_true_label, mis_pred_label

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