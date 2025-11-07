# AMFinder - amfinder_test.py
#
# Copyright (c) 2024-2025 Royal Botanic Gardens, Kew
#
# Evaluates the model on the test roots to obtain metrics.

import random

import numpy as np

random.seed(42)
import amfinder_config as AmfConfig
import amfinder_load as AmfLoad
import amfinder_log as AmfLog
import amfinder_save as AmfSave
from metrics_collector import MetricsCollector
from test_metrics import TestMetrics


def get_colonisation_results(
    x_test,  # Test images
    y_test,  # Labels for images
    filenames,  # filenames as required in load_data_and_files()
    results_dir,  # Directory to save results
    metrics_collector,  # Object to collect metrics
    rows,  # Rows for visualisation (if needed)
    cols,  # Cols for visualisation (if needed)
):
    """
    Method to calculate colonisation metrics only for a particular test set.
    """

    y_test_labels = y_test_labels = np.argmax(y_test, axis=1)

    metrics = TestMetrics(
        x_test=x_test,
        y_test=y_test,
        y_test_labels=y_test_labels,
        metrics_collector=metrics_collector,
        filenames=filenames,
        results_dir=results_dir,
        rows=rows,
        cols=cols,
    )

    AmfLog.info("Allocating metrics variables")
    metrics.get_perfile_metrics(colonised_only=True)

    AmfLog.info("Saving metrics function.")
    AmfSave.save_metrics(metrics.metrics_collector, results_dir)


def run(input_images):
    """
    Runs prediction on a bunch of images.

    :param input_images: input images to use for predictions.
    """

    AmfLog.info("Output colonisation percentage for a set of annotations.")
    metrics_collector = MetricsCollector()

    # Create timestamped folder for results
    results_dir = AmfConfig.get("outdir")
    AmfLog.info(f"Results Directory: {results_dir}")

    # Creating dataset
    dataset_loader = AmfLoad.TileFilesandData(input_images)
    x_test, y_test, filenames, rows, cols = dataset_loader.get_all_data()

    get_colonisation_results(
        x_test,
        y_test,
        filenames,
        results_dir,
        metrics_collector,
        rows,
        cols,
    )

    return 200
