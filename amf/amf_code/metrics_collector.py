# AMFinder - metrics_collector.py
#
# Copyright (c) 2024-2025 Royal Botanic Gardens, Kew
#
# Collects Metrics during model evaluation to be saved to the folder
# containing the test root files loaded for evaluation.

import pandas as pd


class MetricsCollector:
    def __init__(self):
        self.per_class_metrics = {}
        self.generic_metrics = {}
        self.metrics_dataframes = {}
        self.images = {}

    def add_class_metric(self, class_name, metric_name, value):
        if class_name not in self.per_class_metrics:
            self.per_class_metrics[class_name] = {}
        self.per_class_metrics[class_name][metric_name] = value

    def add_generic_metric(self, metric_name, value):
        self.generic_metrics[metric_name] = value

    def add_metrics_dataframe(self, df_name, dataframe):
        self.metrics_dataframes[df_name] = dataframe

    def add_image(self, image_name, image_data):
        self.images[image_name] = image_data

    def convert_to_dataframe(self, metrics_type="class"):
        """Converts stored metrics to a pandas DataFrame."""
        if metrics_type in ["class", "generic"]:
            data_dict = (
                self.per_class_metrics
                if metrics_type == "class"
                else self.generic_metrics
            )
            df = pd.DataFrame.from_dict(data_dict, orient="index").reset_index()
            df.rename(columns={"index": "Name"}, inplace=True)
            return df
        elif metrics_type in self.metrics_dataframes:
            return self.metrics_dataframes[metrics_type]
        else:
            raise ValueError("Unknown metrics type or dataframe name.")
