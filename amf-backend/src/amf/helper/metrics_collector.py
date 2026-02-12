"""Functionality for collecting performance metrics during model evaluation."""

import pandas as pd


class MetricsCollector:
    """Container for performance metrics collected during model evaluation."""

    def __init__(self) -> None:
        """Initialise empty metrics."""
        self.per_class_metrics: dict[str, dict[str, float]] = {}
        self.generic_metrics: dict[str, float] = {}
        self.metrics_dataframes: dict[str, pd.DataFrame] = {}
        self.images: dict[str, str] = {}

    def add_class_metric(self, class_name: str, metric_name: str, value: float) -> None:
        """Add per-class metric.

        Args:
            class_name: Class name.
            metric_name: Metric name.
            value: Metric value.
        """
        if class_name not in self.per_class_metrics:
            self.per_class_metrics[class_name] = {}
        self.per_class_metrics[class_name][metric_name] = value

    def add_generic_metric(self, metric_name: str, value: float) -> None:
        """Add non-class-specific metric `metric_name` with value `value`."""
        self.generic_metrics[metric_name] = value

    def add_metrics_dataframe(self, df_name: str, dataframe: pd.DataFrame) -> None:
        """Add DataFrame of metrics with name `df_name`."""
        self.metrics_dataframes[df_name] = dataframe

    def add_image(self, image_name: str, image_data: str) -> None:
        """Add link to image (e.g. incorrect predictions mosaic) stored on disk.

        Args:
            image_name: Name of the image.
            image_data: Path to image.
        """
        self.images[image_name] = image_data

    def convert_to_dataframe(
        self,
        metrics_type: str = "class",  # TODO use enum?
    ) -> pd.DataFrame:
        """Get DataFrame of all metrics of given type, or an existing metrics DataFrame.

        Args:
            metrics_type: "class" for all per-class metrics, "generic" for all
                non-class-specific metrics, or the name of an existing metrics
                DataFrame.

        Returns: Appropriate DataFrame of metrics.

        Raises:
            ValueError: If the metrics_type is not either "class", "generic", or an
            existing metrics DataFrame name.
        """
        if metrics_type in ["class", "generic"]:
            data_dict = (
                self.per_class_metrics
                if metrics_type == "class"
                else self.generic_metrics
            )
            df = pd.DataFrame.from_dict(data_dict, orient="index").reset_index()
            df.rename(columns={"index": "Name"}, inplace=True)
            return df
        if metrics_type in self.metrics_dataframes:
            return self.metrics_dataframes[metrics_type]
        raise ValueError("Unknown metrics type or dataframe name.")
