# AMFinder - test_metrics.py
#
# Copyright (c) 2024-2025 Royal Botanic Gardens, Kew
#
# Collates metrics and produces visualisations of results after running test.
#

import math
import os
import random
from typing import Any, cast

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# For intermediate images
import seaborn as sns
from loguru import logger
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix

import amf.helper.config as AmfConfig

# import amf.helper.log as AmfLog
from amf.helper.metrics_collector import MetricsCollector

random.seed(42)


class TestMetrics:
    def __init__(
        self,
        x_test: list[NDArray[np.uint8]],
        y_test: list[NDArray[np.uint8]],
        y_test_labels: NDArray[np.int_],
        metrics_collector: MetricsCollector,
        filenames: list[str],
        results_dir: str,
        predicted_labels: NDArray[np.uint8] | None = None,
        predictions: NDArray[np.int64] | None = None,
        predicted_probs: NDArray[np.float32] | None = None,
        rows: list[int] | None = None,
        cols: list[int] | None = None,
    ) -> None:
        self.x_test = x_test
        self.y_test = y_test
        self.y_test_labels = y_test_labels
        self.predicted_labels = predicted_labels
        self.predictions = predictions
        self.predicted_probs = predicted_probs
        self.filenames = filenames
        self.results_dir = results_dir
        self.x_test = x_test
        self.rows = rows
        self.cols = cols
        self.metrics_collector = metrics_collector
        self.colonisation_type = AmfConfig.get("colonisation_type")
        self.class_names = AmfConfig.get("class_names")[self.colonisation_type]

    def safe_divide(
        self, numerator: NDArray[np.float64], denominator: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # Where denominator is zero, return 0, otherwise perform division
        return np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=float),
            where=(denominator != 0),
        )

    def get_conf_matrix(self) -> None:
        if self.predicted_labels is None:
            raise ValueError("Predicted labels must not be None")
        mapped_predictions = [
            self.class_names[label] for label in self.predicted_labels
        ]
        mapped_true_labels = [self.class_names[label] for label in self.y_test_labels]

        # Generate confusion matrix with class names
        cm = confusion_matrix(
            mapped_true_labels, mapped_predictions, labels=self.class_names
        )
        # Plotting
        plt.figure(figsize=(8, 6))
        sns.set(font_scale=1.4)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            annot_kws={"size": 18},
        )
        plt.title("Confusion Matrix", size=18)
        plt.ylabel("Actual Class", size=16)
        plt.xlabel("Predicted Class", size=16)
        save_path = os.path.join(self.results_dir, "confusion_matrix.png")
        plt.savefig(save_path, bbox_inches="tight")
        # self.metrics_collector.add_image(
        #     "confusion_matrix.png", save_path
        # )

        accuracy = self.safe_divide(np.diag(cm), np.sum(cm, axis=1))
        precision = self.safe_divide(np.diag(cm), np.sum(cm, axis=0))
        recall = self.safe_divide(np.diag(cm), np.sum(cm, axis=1))
        f1_score = 2 * self.safe_divide(precision * recall, precision + recall)

        # Handle division by zero for accuracy, precision, recall, and F1
        accuracy = np.nan_to_num(accuracy, nan=0.0)
        precision = np.nan_to_num(precision, nan=0.0)
        recall = np.nan_to_num(recall, nan=0.0)
        f1_score = np.nan_to_num(f1_score, nan=0.0)

        # Print Macro F1 Score
        macro_acc = cast(float, np.average(accuracy))
        macro_precision = cast(float, np.average(precision))
        macro_recall = cast(float, np.average(recall))
        macro_f1 = cast(float, np.average(f1_score))
        self.metrics_collector.add_generic_metric("Macro accuracy", macro_acc)
        self.metrics_collector.add_generic_metric("Macro precision", macro_precision)
        self.metrics_collector.add_generic_metric("Macro recall", macro_recall)
        self.metrics_collector.add_generic_metric("Macro f1 score", macro_f1)
        print(f"Macro F1 Score: {macro_f1 * 100:.2f}%")

        # Print per-class metrics
        print("\nPer-Class Metrics:")
        for class_name, acc, prec, rec, f1 in zip(
            self.class_names, accuracy, precision, recall, f1_score
        ):
            print(f"{class_name}:")
            print(f"  Accuracy: {acc * 100:.2f}%")
            print(f"  Precision: {prec * 100:.2f}%")
            print(f"  Recall: {rec * 100:.2f}%")
            print(f"  F1 Score: {f1 * 100:.2f}%")

            # Add metrics to metrics collector
            self.metrics_collector.add_class_metric(f"{class_name}", "Accuracy", acc)
            self.metrics_collector.add_class_metric(f"{class_name}", "Precision", prec)
            self.metrics_collector.add_class_metric(f"{class_name}", "Recall", rec)
            self.metrics_collector.add_class_metric(f"{class_name}", "F1 Score", f1)

    def get_perfile_metrics(self, colonised_only: bool = False) -> None:
        """
        Gets the metrics separately for each file

        """

        def calculate_am_colonised_percentage(
            series: pd.Series,
        ) -> tuple[float | None, float | None, float | None, float | None]:
            """Calculate % colonised for a series of labels for AM colonisation."""
            class_counts = series.value_counts()
            am_colonised = class_counts.get(0, 0) + class_counts.get(
                5, 0
            )  # Gets the number of AMColonised and Hybrid tiles
            dse_colonised = class_counts.get(4, 0) + class_counts.get(
                5, 0
            )  # Gets the number of DSE and Hybrid tiles
            total_colonised = (
                am_colonised + dse_colonised - class_counts.get(5, 0)
            )  # Number of Hybrid tiles are substracted to avaoid summing it two times
            # Remove background and unreadable
            total_root = (
                class_counts.sum() - class_counts.get(2, 0) - class_counts.get(3, 0)
            )

            perc_am_col = (am_colonised / total_root) * 100 if total_root > 0 else None
            perc_dse_col = (
                (dse_colonised / total_root) * 100 if total_root > 0 else None
            )
            perc_total_col = (
                (total_colonised / total_root) * 100 if total_root > 0 else None
            )
            perc_root = (
                (total_root / class_counts.sum()) * 100
                if class_counts.sum() > 0
                else None
            )
            return perc_am_col, perc_dse_col, perc_total_col, perc_root

        def calculate_erm_colonised_percentage(
            series: pd.Series,
        ) -> tuple[
            float | None,
            float | None,
            float | None,
            float | None,
            float | None,
            float | None,
        ]:
            """Calculate % ErM colonised for a series of labels."""
            class_counts = series.value_counts()
            blue_coil = class_counts.get(0, 0)
            brown_coil = class_counts.get(1, 0)
            type_two = class_counts.get(2, 0)
            dse = class_counts.get(7, 0)
            hybrid_erm = class_counts.get(8, 0)
            hybrid_dse = class_counts.get(9, 0)
            # Remove background, main root and unreadable
            total_root = (
                class_counts.sum()
                - class_counts.get(4, 0)
                - class_counts.get(5, 0)
                - class_counts.get(6, 0)
            )

            perc_blue_coil = None
            perc_brown_coil = None
            perc_type_two = None
            perc_dse = None
            perc_total_col = None
            if total_root > 0:
                perc_blue_coil = (blue_coil / total_root) * 100
                perc_brown_coil = (brown_coil / total_root) * 100
                perc_type_two = (type_two / total_root) * 100
                perc_dse = ((dse + hybrid_dse) / total_root) * 100
                perc_total_col = (
                    (blue_coil + brown_coil + type_two + hybrid_erm) / total_root
                ) * 100

            perc_root = (
                (total_root / class_counts.sum()) * 100
                if class_counts.sum() > 0
                else None
            )

            return (
                perc_blue_coil,
                perc_brown_coil,
                perc_type_two,
                perc_dse,
                perc_total_col,
                perc_root,
            )

        def calculate_accuracy_for_class(
            group: pd.Series, class_label: int
        ) -> float | None:
            """Calculate accuracy for a specific class within a grouped object."""
            correct_predictions = (
                group["Actual Label"] == group["Predicted Label"]
            ) & (group["Actual Label"] == class_label)
            total_instances = group["Actual Label"] == class_label
            if total_instances.sum() == 0:  # Avoid division by zero
                return None
            return cast(
                float, (correct_predictions.sum() / total_instances.sum()) * 100
            )

        def calculate_precision_for_class(group: pd.Series, class_label: int) -> float:
            """Calculate precision for a specific class within a grouped object."""
            tp_predictions = (group["Actual Label"] == group["Predicted Label"]) & (
                group["Actual Label"] == class_label
            )
            tp = tp_predictions.sum()
            fp_predictions = (group["Predicted Label"] == class_label) & (
                group["Actual Label"] != class_label
            )
            fp = fp_predictions.sum()
            return tp / (tp + fp) if (tp + fp) > 0 else 0.0

        def calculate_recall_for_class(group: pd.Series, class_label: int) -> float:
            """Calculate recall for a specific class within a grouped object."""
            tp_predictions = (group["Actual Label"] == group["Predicted Label"]) & (
                group["Actual Label"] == class_label
            )
            tp = tp_predictions.sum()
            fn_predictions = (group["Predicted Label"] != class_label) & (
                group["Actual Label"] == class_label
            )
            fn = fn_predictions.sum()
            return tp / (tp + fn) if (tp + fn) > 0 else 0.0

        def calculate_f1_for_class(group: pd.Series, class_label: int) -> float:
            precision = calculate_precision_for_class(group, class_label)
            recall = calculate_recall_for_class(group, class_label)
            return (
                (2 * precision * recall / (precision + recall))
                if (precision + recall) > 0
                else 0.0
            )

        def calculate_class_count(
            group: pd.Series, class_label: int
        ) -> tuple[int, int]:
            num_actual = (group["Actual Label"] == class_label).sum()
            num_pred = (group["Predicted Label"] == class_label).sum()

            return num_actual, num_pred

        split_files = [os.path.basename(filename) for filename in self.filenames]

        if colonised_only:
            data = {
                "Filename": split_files,
                "Actual Label": self.y_test_labels,
            }
            df = pd.DataFrame(data)
            if self.colonisation_type == "am":
                # Calculate % colonised for actual and predicted labels
                colonisation_metrics = df.groupby("Filename").apply(
                    lambda x: pd.Series(
                        {
                            "Actual % AM + Hybrid Colonised": (
                                calculate_am_colonised_percentage(x["Actual Label"])[0]
                            ),
                            "Actual % DSE + Hybrid Colonised": (
                                calculate_am_colonised_percentage(x["Actual Label"])[1]
                            ),
                            "Actual % Total Colonised including Hybrid": (
                                calculate_am_colonised_percentage(x["Actual Label"])[2]
                            ),
                            "Actual % Root": (
                                calculate_am_colonised_percentage(x["Actual Label"])[3]
                            ),
                        }
                    )
                )
            else:
                colonisation_metrics = df.groupby("Filename").apply(
                    lambda x: pd.Series(
                        {
                            "Actual % Blue Coil Colonised": (
                                calculate_erm_colonised_percentage(x["Actual Label"])[0]
                            ),
                            "Actual % Brown Coil Colonised": (
                                calculate_erm_colonised_percentage(x["Actual Label"])[1]
                            ),
                            "Actual % Type Two Colonised": (
                                calculate_erm_colonised_percentage(x["Actual Label"])[2]
                            ),
                            "Actual % DSE + Hybrid DSE Colonised": (
                                calculate_erm_colonised_percentage(x["Actual Label"])[3]
                            ),
                            "Actual % Total ErM Colonisation (inc. Hybrid ErM)": (
                                calculate_erm_colonised_percentage(x["Actual Label"])[4]
                            ),
                            "Actual % Root": calculate_erm_colonised_percentage(
                                x["Actual Label"]
                            )[5],
                        }
                    )
                )

            # Combine both metrics into a single DataFrame
            file_metrics = colonisation_metrics
            file_metrics_with_index = file_metrics.reset_index()
            file_metrics_with_index.rename(columns={"index": "Filename"}, inplace=True)

            # Then add the modified DataFrame to the MetricsCollector
            self.metrics_collector.add_metrics_dataframe(
                "file_metrics", file_metrics_with_index
            )

        else:
            data = {
                "Filename": split_files,
                "Actual Label": self.y_test_labels,
                "Predicted Label": self.predicted_labels,
                # Include other relevant data as needed
            }
            df = pd.DataFrame(data)

            # Calculate % colonised for actual and predicted labels
            if self.colonisation_type == "am":
                colonisation_metrics = df.groupby("Filename").apply(
                    lambda x: pd.Series(
                        {
                            "Actual % AM + Hybrid Colonised": (
                                calculate_am_colonised_percentage(x["Actual Label"])[0]
                            ),
                            "Actual % DSE + Hybrid Colonised": (
                                calculate_am_colonised_percentage(x["Actual Label"])[1]
                            ),
                            "Actual % Total Colonised including Hybrid": (
                                calculate_am_colonised_percentage(x["Actual Label"])[2]
                            ),
                            "Actual % Root": calculate_am_colonised_percentage(
                                x["Actual Label"]
                            )[3],
                            "Predicted % AM + Hybrid Colonised": (
                                calculate_am_colonised_percentage(x["Predicted Label"])[
                                    0
                                ]
                            ),
                            "Predicted % DSE + Hybrid Colonised": (
                                calculate_am_colonised_percentage(x["Predicted Label"])[
                                    1
                                ]
                            ),
                            "Predicted % Total Colonised including Hybrid": (
                                calculate_am_colonised_percentage(x["Predicted Label"])[
                                    2
                                ]
                            ),
                            "Predicted % Root": calculate_am_colonised_percentage(
                                x["Predicted Label"]
                            )[3],
                        }
                    )
                )
            else:
                colonisation_metrics = df.groupby("Filename").apply(
                    lambda x: pd.Series(
                        {
                            "Actual % Blue Coil Colonised": (
                                calculate_erm_colonised_percentage(x["Actual Label"])[0]
                            ),
                            "Actual % Brown Coil Colonised": (
                                calculate_erm_colonised_percentage(x["Actual Label"])[1]
                            ),
                            "Actual % Type Two Colonised": (
                                calculate_erm_colonised_percentage(x["Actual Label"])[2]
                            ),
                            "Actual % DSE + Hybrid DSE Colonised": (
                                calculate_erm_colonised_percentage(x["Actual Label"])[3]
                            ),
                            "Actual % Total ErM Colonisation (inc. Hybrid ErM)": (
                                calculate_erm_colonised_percentage(x["Actual Label"])[4]
                            ),
                            "Actual % Root": calculate_erm_colonised_percentage(
                                x["Actual Label"]
                            )[5],
                            "Predicted % Blue Coil Colonised": (
                                calculate_erm_colonised_percentage(
                                    x["Predicted Label"]
                                )[0]
                            ),
                            "Predicted % Brown Coil Colonised": (
                                calculate_erm_colonised_percentage(
                                    x["Predicted Label"]
                                )[1]
                            ),
                            "Predicted % Type Two Colonised": (
                                calculate_erm_colonised_percentage(
                                    x["Predicted Label"]
                                )[2]
                            ),
                            "Predicted % DSE + Hybrid DSE Colonised": (
                                calculate_erm_colonised_percentage(
                                    x["Predicted Label"]
                                )[3]
                            ),
                            "Predicted % Total ErM Colonisation (inc. Hybrid ErM)": (
                                calculate_erm_colonised_percentage(
                                    x["Predicted Label"]
                                )[4]
                            ),
                            "Predicted % Root": calculate_erm_colonised_percentage(
                                x["Predicted Label"]
                            )[5],
                        }
                    )
                )

            # Calculate per-class accuracy for each filename
            class_accuracies = df.groupby("Filename").apply(
                lambda group: pd.Series(
                    {
                        f"Accuracy Class {self.class_names[i]}": (
                            calculate_accuracy_for_class(group, i)
                        )
                        for i in range(len(self.class_names))
                    }
                )
            )

            class_precisions = df.groupby("Filename").apply(
                lambda group: pd.Series(
                    {
                        f"Precision Class {self.class_names[i]}": (
                            calculate_precision_for_class(group, i)
                        )
                        for i in range(len(self.class_names))
                    }
                )
            )

            # Calculate per-class recall for each filename
            class_recalls = df.groupby("Filename").apply(
                lambda group: pd.Series(
                    {
                        f"Recall Class {self.class_names[i]}": (
                            calculate_recall_for_class(group, i)
                        )
                        for i in range(len(self.class_names))
                    }
                )
            )

            # Calculate per-class F1 score for each filename
            class_f1_scores = df.groupby("Filename").apply(
                lambda group: pd.Series(
                    {
                        f"F1 Score Class {self.class_names[i]}": calculate_f1_for_class(
                            group, i
                        )
                        for i in range(len(self.class_names))
                    }
                )
            )

            gt_class_quantities = df.groupby("Filename").apply(
                lambda group: pd.Series(
                    {
                        **{
                            f"Num Actual {class_name}": calculate_class_count(group, i)[
                                0
                            ]
                            for i, class_name in enumerate(self.class_names)
                        },
                        **{
                            f"Num Pred {class_name}": calculate_class_count(group, i)[1]
                            for i, class_name in enumerate(self.class_names)
                        },
                    }
                )
            )

            # Combine both metrics into a single DataFrame
            file_metrics = (
                colonisation_metrics.join(class_accuracies)
                .join(class_precisions)
                .join(class_recalls)
                .join(class_f1_scores)
                .join(gt_class_quantities)
            )
            file_metrics_with_index = file_metrics.reset_index()
            file_metrics_with_index.rename(columns={"index": "Filename"}, inplace=True)

            # Then add the modified DataFrame to the MetricsCollector
            self.metrics_collector.add_metrics_dataframe(
                "file_metrics", file_metrics_with_index
            )

        return

    def plot_class_examples(
        self,
        axes: NDArray[matplotlib.axes.Axes],
        example_indices: list[np.uint8],
        example_fileparts: list[str],
    ) -> None:
        """
        Plots example images where a specific class was incorrectly predicted.
        """
        if self.predicted_labels is None:
            raise ValueError("Predicted labels cannot be None")
        if self.rows is None or self.cols is None:
            raise ValueError("rows and cols cannot be None")
        for i, (example_idx, example_filepart) in enumerate(
            zip(example_indices, example_fileparts)
        ):
            if i >= len(axes.flat):
                break
            img = self.x_test[example_idx].squeeze()
            predicted_label = self.class_names[self.predicted_labels[example_idx]]
            img = np.transpose(img, (1, 2, 0))
            axes.flat[i].imshow(img, cmap="gray" if img.ndim == 2 else None)
            axes.flat[i].set_title(
                f"Prediction: {predicted_label}\nColumn: {self.cols[example_idx]}, "
                f"Row: {self.rows[example_idx]}",
                fontsize=16,
            )
            axes.flat[i].axis("off")

    def get_pred_by_file(self) -> None:
        incorrect_indices = np.where(self.predicted_labels != self.y_test_labels)[0]

        # Group incorrect predictions by filename
        incorrect_by_filename: dict[str, list[dict[str, Any]]] = {}
        for idx in incorrect_indices:
            split = os.path.splitext(os.path.basename(self.filenames[idx]))
            filename = split[0]
            if filename not in incorrect_by_filename:
                incorrect_by_filename[filename] = []
            incorrect_by_filename[filename].append(
                {"index": idx, "file_part": filename}
            )

        for filename, indices in incorrect_by_filename.items():
            # AmfLog.info(f"Processing incorrect prediction images for {filename}")
            logger.info(f"Processing incorrect prediction images for {filename}")
            file_dir = os.path.join(self.results_dir, filename)
            os.makedirs(file_dir, exist_ok=True)  # Create a directory for each file

            # Create a figure for the incorrect predictions associated with the current
            # filename
            for class_idx, class_name in enumerate(self.class_names):
                # Find indices of incorrect predictions for this class
                class_incorrect_indices = [
                    idx["index"]
                    for idx in indices
                    if self.y_test_labels[idx["index"]] == class_idx
                ]
                class_incorrect_fileparts = [
                    idx["file_part"]
                    for idx in indices
                    if self.y_test_labels[idx["index"]] == class_idx
                ]
                # Skip if there are no incorrect predictions for this class
                if not class_incorrect_indices:
                    continue

                num_incorrect = len(class_incorrect_indices)
                num_rows, num_columns = self.find_optimal_layout(
                    num_incorrect + 3
                )  # +3 for the additional class examples
                fig, axes = plt.subplots(
                    num_rows,
                    num_columns,
                    figsize=(5 * num_columns, 5 * num_rows),
                    squeeze=False,
                )
                fig.suptitle(
                    f"Incorrect {class_name} Predictions for {filename}", fontsize=30
                )

                self.plot_class_examples(
                    axes, class_incorrect_indices, class_incorrect_fileparts
                )

                total_plots = num_rows * num_columns
                for unused_ax_idx in range(num_incorrect, total_plots):
                    unused_row, unused_col = divmod(unused_ax_idx, num_columns)
                    axes[unused_row, unused_col].axis("off")
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                image_path = os.path.join(file_dir, f"incorrect_{class_name}.png")
                plt.savefig(image_path)
                self.metrics_collector.add_image(
                    f"{filename}/incorrect_{class_name}.png", image_path
                )
                plt.close(fig)  # Close the figure to free memory

    def find_optimal_layout(self, num_items: int) -> tuple[int, int]:
        """
        Find the number of rows and columns for a grid layout with num_items,
        aiming for a layout that is as square as possible.
        """
        sqrt_val = int(math.sqrt(num_items))
        if sqrt_val * sqrt_val == num_items:
            # Perfect square, equal rows and columns
            return sqrt_val, sqrt_val

        for extra in range(1, sqrt_val + 1):
            # Attempt to find factors by incrementally checking numbers larger than the
            # square root
            if num_items % (sqrt_val + extra) == 0:
                return num_items // (sqrt_val + extra), sqrt_val + extra

        # If no factors found that make a perfect rectangle (e.g., prime numbers),
        # use the closest square higher than the number of items
        nearest_square = (sqrt_val + 1) ** 2
        row, col = self.find_optimal_layout(nearest_square)
        # Adjust rows if the number of items doesn't require all rows in the nearest
        # square layout
        if num_items <= row * (col - 1):
            return row, col - 1
        return row, col

    def get_pred_by_class(self) -> None:
        """
        Gets the predictions in each class that were incorrect and plots them in
        separate files for each class.
        """

        if self.predicted_labels is None:
            raise ValueError("Predicted labels must not be None")

        # Determine unique classes in the dataset
        unique_classes = np.unique(self.y_test_labels)

        # Iterate over each class to plot its incorrectly predicted images
        for class_index in unique_classes:
            # AmfLog.info(
            #     "Processing incorrect prediction images for class "
            #     f"{self.class_names[class_index]}"
            # )
            logger.info(
                f"Processing incorrect prediction images for class \
                {self.class_names[class_index]}"
            )
            # Indices where this class is the true class but was predicted incorrectly
            incorrect_indices = np.where(
                (self.y_test_labels == class_index)
                & (self.predicted_labels != class_index)
            )[0]

            # If there are any incorrect predictions for this class, plot them
            if len(incorrect_indices) > 0:
                num_incorrect = len(incorrect_indices)

                # Calculate the number of columns for subplot based on the number of
                # incorrect images, with a max of 5 columns
                num_rows, num_columns = self.find_optimal_layout(num_incorrect)

                # Create a figure for the incorrect predictions of the current class
                fig, axes = plt.subplots(
                    num_rows,
                    num_columns,
                    figsize=(5 * num_columns, 5 * num_rows),
                    squeeze=False,
                )
                fig.suptitle(
                    f"Incorrect Predictions for Class {self.class_names[class_index]}",
                    fontsize=16,
                )

                for i, idx in enumerate(incorrect_indices):
                    row, col = divmod(i, num_columns)
                    ax = axes[row, col]
                    # This should already be normalised
                    img = self.x_test[
                        idx
                    ].squeeze()  # Adjust for your dataset's shape/format
                    # This is the opposite of the transpose in amf.helper.segmentation
                    img = np.transpose(img, (1, 2, 0))
                    ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
                    ax.set_title(
                        f"Pred: {self.class_names[self.predicted_labels[idx]]}",
                        fontsize=10,
                    )
                    ax.axis("off")

                # Hide any unused subplots
                for j in range(i + 1, num_rows * num_columns):
                    axes.flat[j].axis("off")

                plt.tight_layout(
                    rect=[0, 0.03, 1, 0.95]
                )  # Adjust layout to make room for the suptitle
                save_path = os.path.join(
                    self.results_dir, f"incorrect_{self.class_names[class_index]}.png"
                )
                plt.savefig(save_path)

    def get_num_tiles_per_confidence(self) -> None:
        if self.predicted_probs is None:
            raise ValueError("Predicted probabilities must not be None")

        total_tiles = self.predicted_probs.shape[0]
        max_probs = self.predicted_probs.max(axis=1)

        thresholds = np.linspace(0.1, 1.0, 10)
        counts, _ = np.histogram(max_probs, bins=np.concatenate(([0], thresholds)))

        cumulative_count_dist = np.cumsum(counts)
        dict_percentages = {
            f"Tiles with confidence <= {threshold:.1f} in %": 100 * count / total_tiles
            for threshold, count in zip(thresholds, cumulative_count_dist)
        }
        df_percentages = pd.DataFrame([dict_percentages])
        save_path = os.path.join(self.results_dir, "num_tiles_per_threshold.csv")
        df_percentages.to_csv(
            save_path,
            encoding="utf-8",
            index=False,
            lineterminator="\n",
            float_format="%g",
        )

    def get_metrics_with_threshold_comparison(self) -> None:
        if self.predicted_probs is None:
            raise ValueError("Predicted probabilities must not be None")
        if self.predicted_labels is None:
            raise ValueError("Predicted labels must not be None")
        if self.predictions is None:
            raise ValueError("Predictions must not be None")

        thresholds = np.linspace(0.1, 1.0, 10)
        thresholds = np.round(thresholds, 1)

        folder_path = os.path.join(
            self.results_dir, "class_metrics_after_manual_labelling"
        )
        os.makedirs(folder_path, exist_ok=True)

        for conversion_threshold in thresholds:
            max_probs = self.predicted_probs.max(axis=1)
            rows_to_modify = max_probs <= conversion_threshold
            adjusted_predicted_labels = self.predicted_labels.copy()

            adjusted_predicted_labels[rows_to_modify] = np.argmax(
                self.predictions[rows_to_modify], axis=1
            )

            mapped_predictions = [
                self.class_names[label] for label in adjusted_predicted_labels
            ]
            mapped_true_labels = [
                self.class_names[label] for label in self.y_test_labels
            ]

            # Generate confusion matrix with class names
            cm = confusion_matrix(
                mapped_true_labels, mapped_predictions, labels=self.class_names
            )

            # Calculate per-class metrics
            accuracy = self.safe_divide(np.diag(cm), np.sum(cm, axis=1))
            precision = self.safe_divide(np.diag(cm), np.sum(cm, axis=0))
            recall = self.safe_divide(np.diag(cm), np.sum(cm, axis=1))
            f1_score = 2 * self.safe_divide(precision * recall, precision + recall)

            # Handle division by zero for accuracy, precision, recall, and F1
            accuracy = np.nan_to_num(accuracy, nan=0.0)
            precision = np.nan_to_num(precision, nan=0.0)
            recall = np.nan_to_num(recall, nan=0.0)
            f1_score = np.nan_to_num(f1_score, nan=0.0)

            # Save updated per class metrics
            save_path = os.path.join(
                folder_path, f"threshold_{conversion_threshold}.csv"
            )
            data = []
            for class_name, acc, prec, rec, f1 in zip(
                self.class_names, accuracy, precision, recall, f1_score
            ):
                data.append(
                    {
                        "class_name": class_name,
                        "Accuracy": acc,
                        "Precision": prec,
                        "Recall": rec,
                        "F1 Score": f1,
                    }
                )
                df_res = pd.DataFrame(data)

                # Add averages
                averages = {
                    "class_name": "Average",
                    "Accuracy": np.mean(accuracy),
                    "Precision": np.mean(precision),
                    "Recall": np.mean(recall),
                    "F1 Score": np.mean(f1_score),
                }
                average_df = pd.DataFrame([averages])
                df_res = pd.concat([df_res, average_df], ignore_index=True)

                # Save results
                df_res.to_csv(
                    save_path,
                    encoding="utf-8",
                    index=False,
                    lineterminator="\n",
                    float_format="%g",
                )
