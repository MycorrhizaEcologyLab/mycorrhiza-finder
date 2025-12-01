import os
import time
from typing import Type

# Import from https://github.com/markus93/NN_calibration/blob/master/scripts/calibration/cal_methods.py
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import torch
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, minimize
from sklearn.metrics import f1_score, log_loss
from torch.utils.data import DataLoader
from tqdm import tqdm

import amf.helper.config as AmfConfig
import amf.helper.load as AmfLoad
import amf.helper.log as AmfLog
import amf.helper.model as AmfModel


# Defining relevant functions
def ECE(
    conf: NDArray[np.float64],
    pred: NDArray[np.int_],
    true: NDArray[np.int_],
    bin_size: float = 0.1,
) -> float:
    """
    Expected Calibration Error

    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)

    Returns:
        ece: expected calibration error
    """

    upper_bounds = np.arange(bin_size, 1 + bin_size, bin_size)  # Get bounds of bins

    n = len(conf)
    ece = 0  # Starting error

    for (
        conf_thresh
    ) in upper_bounds:  # Go through bounds and find accuracies and confidences
        acc, avg_conf, len_bin = compute_acc_bin(
            conf_thresh - bin_size, conf_thresh, conf, pred, true
        )
        ece += np.abs(acc - avg_conf) * len_bin / n  # Add weigthed difference to ECE

    return ece


def MCE(
    conf: NDArray[np.float64],
    pred: NDArray[np.int_],
    true: NDArray[np.int_],
    bin_size: float = 0.1,
) -> float:
    """
    Maximal Calibration Error

    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)

    Returns:
        mce: maximum calibration error
    """

    upper_bounds = np.arange(bin_size, 1 + bin_size, bin_size)

    cal_errors: list[float] = []

    for conf_thresh in upper_bounds:
        acc, avg_conf, _ = compute_acc_bin(
            conf_thresh - bin_size, conf_thresh, conf, pred, true
        )
        cal_errors.append(np.abs(acc - avg_conf))

    return max(cal_errors)


def softmax(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute softmax values for each sets of scores in x.

    Parameters:
        x (numpy.ndarray): array containing m samples with n-dimensions (m,n)
    Returns:
        x_softmax (numpy.ndarray) softmaxed values for initial (m,n) array
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=1)  # type: ignore[no-any-return]


def compute_acc_bin(
    conf_thresh_lower: float,
    conf_thresh_upper: float,
    conf: NDArray[np.float64],
    pred: NDArray[np.int_],
    true: NDArray[np.int_],
) -> tuple[float, float, int]:
    """
    # Computes accuracy and average confidence for bin

    Args:
        conf_thresh_lower (float): Lower Threshold of confidence interval
        conf_thresh_upper (float): Upper Threshold of confidence interval
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels

    Returns:
        (accuracy, avg_conf, len_bin): accuracy of bin, confidence of bin and number of
            elements in bin.
    """
    filtered_tuples = [
        x
        for x in zip(pred, true, conf)
        if x[2] > conf_thresh_lower and x[2] <= conf_thresh_upper
    ]

    len_bin = len(filtered_tuples)  # How many elements falls into given bin

    if len_bin < 1:
        return 0, 0, 0

    correct = len(
        [x for x in filtered_tuples if x[0] == x[1]]
    )  # How many correct labels
    avg_conf = sum([x[2] for x in filtered_tuples]) / len_bin  # Avg confidence of BIN
    accuracy = float(correct) / len_bin  # accuracy of BIN
    return accuracy, avg_conf, len_bin


# Defining model for TemperatureScaling
class TemperatureScaling:
    def __init__(self, temp: float = 1, maxiter: int = 50, solver: str = "BFGS"):
        """
        Initialize class

        Params:
            temp (float): starting temperature, default 1
            maxiter (int): maximum iterations done by optimizer, however 8 iterations
                have been maximum.
        """
        self.temp = temp
        self.maxiter = maxiter
        self.solver = solver

    def _loss_fun(
        self, x: float, probs: NDArray[np.float64], true: NDArray[np.int_]
    ) -> float:
        # Calculates the loss using log-loss (cross-entropy loss)
        scaled_probs = self.predict(probs, x)
        loss: float = log_loss(y_true=true, y_pred=scaled_probs)
        return loss

    # Find the temperature
    def fit(
        self, logits: NDArray[np.float64], true: NDArray[np.int_]
    ) -> OptimizeResult:
        """
        Trains the model and finds optimal temperature

        Params:
            logits: the output from neural network for each class
                (shape [samples, classes]).
            true: one-hot-encoding of true labels.

        Returns:
            the results of optimizer after minimizing is finished.
        """

        # true = true.flatten()  # Flatten y_val
        opt = minimize(
            self._loss_fun,
            x0=1,
            args=(logits, true),
            options={"maxiter": self.maxiter},
            method=self.solver,
        )
        self.temp = opt.x[0]

        return opt

    def predict(
        self, logits: NDArray[np.float64], temp: float | None = None
    ) -> NDArray[np.float64]:
        """
        Scales logits based on the temperature and returns calibrated probabilities

        Params:
            logits: logits values of data (output from neural network) for each class
                (shape [samples, classes])
            temp: if not set use temperatures find by model or previously set.

        Returns:
            calibrated probabilities (nd.array with shape [samples, classes])
        """

        if not temp:
            return softmax(logits / self.temp)

        return softmax(logits / temp)


def evaluate(
    probs: NDArray[np.float64],
    y_true: NDArray[np.int_],
    verbose: bool = False,
    normalize: bool = False,
    bins: int = 15,
) -> tuple[float, float, float, float, float]:
    """
    Evaluate model using various scoring measures: Error Rate, ECE, MCE, NLL, MacroF1
    Score

    Params:
        probs: a list containing probabilities for all the classes with a shape of
            (samples, classes)
        y_true: a list containing the actual class labels
        verbose: (bool) are the scores printed out. (default = False)
        normalize: (bool) in case of 1-vs-K calibration, the probabilities need to be
            normalized.
        bins: (int) - into how many bins are probabilities divided (default = 15)

    Returns:
        (error, ece, mce, loss, macrof1), returns various scoring measures
    """

    preds = np.argmax(probs, axis=1)  # Take maximum confidence as prediction
    y_true = np.argmax(y_true, axis=1)  # Convert from one-hot-encoded to class label

    if normalize:
        confs = np.max(probs, axis=1) / np.sum(probs, axis=1)
        # Check if everything below or equal to 1?
    else:
        confs = np.max(probs, axis=1)  # Take only maximum confidence

    accuracy = metrics.accuracy_score(y_true, preds) * 100
    error = 100 - accuracy

    # Calculate ECE
    ece = ECE(confs, preds, y_true, bin_size=1 / bins)
    # Calculate MCE
    mce = MCE(confs, preds, y_true, bin_size=1 / bins)

    # Calculate Log loss
    loss = log_loss(y_true=y_true, y_pred=probs)

    # Calculate MacroF1
    macrof1 = f1_score(y_true=y_true, y_pred=preds, average="macro")

    if verbose:
        print("Accuracy:", accuracy)
        print("Error:", error)
        print("ECE:", ece)
        print("MCE:", mce)
        print("Loss:", loss)
        print("MacroF1:", macrof1)

    return (error, ece, mce, loss, macrof1)


def cal_results(
    fn: Type[TemperatureScaling],
    logits_data: tuple[NDArray[np.float64], NDArray[np.int_]],
) -> tuple[float, pd.DataFrame, NDArray[np.float64]]:
    """
    Calibrate models scores, using output from logits files and given function (fn).
    There are implemented to different approaches "all" and "1-vs-K" for calibration,
    the approach of calibration should match with function used for calibration.

    TODO: split calibration of single and all into separate functions for more use
    cases.

    Params:
        fn (class): class of the calibration method used. It must contain methods "fit"
            and "predict",
            where first fits the models and second outputs calibrated probabilities.
        logits_data (tuple): Tuple containing two arrays - (logits and one-hot-encoded
            labels)
        approach (string): "all" for multiclass calibration and "1-vs-K" for 1-vs-K
            approach.

    Returns:
        df (pandas.DataFrame): dataframe with calibrated and uncalibrated results for
            all the input files.

    """

    # Empty Dataframe
    df = pd.DataFrame(columns=["Name", "Error", "ECE", "MCE", "Loss", "Macro F1"])

    logits, labels = logits_data

    name = "Calibration"
    AmfLog.info("Calibrating model")
    t1 = time.time()

    # Defining labels and model
    # labels = labels.flatten()
    model = fn()

    # Fitting the model to logits and labels
    model.fit(logits, labels)
    probs = model.predict(logits)

    # Performance before calibration
    error, ece, mce, loss, macrof1 = evaluate(softmax(logits), labels, verbose=True)

    # Perfomrance after calibration
    error2, ece2, mce2, loss2, macrof12 = evaluate(probs, labels, verbose=False)

    df.loc[0] = [name, error, ece, mce, loss, macrof1]
    df.loc[1] = [(name + "_calib"), error2, ece2, mce2, loss2, macrof12]

    print(
        "Error %f; ece %f; mce %f; loss %f, macrof1 %f"
        % evaluate(probs, labels, verbose=False, normalize=True)
    )

    t2 = time.time()
    print("Time taken for this calibration:", (t2 - t1), "\n")

    return model.temp, df, probs


# TODO this can hit RAM limits with large datasets. Would be
# good to look into a solution to limit memory use here
def run(input_files: list[str]) -> int:
    """
    Loads a specified, trained network, and estimates a temperature
    based on Platt scaling for calibration by logistic regression.

    :param input_files: List of input images to train with.
    """
    # Trained model
    model = AmfModel.load()

    # Assign correct device, depending on cpu or gpu
    device = AmfConfig.get("device")
    model = model.to(device)

    # Freezing model weights
    model.eval()

    # Categorise train path for logistic regression
    cal_img_list = AmfLoad.categorise_path(input_files)["train"]

    # Validate input folder structure
    if not cal_img_list:
        AmfLog.error("There is no train subfolder", AmfLog.ERR_NO_DATA)
        return 500

    # Create timestamped folder for results
    results_dir = AmfConfig.get("outdir")
    print(f"Results Directory: {results_dir}")

    # Extracting tiles and labels using list logic, to omit getitem() method.
    cal_dataset = AmfLoad.TileFilesandData(cal_img_list)
    x_cal, y_cal, filenames, rows, cols = cal_dataset.get_all_data()

    # Convert to CustomDataset and create DataLoader
    cal_normalised_dataset = AmfLoad.CustomNormalisedDataset(x_cal, y_cal)
    bs = AmfConfig.get("batch_size")
    num_workers = AmfConfig.get("num_workers")
    cal_dataset_loader = DataLoader(
        cal_normalised_dataset, batch_size=bs, shuffle=False, num_workers=num_workers
    )

    batch_count = len(cal_dataset_loader)

    # Get logits from all tiles
    all_logits = []
    all_labels = []
    AmfLog.info("Calculating model outputs for use in calibration")
    with torch.no_grad():  # Disabling gradient calculations
        for batch_x, batch_y in tqdm(
            cal_dataset_loader,
            total=batch_count,
            desc="Processing batches for calibration",
        ):
            # Load to correct device
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Forward pass
            outputs = model(batch_x)
            outputs_array = outputs.cpu().numpy()
            all_logits.extend(outputs_array)

            labels_array = batch_y.cpu().numpy()
            all_labels.extend(labels_array)

    all_logits_array = np.stack(all_logits, axis=0)
    all_labels_array = np.stack(all_labels, axis=0)

    # Directly pass to cal_results
    calmodel_temp, temp_scale_df, calibrated_probs = cal_results(
        TemperatureScaling, logits_data=(all_logits_array, all_labels_array)
    )

    # Create results dataframe
    # Assuming 'calibrated_probs' has one probability set per sample
    if not len(filenames) == len(rows) == len(cols) == len(calibrated_probs):
        raise ValueError(
            f"Length mismatch: {len(filenames)} filenames, {len(rows)} rows, "
            f"{len(cols)} cols, {len(calibrated_probs)} calibrated_probs"
        )

    colonisation_type = AmfConfig.get("colonisation_type")

    # Create DataFrame with calculated probabilities and corresponding metadata
    if colonisation_type == "am":
        calibrated_results_df = pd.DataFrame(
            {
                "Filename": filenames,
                "Row": rows,
                "Col": cols,
                "AMColonised": calibrated_probs[:, 0],
                "Uncolonised": calibrated_probs[:, 1],
                "Background": calibrated_probs[:, 2],
                "Unreadable": calibrated_probs[:, 3],
                "DSE": calibrated_probs[:, 4],
                "Hybrid": calibrated_probs[:, 5],
            }
        )
    else:
        calibrated_results_df = pd.DataFrame(
            {
                "Filename": filenames,
                "Row": rows,
                "Col": cols,
                "BlueCoils": calibrated_probs[:, 0],
                "BrownCoils": calibrated_probs[:, 1],
                "TypeTwo": calibrated_probs[:, 2],
                "Uncolonised": calibrated_probs[:, 3],
                "Background": calibrated_probs[:, 4],
                "MainRoot": calibrated_probs[:, 5],
                "Unreadable": calibrated_probs[:, 6],
                "DSE": calibrated_probs[:, 7],
                "HybridErm": calibrated_probs[:, 8],
                "HybridDse": calibrated_probs[:, 9],
            }
        )

    # Define file paths for saving
    base_name = (
        os.path.basename(AmfConfig.get("model"))
        if colonisation_type == "am"
        else os.path.basename(AmfConfig.get("model_erm"))
    )
    model_name = os.path.splitext(base_name)[0]
    temp_value_path = os.path.join(results_dir, f"{model_name}_temperature_value.txt")
    temp_scale_path = os.path.join(
        results_dir, f"{model_name}_temperature_scaling_results.csv"
    )
    calibrated_probs_path = os.path.join(
        results_dir, f"{model_name}_calibrated_probs.csv"
    )

    # Save the temperature value in a text file
    with open(temp_value_path, "w") as temp_file:
        temp_file.write(f"{calmodel_temp}")

    # Save DataFrames
    temp_scale_df.to_csv(temp_scale_path, sep=",", encoding="utf-8", index=False)
    calibrated_results_df.to_csv(
        calibrated_probs_path, sep=",", encoding="utf-8", index=False
    )

    print(f"Temperature value saved to: {temp_value_path}")
    print(f"Temperature scaling results saved to: {temp_scale_path}")
    print(f"Calibrated probabilities saved to: {calibrated_probs_path}")

    return 200
