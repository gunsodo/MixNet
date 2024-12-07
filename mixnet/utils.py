"""Utility functions"""

import logging
import sys
from typing import Any, Dict, List, Literal

import scipy
import sklearn
import numpy as np
import tensorflow as tf
import yaml
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    f1_score,
    recall_score,
)

from ._dataset_metadata import DATASETS, DatasetMetadata


logger = logging.getLogger(__name__)


def load_raw(dataset: Literal[
    "OpenBMI",
    "BCIC2a",
    "BCIC2b",
    "SMR_BCI",
    "BNCI2015_001",
    "HighGamma",
]):
    """
    Download publicly available dataset with the given name from the internet

    Parameters
    ----------
    dataset: {"OpenBMI", "BCIC2a", "BCIC2b", "SMR_BCI", "BNCI2015_001", "HighGamma"}
        Name of the dataset
    """
    dataset: DatasetMetadata = DATASETS[dataset]
    dataset.load()


def compute_class_weight(y_train):
    """compute class balancing

    Args:
        y_train (list, ndarray): [description]

    Returns:
        (dict): class weight balancing
    """
    return dict(
        zip(
            np.unique(y_train),
            sklearn.utils.class_weight.compute_class_weight(
                class_weight="balanced", classes=np.unique(y_train), y=y_train
            ),
        )
    )


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def write_log(filepath="test.log", data=[], mode="w"):
    """
    filepath: path to save
    data: list of data
    mode: a = update data to file, w = write a new file
    """
    try:
        with open(filepath, mode) as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(data)
    except IOError:
        raise Exception("I/O error")


def zero_padding(data, pad_size=4):
    if len(data.shape) != 4:
        raise Exception("Dimension is not match!, must have 4 dims")
    new_shape = int(data.shape[2] + (2 * pad_size))
    data_pad = np.zeros((data.shape[0], data.shape[1], new_shape, new_shape))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data_pad[i, j, :, :] = np.pad(
                data[i, j, :, :], [pad_size, pad_size], mode="constant"
            )
    print(data_pad.shape)
    return data_pad


def butter_bandpass_filter(data, lowcut, highcut, sfreq, order):
    nyq = 0.5 * sfreq
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy.signal.butter(order, [low, high], btype="band")
    y = scipy.signal.filtfilt(b, a, data)
    return y


def resampling(data, sfreq, data_len):
    if len(data.shape) != 3:
        raise Exception("Dimesion error",
                        "--> please use three-dimensional input")
    new_smp_point = int(data_len * sfreq)
    data_resampled = np.zeros((data.shape[0], data.shape[1], new_smp_point))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data_resampled[i, j, :] = scipy.signal.resample(
                data[i, j, :], new_smp_point
            )
    return data_resampled


def psd_welch(data, smp_freq):
    if len(data.shape) != 3:
        raise Exception("Dimension Error, must have 3 dimension")
    n_samples, n_chs, n_points = data.shape
    data_psd = np.zeros((n_samples, n_chs, 89))
    for i in range(n_samples):
        for j in range(n_chs):
            freq, power_den = scipy.signal.welch(
                data[i, j], smp_freq, nperseg=n_points)
            index = np.where((freq >= 8) & (freq <= 30))[0].tolist()
            # print("the length of---", len(index))
            data_psd[i, j] = power_den[index]
    return data_psd


def mean_dict(data: List[Dict[str, Any]]) -> dict[str, Any]:
    return {k: tf.reduce_mean(d[k]) for k in data[0] for d in data}


def log_dict(data: Dict[str, float]) -> None:
    for k, v in data.items():
        logger.info(" - %s: %.4f", k, v)


def sen_spec(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    recall = recall_score(y_true, y_pred, pos_label=1, average="binary")
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    f1_score_binary = 2 * tp / (2 * tp + fp + fn)
    f1_score_macro = f1_score(y_true, y_pred, average="macro")
    f1_score_weighted = f1_score(y_true, y_pred, average="weighted")
    logger.info("Verifying sensitivity {} and recall {}".format(
        sensitivity, recall))
    return sensitivity, specificity, f1_score_binary, f1_score_macro, f1_score_weighted


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


def get_yaml_path() -> str:
    assert len(sys.argv) == 2, "Number of arguments must be 2, python run.py yaml_file_path"
    return sys.argv[-1]
