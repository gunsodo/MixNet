"""Data loader"""

import os
from typing import Literal, Optional, Tuple

import numpy as np

from ._dataset_metadata import DATASET_NAMES, FOLDER_NAME
from .utils import zero_padding


class DataLoader:
    """Data loader customized for MixNet"""

    def __init__(
        self,
        dataset: Literal[tuple(DATASET_NAMES)],
        train_type: Literal["subject_dependent", "subject_independent"],
        data_type: Literal["fbcsp", "spectral_spatial", "time_domain"],
        data_format: Optional[Literal["NCTD", "NDCT", "NTCD", "NSHWD"]] = None,
        dataset_path: str = f"/{FOLDER_NAME}",
        prefix_name: str = "S",
        num_class: int = 2,
        subject: Optional[int] = None,
        n_component: Optional[int] = None,
        **kwargs
    ):
        self.dataset = dataset
        self.train_type = train_type
        self.data_type = data_type
        self.dataset_path = dataset_path
        self.subject = subject
        self.data_format = data_format
        self.fold = None
        self.prefix_name = prefix_name
        self.num_class = num_class
        self.n_component = n_component
        self.kwargs = kwargs

    @property
    def path(self) -> str:
        """Dataset path"""

        path = os.path.join(
            self.dataset_path,
            self.dataset,
            self.data_type,
            self.num_class,
            "_class"
        )

        if self.n_component is not None:
            path = os.path.join(path, f"{self.n_component}_csp_components")

        path = os.path.join(path, self.train_type)
        return path

    def _change_data_format(self, X: np.ndarray) -> np.ndarray:
        if self.data_format == "NCTD":
            # (#n_trial, #channels, #time, #depth)
            X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        elif self.data_format == "NDCT":
            # (#n_trial, #depth, #channels, #time)
            X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
        elif self.data_format == "NTCD":
            # (#n_trial, #time, #channels, #depth)
            if X.ndim == 3:
                X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
                X = np.swapaxes(X, 1, 3)
            elif X.ndim == 4:
                X = np.swapaxes(X, 2, 3)
        elif self.data_format == "NSHWD":
            # (#n_trial, #Freqs, #height, #width, #depth)
            X = zero_padding(X)
            X = X.reshape(X.shape[0], X.shape[1], X.shape[2], X.shape[3], 1)
        return X

    def _load_set(self, split: str, fold: int) -> Tuple[np.ndarray, np.ndarray]:
        X, y = np.array([]), np.array([])
        suffix = f"{split}_{self.prefix_name}{self.subject:03d}_fold{fold:03d}.npy"
        x_path = os.path.join(self.path, f"X_{suffix}")
        y_path = os.path.join(self.path, f"y_{suffix}")
        X = self._change_data_format(np.load(x_path, allow_pickle=True))
        y = np.load(y_path, allow_pickle=True)
        return X, y

    def load_train_set(self, fold):
        return self._load_set(split="train", fold=fold)

    def load_val_set(self, fold):
        return self._load_set(split="val", fold=fold)

    def load_test_set(self, fold):
        return self._load_set(split="test", fold=fold)
