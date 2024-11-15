import logging
import os
from dataclasses import dataclass
from functools import lru_cache, partial
from typing import Callable, Dict, List, Optional, Union
from tqdm import trange

import moabb
import numpy as np
import wget
from moabb.paradigms import MotorImagery


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FOLDER_NAME = "datasets"


@dataclass
class DatasetMetadata:
    name: str
    num_subjects: int
    base_url: Optional[str] = None

    load_single_subject: Callable[[str, int, str], None]
    # fallback_url: str

    @property
    @lru_cache
    def save_path(self) -> str:
        save_path = f"{FOLDER_NAME}/{self.name}/raw"
        os.makedirs(save_path, exist_ok=True)
        return save_path

    def load(self):
        logger.info("Downloading dataset %s to %s", self.name, self.save_path)
        for subject in trange(1, self.num_subjects + 1):
            self.load_single_subject(self.base_url, subject, self.save_path)


def _load_bci_dataset_with_sessions(
    base_url: str,
    subject: int,
    save_path: str,
    url_format: str,
    name_format: str,
    sessions: List[Union[str, int]],
) -> None:
    for session in sessions:
        file_name = name_format.format(subject=subject, session=session)
        file_save_path = os.path.join(save_path, file_name)

        if os.path.exists(file_save_path):
            os.remove(file_save_path)

        file_url = url_format.format(
            base_url=base_url,
            file_name=file_name,
            subject=subject,
            session=session,
        )
        wget.download(file_url, file_save_path)


def _load_bci_dataset_from_moabb(
    # pylint: disable-next=unused-argument
    base_url: str,
    subject: int,
    save_path: str,
    moabb_name: str,
    n_classes: int,
    meta_key: str,
    train_test_idx: List[str],
) -> None:
    dataset = getattr(moabb.datasets, moabb_name)()
    paradigm = MotorImagery(n_classes=n_classes)
    X, y, meta = paradigm.get_data(dataset=dataset, subjects=[subject])
    train_idx = list(meta[meta[meta_key] == train_test_idx[0]].index)
    test_idx = list(meta[meta[meta_key] == train_test_idx[1]].index)

    np.savez(
        f"{save_path}/S{subject:02d}.npz",
        X_train=X[train_idx],
        X_test=X[test_idx],
        y_train=y[train_idx],
        y_test=y[test_idx],
    )


OpenBMIDataset = DatasetMetadata(
    name="OpenBMI",
    num_subjects=54,
    base_url="ftp://parrot.genomics.cn/gigadb/pub/10.5524/100001_101000/100542",
    load_single_subject=partial(
        _load_bci_dataset_with_sessions,
        url_format="{base_url}/session{session}/{subject}{file_name}",
        name_format="/sess{session:02d}_subj{subject:02d}_EEG_MI.mat",
        sessions=[1, 2],
    )
)


BCIC2aDataset = DatasetMetadata(
    name="BCIC2a",
    num_subjects=9,
    base_url="https://lampx.tugraz.at/~bci/database/001-2014",
    load_single_subject=partial(
        _load_bci_dataset_with_sessions,
        url_format="{base_url}/{file_name}",
        name_format="/A{subject:02d}{session}.mat",
        sessions=["T", "E"],
    )
)


BCIC2bDataset = DatasetMetadata(
    name="BCIC2b",
    num_subjects=9,
    base_url="https://lampx.tugraz.at/~bci/database/004-2014",
    load_single_subject=partial(
        _load_bci_dataset_with_sessions,
        url_format="{base_url}/{file_name}",
        name_format="/B{subject:02d}{session}.mat",
        sessions=["T", "E"],
    )
)

SMRBCIDataset = DatasetMetadata(
    name="SMR_BCI",
    num_subjects=14,
    base_url="https://lampx.tugraz.at/~bci/database/002-2014",
    load_single_subject=partial(
        _load_bci_dataset_with_sessions,
        url_format="{base_url}/{file_name}",
        name_format="/S{subject:02d}{session}.mat",
        sessions=["T", "E"],
    )
)


BNCI2015001Dataset = DatasetMetadata(
    name="BNCI2015_001",
    num_subjects=12,
    load_single_subject=partial(
        _load_bci_dataset_from_moabb,
        moabb_name="BNCI2015_001",
        n_classes=2,
        meta_key="session",
        train_test_idx=["0A", "1B"],
    )
)


HighGammaDataset = DatasetMetadata(
    name="HighGamma",
    num_subjects=14,
    load_single_subject=partial(
        _load_bci_dataset_from_moabb,
        moabb_name="Schirrmeister2017",
        n_classes=4,
        meta_key="run",
        train_test_idx=["0train", "1test"],
    )
)


DATASETS: Dict[str, DatasetMetadata] = {
    d.name: d
    for d in [
        OpenBMIDataset,
        BCIC2aDataset,
        BCIC2bDataset,
        SMRBCIDataset,
        BNCI2015001Dataset,
        HighGammaDataset,
    ]
}
DATASET_NAMES: List[str] = list(DATASETS.keys())
