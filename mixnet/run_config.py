from typing import List, Literal, Tuple, Type

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RunConfig(BaseModel):
    dataset: Literal["BCIC2a", "BCIC2b", "BNCI2015_001", "SMR_BCI", "HighGamma", "OpenBMI"] = "HighGamma"
    train_type: Literal["subject_dependent", "subject_independent"] = "subject_dependent"
    num_class: int = 2
    log_dir: str = "logs"
    subjects: List[int] = None
    gpu: str = "0"


class MixNetConfig(RunConfig):
    name: str = "MixNet"
    data_type: str = "spectral_spatial_signals"

    latent_dim: int = 64
    loss_weights: List[int] = Field(default_factory=lambda x: [1.0, 1.0, 1.0])
    adaptive_gradient: bool = True
    policy: str = "HistoricalTangentSlope"
    margin: float = 1.0
    n_component: int = None
    warmup: int = 5
