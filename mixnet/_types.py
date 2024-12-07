import os
from typing import Any, Dict, List, Literal, Optional, Union

from keras.metrics import SparseCategoricalAccuracy
from pydantic import BaseModel, ConfigDict, field_validator


class LossConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    loss: List[Any] = None
    weights: Union[float, List[float], None] = None
    adaptive_gradient: Union[bool, List[bool]] = False
    warmup_epoch: int = 5
    policy: Optional[str] = None

    @property
    def loss_names(self) -> List[str]:
        return [l.__name__ for l in self.loss]

    @property
    def loss_dict(self) -> Dict[str, Any]:
        return {l.__name__: l for l in self.loss}

    @property
    def n_losses(self) -> int:
        return len(self.loss)

    @property
    def n_adaptive(self) -> int:
        return len(self.adaptive_gradient)

    def use_adaptive_gradient(self) -> bool:
        return any(self.adaptive_gradient)

    @property
    def init_loss_weights(self) -> List[float]:
        return (
            [
                1.0 / self.n_adaptive if self.adaptive_gradient[i] else 1.0
                for i in range(self.n_losses)
            ]
            if self.weights is None
            else self.weights
        )

    @field_validator("weights")
    @classmethod
    def strict_list(cls, v: Union[float, List[float]]) -> List[float]:
        if isinstance(v, float):
            return [v]
        return v

    @field_validator("adaptive_gradient")
    @classmethod
    def check_adaptive_gradient(cls, v: Union[bool, List[bool]]) -> List[bool]:
        if isinstance(v, list):
            assert len(v) == len(cls.loss) or not isinstance(v, list)
        return [v] * cls.n_losses


class TimeHistoryConfig(BaseModel):
    time_log: str


class ReduceLRConfig(BaseModel):
    patience: int
    factor: float
    lr: float = 1e-2
    min_lr: float = 1e-3
    mode: Literal["min", "max"] = "min"
    verbose: int = 1
    warmup_epoch: int = 5
    monitor: str = "val_loss"

    def setup(self, lr: float):
        self.lr = lr


class ModelCheckpointConfig(BaseModel):
    verbose: int = 1
    monitor: str = "val_loss"
    save_best_only: bool = True
    save_weight_only: bool = True


class EarlyStoppingConfig(BaseModel):
    patience: int = 20
    warmup_epoch: int = 0
    monitor: str = "val_loss"


class CallbackConfig(BaseModel):
    early_stopping: EarlyStoppingConfig
    reduce_lr = ReduceLRConfig
    model_checkpoint: ModelCheckpointConfig
    time_history: TimeHistoryConfig


class TrainingConfig(BaseModel):
    num_class: int
    epochs: int = 200
    batch_size: int = 100
    shuffle: bool = True
    seed: int = 1234
    class_balancing: bool = True

    @property
    def f1_average(self) -> str:
        return "binary" if self.num_class == 2 else "macro"


class OptimizerConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    optimizer: Any
    lr: float = 1e-2

    def setup(self):
        self.optimizer.lr = self.lr


class MetricConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    train: SparseCategoricalAccuracy = SparseCategoricalAccuracy()
    val: SparseCategoricalAccuracy = SparseCategoricalAccuracy()
    test: SparseCategoricalAccuracy = SparseCategoricalAccuracy()
    pred: SparseCategoricalAccuracy = SparseCategoricalAccuracy()


class LogConfig(BaseModel):
    log_path: str = "logs"
    log_prefix: str = "model"

    @property
    def weights_dir(self):
        return os.path.join(self.log_path, f"{self.log_prefix}_out_weights.h5")

    @property
    def csv_dir(self):
        return os.path.join(self.log_path, f"{self.log_prefix}_out_log.csv")

    @property
    def time_log(self):
        return os.path.join(self.log_path, f"{self.log_prefix}_time_log.csv")


class Config(BaseModel):
    data_format: Literal["channels_first", "channels_last"]

    callback: CallbackConfig
    log: LogConfig
    loss: LossConfig
    optimizer: OptimizerConfig
    training: TrainingConfig
    metric: MetricConfig = MetricConfig()
