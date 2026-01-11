from pathlib import Path
from importlib.resources import files
from typing import Tuple, List

from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


class DataSettings(BaseModel):
    """Settings for data generation."""
    train_ratio: float = 0.8

class ModelSettings(BaseModel):
    input_depth: int = 1
    num_classes: int = 10
    layer_depth: List[int] = [32, 64]
    layer_kernel_sizes: List[List[int]] = [[3, 3], [3,3]]
    strides: List[int] = [2, 2] 
#increased spatial size for faster training bc it literally took 10 min before this
    dropout_rate: float = 0.1
    shape: List[int] = [28, 28]

class TrainingSettings(BaseModel):
    """Settings for model training."""
    batch_size: int = 128
    num_iters: int = 1000
    learning_rate: float = 0.001
    L2_weight: float = 1e-3

class AppSettings(BaseSettings):
    """Main application settings."""

    debug: bool = False
    random_seed: int = 427 #for ece 427 ofc
    data: DataSettings = DataSettings()
    training: TrainingSettings = TrainingSettings()
    model: ModelSettings = ModelSettings()

    model_config = SettingsConfigDict(
        toml_file=files("hw03").joinpath("config.toml"),
        env_nested_delimiter="__",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """
        Set the priority of settings sources.

        We use a TOML file for configuration.
        """
        return (
            init_settings,
            TomlConfigSettingsSource(settings_cls),
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )

def load_settings() -> AppSettings:
    """Load application settings."""
    return AppSettings()
