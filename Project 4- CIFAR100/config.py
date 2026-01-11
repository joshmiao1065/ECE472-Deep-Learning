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

#in data.py, change back to cifar10 and change num_classes here in model seetings back to 10
class ModelSettings(BaseModel):
    input_depth: int = 3
    num_classes: int = 10
    layer_depth: List[int] = [64, 128, 256]
    layer_kernel_sizes: List[List[int]] = [[3, 3], [3,3], [3,3]]
    strides: List[int] = [1, 1, 1] 
     #dropout_rate: float = 0.2
    shape: List[int] = [32, 32]
    noise_std: int = 0.05

class TrainingSettings(BaseModel):
    """Settings for model training."""
    batch_size: int = 256
    num_iters: int = 150000
    learning_rate: float = 1.5e-4
    L2_weight: float = 1.5e-4
    #tried to implement cosine decay with linear warmup but that was low key cheeks
    #warmup_steps: int = 800
    #initial_rate: float = 1e-5
    #end_value: float = 1e-5

class AppSettings(BaseSettings):
    """Main application settings."""

    debug: bool = False
    random_seed: int = 427 #for ece 427 ofc
    data: DataSettings = DataSettings()
    training: TrainingSettings = TrainingSettings()
    model: ModelSettings = ModelSettings()

    model_config = SettingsConfigDict(
        toml_file=files("hw04").joinpath("config.toml"),
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
