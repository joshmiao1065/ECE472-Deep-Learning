from pathlib import Path
from importlib.resources import files
from typing import Tuple

from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


class DataSettings(BaseModel):
    """Settings for data generation."""
    num_features: int = 2
    num_outputs: int = 1
    num_samples: int = 2000  # 1000 per spiral
    noise: float = 0.1


class ModelSettings(BaseModel):
    """Settings for model architecture."""
    num_hl: int = 16
    hl_width: int = 256
    latent_dim: int = 2048  # SAE latent dimension


class MLPTrainingSettings(BaseModel):
    """Settings for MLP classifier training."""
    batch_size: int = 64
    num_iters: int = 2000
    learning_rate: float = 0.001
    epsilon: float = 1e-6
    # Cosine decay settings
    lr_decay_alpha: float = 0.01
    use_cosine_decay: bool = False # the decay was detrimental to training MLP idk why and training the MLP isnt a big enough deal to warrant this additional effort


class SAETrainingSettings(BaseModel):
    """Settings for SAE training."""
    batch_size: int = 128
    num_iters: int = 5000  
    learning_rate: float = 0.01
    lambda_sparsity: float = 0.001
    # Cosine decay settings
    lr_decay_alpha: float = 0.005
    use_cosine_decay: bool = False #for some reason it either does a really good job of minimizing recon loss or sparse loss but not both


class PlottingSettings(BaseModel):
    """Settings for plotting."""
    figsize: Tuple[int, int] = (10, 10)
    dpi: int = 200
    output_dir: Path = Path("hw07/artifacts")
    linspace: int = 1000


class AppSettings(BaseSettings):
    """Main application settings."""
    debug: bool = False
    random_seed: int = 427
    data: DataSettings = DataSettings()
    model: ModelSettings = ModelSettings()
    mlp_training: MLPTrainingSettings = MLPTrainingSettings()
    sae_training: SAETrainingSettings = SAETrainingSettings()
    plotting: PlottingSettings = PlottingSettings()

    model_config = SettingsConfigDict(
        toml_file=files("hw07").joinpath("config.toml") if files("hw07").joinpath("config.toml").is_file() else None,
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
        """Set the priority of settings sources."""
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