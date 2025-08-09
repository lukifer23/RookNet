"""Pydantic models describing the project's configuration.

These models mirror the structure of ``configs/config.v2.yaml`` and provide
type validation as well as sensible defaults for optional fields.

The top-level :class:`ConfigModel` requires all major sections (``model``,
``training`` …), ensuring that a missing section results in a clear validation
error.  Individual fields inside those sections carry defaults so that minor
omissions fall back to reasonable values.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, ConfigDict


class ChessTransformerConfig(BaseModel):
    input_channels: int = 12
    cnn_channels: int = 384
    cnn_blocks: int = 20
    transformer_layers: int = 8
    attention_heads: int = 8


class ModelConfig(BaseModel):
    type: str = Field("ChessTransformer")
    chess_transformer: ChessTransformerConfig = ChessTransformerConfig()


class TemperatureConfig(BaseModel):
    initial: float = 1.0
    final: float = 0.1
    decay_after_games: int = 1000


class AlphaZeroConfig(BaseModel):
    iterations: int = 1000
    games_per_iteration: int = 50
    epochs_per_iteration: int = 2
    batch_size: int = 256
    replay_buffer_size: int = 500_000
    min_replay_buffer_size: int = 10_000
    evaluation_interval: int = 5
    evaluation_games: int = 20
    opponent_update_threshold: float = 0.55
    learning_rate: float = 0.0001
    min_learning_rate: float = 0.00001
    lr_decay_steps: int = 500
    temperature: TemperatureConfig = TemperatureConfig()
    gradient_clip: float = 1.0


class CheckpointsConfig(BaseModel):
    dir: str = "models/alpha_zero_checkpoints"
    save_interval: int = 2
    latest_player_model: str = "latest_player.pt"
    best_opponent_model: str = "best_opponent.pt"
    initial_opponent_model: str = ""


class MCTSConfig(BaseModel):
    simulations: int = 100
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.6
    dirichlet_epsilon: float = 0.40


class OptimizerConfig(BaseModel):
    type: str = "AdamW"
    weight_decay: float = 0.0001
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1.0e-8


class SchedulerConfig(BaseModel):
    type: str = "CosineAnnealingWarmRestarts"
    T_0: int = 10
    T_mult: int = 2


class TrainingConfig(BaseModel):
    alpha_zero: AlphaZeroConfig = AlphaZeroConfig()
    checkpoints: CheckpointsConfig = CheckpointsConfig()
    mcts: MCTSConfig = MCTSConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    gradient_clip: float = 1.0
    use_amp: bool = True


class DataConfig(BaseModel):
    replay_buffer_dir: str = "data/replay"
    wds_shard_pattern: str = "shard-%06d.tar"
    use_webdataset: bool = False


class StockfishConfig(BaseModel):
    path: str = "stockfish"
    default_depth: int = 15


class LC0Config(BaseModel):
    path: str = "lc0"
    default_nodes: int = 80_000


class EvaluationConfig(BaseModel):
    default_model_to_evaluate: str = (
        "models/alpha_zero_checkpoints/latest_player.pt"
    )
    stockfish: StockfishConfig = StockfishConfig()
    lc0: LC0Config = LC0Config()


class ApiDefaultsConfig(BaseModel):
    stockfish_depth: int = 10
    stockfish_time_limit: float = 1.0
    lc0_nodes: int = 24_000
    lc0_time_limit: float = 3.0
    ai_model_depth: int = 5


class WebGUIConfig(BaseModel):
    model_path: str = "default"
    api_defaults: ApiDefaultsConfig = ApiDefaultsConfig()
    quiet_logging: bool = True


class SystemConfig(BaseModel):
    device: str = "cuda"
    compile_model: bool = False
    self_play_workers: int = 5
    evaluation_workers: int = 5
    dataloader_workers: int = 0
    use_manager_queue: bool = True
    num_workers: int = 4
    pin_memory: bool = True


class LoggingConfig(BaseModel):
    log_dir: str = "logs"
    use_wandb: bool = False
    wandb_project: str = "NativeChessTransformer"
    wandb_entity: str | None = None


class ConfigModel(BaseModel):
    """Complete configuration model."""

    model_config = ConfigDict(extra="forbid")

    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    evaluation: EvaluationConfig
    web_gui: WebGUIConfig
    system: SystemConfig
    logging: LoggingConfig

