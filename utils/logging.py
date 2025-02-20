import wandb
from typing import Any, Dict, Optional, Union
from pathlib import Path

class WandbLogger:
    """Wrapper for Weights & Biases logging."""
    
    def __init__(
        self,
        project: str,
        config: Dict[str, Any],
        name: Optional[str] = None,
        tags: Optional[list[str]] = None,
        dir: Optional[Union[str, Path]] = None
    ):
        """Initialize W&B logger.
        
        Args:
            project: W&B project name.
            config: Configuration dictionary.
            name: Run name (optional).
            tags: List of tags for the run (optional).
            dir: Directory to store W&B files (optional).
        """
        self.run = wandb.init(
            project=project,
            name=name,
            config=config,
            tags=tags,
            dir=dir
        )
        
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to W&B.
        
        Args:
            metrics: Dictionary of metrics to log.
            step: Optional step number.
        """
        if step is not None:
            metrics["step"] = step
        wandb.log(metrics)
        
    def log_model(self, model_path: Union[str, Path], name: str):
        """Log a model to W&B.
        
        Args:
            model_path: Path to model files.
            name: Name for the model artifact.
        """
        artifact = wandb.Artifact(name=name, type="model")
        artifact.add_dir(model_path)
        self.run.log_artifact(artifact)
        
    def finish(self):
        """End the W&B run."""
        wandb.finish()

def setup_logging(
    project_name: str,
    config: Dict[str, Any],
    run_name: Optional[str] = None,
    tags: Optional[list[str]] = None,
    log_dir: Optional[Union[str, Path]] = None
) -> WandbLogger:
    """Set up logging for a training run.
    
    Args:
        project_name: Name of the project.
        config: Configuration dictionary.
        run_name: Optional name for this run.
        tags: Optional list of tags.
        log_dir: Optional directory for log files.
        
    Returns:
        WandbLogger: Initialized logger.
    """
    return WandbLogger(
        project=project_name,
        config=config,
        name=run_name,
        tags=tags,
        dir=log_dir
    )

def log_game_metrics(
    logger: WandbLogger,
    game_idx: int,
    game_reward: float,
    num_moves: int,
    loss: Optional[float] = None,
    additional_metrics: Optional[Dict[str, Any]] = None
):
    """Log metrics for a single game.
    
    Args:
        logger: The logger instance.
        game_idx: Game index/number.
        game_reward: Total reward for the game.
        num_moves: Number of moves in the game.
        loss: Optional loss value.
        additional_metrics: Optional additional metrics to log.
    """
    metrics = {
        "game": game_idx,
        "game_reward": game_reward,
        "num_moves": num_moves,
    }
    
    if loss is not None:
        metrics["loss"] = loss
        
    if additional_metrics:
        metrics.update(additional_metrics)
        
    logger.log(metrics)

def log_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Log training configuration and return it with any modifications.
    
    Args:
        config: Training configuration dictionary.
        
    Returns:
        Dict[str, Any]: Possibly modified configuration dictionary.
    """
    # Add any derived configuration values
    if "total_steps" not in config and "num_games" in config and "moves_per_game" in config:
        config["total_steps"] = config["num_games"] * config["moves_per_game"]
        
    # Add any computed parameters
    if "learning_rate" in config and "warmup_steps" in config:
        config["total_warmup_steps"] = int(config["warmup_steps"] * config.get("total_steps", 1000))
        
    return config

def log_completion(
    logger: WandbLogger,
    prompt: str,
    completion: str,
    reward: float,
    step: int
):
    """Log a model completion with its reward.
    
    Args:
        logger: The logger instance.
        prompt: Input prompt.
        completion: Model's completion.
        reward: Reward value.
        step: Current step number.
    """
    logger.log({
        "completions": wandb.Table(
            columns=["step", "prompt", "completion", "reward"],
            data=[[str(step), prompt, completion, reward]]
        )
    })
