"""
Pure PyTorch model wrapper.
Provides model initialization and optimizer configuration without Lightning dependencies.
"""

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from utils.lr_scheduler import linear_warmup_cosine_decay


class PytorchModelWrapper(nn.Module):
    """
    Wraps a model and provides optimizer/scheduler configuration.
    Equivalent to ClassificationModule but without Lightning.
    """

    def __init__(
            self,
            model: nn.Module,
            n_classes: int,
            lr: float = 0.001,
            weight_decay: float = 0.0,
            optimizer: str = "adam",
            scheduler: bool = False,
            max_epochs: int = 1000,
            warmup_epochs: int = 20,
            beta_1: float = 0.9,
            beta_2: float = 0.999,
            **kwargs
    ):
        super().__init__()
        self.model = model
        self.n_classes = n_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.use_scheduler = scheduler
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.beta_1 = beta_1
        self.beta_2 = beta_2

        # Store extra kwargs for flexibility
        self.hparams = {
            'n_classes': n_classes,
            'lr': lr,
            'weight_decay': weight_decay,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'max_epochs': max_epochs,
            'warmup_epochs': warmup_epochs,
            'beta_1': beta_1,
            'beta_2': beta_2,
            **kwargs
        }

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        """
        Configure optimizer and optional scheduler.
        Returns: (optimizer, scheduler or None)
        """
        betas = (self.beta_1, self.beta_2)

        if self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.lr,
                betas=betas,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name == "adamW":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.lr,
                betas=betas,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        else:
            raise NotImplementedError(f"Optimizer {self.optimizer_name} not supported")

        scheduler = None
        if self.use_scheduler:
            scheduler = LambdaLR(
                optimizer,
                linear_warmup_cosine_decay(self.warmup_epochs, self.max_epochs)
            )

        return optimizer, scheduler


def create_model(model_cls, config):
    """
    Create a wrapped model from a model class and config.

    Args:
        model_cls: The model class (e.g., TCFormer, EEGNet)
        config: Configuration dictionary

    Returns:
        PytorchModelWrapper instance
    """
    # Get the underlying model module class name
    # Models like TCFormer inherit from ClassificationModule and wrap a Module
    # We need to extract/create the underlying module directly

    model_kwargs = config["model_kwargs"].copy()
    n_channels = config["n_channels"]
    n_classes = config["n_classes"]
    max_epochs = config["max_epochs"]

    # Separate training parameters from model architecture parameters
    # These go to the wrapper, not the underlying module
    training_params = ["lr", "weight_decay", "optimizer", "scheduler",
                       "warmup_epochs", "warmup_epochs_loso", "beta_1", "beta_2"]
    wrapper_kwargs = {}
    for param in training_params:
        if param in model_kwargs:
            wrapper_kwargs[param] = model_kwargs.pop(param)

    # Import the module class based on model name
    model_name = model_cls.lower()

    # Create the underlying module (not the Lightning wrapper)
    # model_kwargs now only contains architecture parameters
    module = _create_module(model_name, n_channels, n_classes, model_kwargs)

    # Wrap with our pure PyTorch wrapper
    wrapper = PytorchModelWrapper(
        model=module,
        n_classes=n_classes,
        max_epochs=max_epochs,
        **wrapper_kwargs
    )

    return wrapper


def _create_module(model_name: str, n_channels: int, n_classes: int, model_kwargs: dict):
    """Create the underlying nn.Module for a given model name."""

    if model_name == "tcformer":
        from models.tcformer import TCFormerModule
        return TCFormerModule(n_channels=n_channels, n_classes=n_classes, **model_kwargs)

    else:
        raise NotImplementedError(f"Model {model_name} not supported in pure PyTorch mode")
