from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn
from lightning import Trainer

from halo.models.flow import FlowNetwork
from halo.data import normalize
from typing import Any, Sequence


CONTEXT_DIM = 32
HIDDEN_DIMS = (128, 128, 128)


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: Sequence[int],
    *,
    use_batch_norm: bool = True,
    dropout_rate: float = 0.0,
) -> nn.Sequential:
    """Creates a multi-layer perceptron (MLP) with optional batch normalization and dropout."""
    dims = (input_dim, *tuple(hidden_dims), output_dim)
    if any(dim < 1 for dim in dims):
        raise ValueError(f"all dimensions must be positive, got {dims}")
    if not 0.0 <= dropout_rate < 1.0:
        raise ValueError(f"dropout must be in [0, 1), got {dropout_rate}")

    layers: list[nn.Module] = []
    for index, (in_dim, out_dim) in enumerate(zip(dims, dims[1:])):
        layers.append(nn.Linear(in_dim, out_dim))
        if index < len(dims) - 2:
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.SiLU())
            if dropout_rate:
                layers.append(nn.Dropout(dropout_rate))
    return nn.Sequential(*layers)


def build_encoder(
    model: str,
    model_kwargs: dict[str, Any]
) -> nn.Module:
    """
    model: graph or full
    """
    from halo.models.fcn import FullNetwork
    from halo.models.gnn import GraphNetwork
    return locals()[f"{model.capitalize()}Network"](**model_kwargs)

def build_model(
    model_type: str,
    model_kwargs: dict[str, Any],
    **flow_kwargs: Any,
) -> FlowNetwork:
    """
    model: graph or full
    """
    encoder = build_encoder(model_type, model_kwargs)
    model = FlowNetwork(encoder=encoder, **flow_kwargs)
    model.set_model_config(model_type, model_kwargs)
    return model


def get_model(model_config, gal_features_dim: int):
    encoder_kwargs = {
        "gal_features_dim": gal_features_dim,
        "context_dim": CONTEXT_DIM,
        "hidden_dims": HIDDEN_DIMS
    }
    return build_model(
        model_config.name,
        encoder_kwargs,
        context_dim=CONTEXT_DIM,
    )
    

def load_model(checkpoint_file: Path) -> FlowNetwork:
    checkpoint = torch.load(checkpoint_file, map_location="cpu")

    encoder = build_encoder(
        checkpoint["model_type"],
        checkpoint["model_kwargs"],
    )

    model = FlowNetwork.load_from_checkpoint(
        checkpoint_file,
        encoder=encoder,
        map_location="cpu",
    )

    model.set_model_config(
        checkpoint["model_type"],
        checkpoint["model_kwargs"],
    )

    return model.eval()

def get_predictions(
    model: FlowNetwork,
    test_loader,
    trainer: Trainer,
    mean: Tensor,
    std: Tensor,
):
    outputs = trainer.predict(model, test_loader)

    truths = torch.cat([out["target"].cpu() for out in outputs], dim=0)
    samples = torch.cat([out["samples"].cpu() for out in outputs], dim=1)
    log_probs = torch.cat([out["log_prob"].cpu() for out in outputs], dim=0)

    mean = mean.cpu()
    std = std.cpu()

    truths = normalize(truths, mean, std, inverse=True)
    samples = normalize(samples, mean, std, inverse=True)

    return truths, samples, log_probs