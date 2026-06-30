# utils / metrics.py

# -----------------
# Imports
# -----------------
# general
import numpy as np
from typing import Tuple, List, Union, Sequence, Dict, TypeVar
from collections import defaultdict
import pandas as pd
from pathlib import Path

# torch
import torch
from torch.utils.data import DataLoader

# coverage
from lampe.diagnostics import expected_coverage_mc

# local
from .data_utils import load_data
from .model_utils import load_model
from .config import Config, ModelType
from models.gnn_model import GraphModel
from models.fcn_model import FlowModel

device = "cuda" if torch.cuda.is_available() else "cpu"

__all__ = [
    "avg_logprob",
    "avg_rmse",
    "plot_avg_metric",
    "plot_all_redshifts",
    "get_coverage",
    "compute_coverages",
    "save_coverage_csv",
    "plot_coverage"
]

# -----------------
# Constants
# -----------------

T = TypeVar("T")
Coverage = Tuple[torch.Tensor, torch.Tensor]
CoverageGrid = Dict[str, Dict[str, Dict[int, Coverage]]]

# -----------------
# Functions
# -----------------

# ---- Sharpness metrics ----

def _bin_(data, nbins=50):
    """ Returns bins."""
    bins = np.linspace(data.min(), data.max(), nbins+1)
    centers = 0.5*(bins[:-1] + bins[1:])
    bin_idx = np.digitize(data, bins)
    
    return bins, centers, bin_idx

def avg_logprob(truths, log_probs, nbins=50):
    """ Computes average log probability per bin."""    
    bins, centers, bin_idx = _bin_(truths, nbins)
    avg_logp = [
        log_probs[(truths>=bins[i]) & (truths<bins[i+1])].mean()
        for i in range(nbins)
    ]
    
    return centers, avg_logp

def avg_rmse(truths, samples,nbins=50):
    """Compute RMSE per bin."""
    bins, centers, bin_idx = _bin_(truths, nbins)
    pred_means = samples.mean(axis=0)
    
    rmse = []
    for b in range(1, nbins+1):
        mask = bin_idx == b
        rmse.append(np.sqrt(np.mean((pred_means[mask] - truths[mask])**2, axis=1)))
    
    return centers, rmse

def plot_avg_metric(centers, metric, z, ax):
    """Plot average metric per bin."""    
    ax.plot(centers, metric, marker='o', label=f"z={z}")
    
def plot_all_redshifts(truths, log_probs, samples):
    import matplotlib.pyplot as plt
    
    _, ax = plt.subplots()
    
    for z in [3,4,5,6]:
        plot_avg_metric(truths.numpy(), log_probs.numpy(), z, ax)

    ax.set_xlabel("True log(HaloMass)")
    ax.set_ylabel("Average log p(true mass)")
    ax.legend(title="Redshift")
    plt.show()

# ---- Coverage ----

def _get_lampe_pairs(
    cfg: Config,
    dataloader: DataLoader,
    model: Union[GraphModel, FlowModel]
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Generate (true, posterior) pairs for coverage analysis.
    """
    model = model.to(device).eval()

    if cfg.model_type == ModelType.GNN:
        def fetch(batch):
            batch = batch.to(device)
            return batch.y, model(batch)
    else:
        def fetch(batch):
            x, y = batch
            return y.to(device), model(x.to(device))

    thetas = []
    samples = []
    with torch.no_grad():
        for batch in dataloader:
            theta_i, sample_i = fetch(batch)
            thetas.append(theta_i)
            samples.append(sample_i)

    theta = torch.cat(thetas, dim=0)
    samples = torch.cat(samples, dim=0)

    return list(zip(theta, samples))

def get_coverage(
    cfg: Config
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates posterior coverage of a trained model.
    """
    dataloader = load_data(cfg, only_test=True)
    model = load_model(cfg)
    pairs = _get_lampe_pairs(cfg, dataloader, model)

    return expected_coverage_mc(model.flow, pairs, device=device)


def _ensure_seq(x: Union[T, Sequence[T]]) -> Sequence[T]:
    return x if isinstance(x, (list, tuple)) else [x]


def compute_coverages(
    model_types: Union[str, Sequence[str]],
    suites:      Union[str, Sequence[str]],
    redshifts:   Union[int, Sequence[int]],
    data_dir:    str,
    model_dir:   str,
) -> CoverageGrid:
    """
    Compute coverage across any axis of the form:
    model type x suite x redshift.

    Returns:
      results[model_type][suite][z] = (levels, covers)
    """
    model_types = _ensure_seq(model_types)
    suites = _ensure_seq(suites)
    redshifts = _ensure_seq(redshifts)

    results: CoverageGrid = defaultdict(lambda: defaultdict(dict))

    for m in model_types:
        for s in suites:
            cfg = Config(
                data_dir=data_dir,
                ckpt_dir=model_dir,
                model_type=m,
                sim=s,
                z=redshifts[0],
            )
            for z in redshifts:
                cfg.z = z
                lvl, cvr = get_coverage(cfg)
                results[m][s][z] = (lvl, cvr)

    return results


def _write_df_to_csv(
    df: pd.DataFrame, 
    path: Path
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def save_coverage_csv(
    levels: Sequence[float],
    covers:  Sequence[float],
    filepath: Union[str, Path],
) -> Path:
    df = pd.DataFrame({"level": levels, "coverage": covers})
    return _write_df_to_csv(df, Path(filepath))


def plot_coverage(
    levels: Sequence[float], 
    covers:  Sequence[float],
    ax
):
    ax.plot(levels, covers, marker="o")