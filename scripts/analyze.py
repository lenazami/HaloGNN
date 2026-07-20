# TODO: get rid of this
# ============================================
# scripts/analyze.py - Analysis and evaluation
# ============================================
import matplotlib.pyplot as plt
import torch

from utils import Config, load_data, load_model, get_predictions
from utils.metrics import avg_logprob, avg_rmse, get_coverage, save_coverage_csv
from utils.logger import info, debug, warning, set_verbosity

def analyze_model(cfg: Config):
    """Run analysis for a trained model."""
    logger.info(f"Analyzing {cfg.model_type.value} for {cfg.sim.value} z={cfg.z}")
    
    # Load test data and model
    test_loader = load_data(cfg, only_test=True)
    model = load_model(cfg)
    
    # Get predictions
    truths, predictions, log_probs = get_predictions(cfg, test_loader, model)
    
    # Compute metrics
    centers_lp, avg_lp = avg_logprob(truths.numpy(), log_probs.numpy())
    centers_rmse, rmse = avg_rmse(truths.numpy(), predictions.numpy())
    levels, coverages = get_coverage(cfg)
    
    # Save results
    results_dir = cfg.results_dir / "metrics" / cfg.model_type.value
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save coverage
    coverage_file = results_dir / f"coverage_{cfg.sim.value}_z{cfg.z}.csv"
    save_coverage_csv(levels, coverages, coverage_file)
    
    # Create plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Log probability plot
    axes[0].plot(centers_lp, avg_lp, 'o-')
    axes[0].set_xlabel("True log(M_halo)")
    axes[0].set_ylabel("Average log p(true mass)")
    axes[0].set_title(f"Log Probability - {cfg.sim.value} z={cfg.z}")
    
    # RMSE plot
    axes[1].plot(centers_rmse, rmse, 'o-')
    axes[1].set_xlabel("True log(M_halo)")
    axes[1].set_ylabel("RMSE")
    axes[1].set_title(f"RMSE - {cfg.sim.value} z={cfg.z}")
    
    # Coverage plot
    axes[2].plot(levels, coverages, 'o-')
    axes[2].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[2].set_xlabel("Credibility Level")
    axes[2].set_ylabel("Coverage")
    axes[2].set_title(f"Coverage - {cfg.sim.value} z={cfg.z}")
    
    plt.tight_layout()
    plot_file = cfg.results_dir / "figures" / f"{cfg.model_type.value}_{cfg.sim.value}_z{cfg.z}.png"
    plot_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_file, dpi=150)
    plt.close()
    
    logger.info(f"Results saved to {cfg.results_dir}")
    
    return {
        "avg_logprob": float(np.mean(avg_lp)),
        "avg_rmse": float(np.mean(rmse)),
        "coverage_error": float(np.mean(np.abs(levels - coverages))),
    }

def run_full_analysis(cfg: Config):
    """Run complete analysis pipeline."""
    results = {}
    for obs_only in [False, True]:
        cfg.observables_only = obs_only
        key = f"obs_{obs_only}"
        results[key] = analyze_model(cfg)
    
    # Save summary
    summary_file = cfg.results_dir / "reports" / f"summary_{cfg.model_type.value}_{cfg.sim.value}_z{cfg.z}.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    
    import json
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Analysis complete. Summary saved to {summary_file}")
    
    
    
# TODO: add this later!!!

# ============================================
# scripts/analyze_enhanced.py - Complete analysis pipeline
# ============================================
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Tuple, Dict, List
from pathlib import Path
import json
from scipy import stats

from utils import Config, ModelType, load_data, load_model, get_predictions
from utils.metrics import get_coverage, save_coverage_csv


# ============================================
# Halo Mass Function Analysis
# ============================================
def compute_hmf_with_uncertainties(
    true_masses: np.ndarray,
    predicted_mass_samples: np.ndarray,
    volume: float,
    bins: int = 20,
    mass_range: Tuple[float, float] = (10, 13)
) -> Tuple[np.ndarray, ...]:
    """
    Compute halo mass function with uncertainties from posterior samples.
    
    Args:
        true_masses: True halo masses (log scale)
        predicted_mass_samples: Posterior samples [n_samples, n_halos]
        volume: Simulation volume in (Mpc/h)^3
        bins: Number of mass bins
        mass_range: Range of log masses
    
    Returns:
        bin_centers, true_hmf, pred_hmf_median, pred_hmf_lower, pred_hmf_upper
    """
    mass_bins = np.linspace(mass_range[0], mass_range[1], bins + 1)
    bin_centers = (mass_bins[1:] + mass_bins[:-1]) / 2
    bin_widths = mass_bins[1:] - mass_bins[:-1]
    
    # True HMF
    counts, _ = np.histogram(true_masses, bins=mass_bins)
    true_hmf = counts / (volume * bin_widths)
    
    # HMF from each posterior sample
    n_samples = predicted_mass_samples.shape[0]
    sample_hmfs = np.zeros((n_samples, bins))
    
    for i in range(n_samples):
        counts, _ = np.histogram(predicted_mass_samples[i], bins=mass_bins)
        sample_hmfs[i] = counts / (volume * bin_widths)
    
    # Get percentiles
    pred_hmf_median = np.median(sample_hmfs, axis=0)
    pred_hmf_lower = np.percentile(sample_hmfs, 16, axis=0)
    pred_hmf_upper = np.percentile(sample_hmfs, 84, axis=0)
    
    return bin_centers, true_hmf, pred_hmf_median, pred_hmf_lower, pred_hmf_upper

# ============================================
# Calibration and Sharpness Metrics
# ============================================
def compute_calibration_metrics(
    truths: np.ndarray, 
    predictions: np.ndarray,
    nbins: int = 50
) -> Dict:
    """
    Compute comprehensive calibration metrics.
    
    Metrics include:
    - Average log probability (higher is better - indicates the model assigns high probability to true values)
    - RMSE (lower is better - point estimate accuracy)
    - Bias (should be near 0 - systematic over/under estimation)
    - Interval scores (lower is better - combines coverage and sharpness)
    - CRPS (Continuous Ranked Probability Score - lower is better)
    """
    # Get predictions stats
    pred_mean = predictions.mean(axis=0)
    pred_std = predictions.std(axis=0)
    
    # Basic metrics
    rmse = np.sqrt(np.mean((pred_mean - truths) ** 2))
    bias = np.mean(pred_mean - truths)
    
    # Interval Score (IS) - penalizes both poor coverage and wide intervals
    # For 68% credible interval (1-sigma)
    alpha = 0.32  # 1 - 0.68
    lower = np.percentile(predictions, 16, axis=0)
    upper = np.percentile(predictions, 84, axis=0)
    interval_width = upper - lower
    
    # Penalty for being outside interval
    lower_penalty = (2 / alpha) * (lower - truths) * (truths < lower)
    upper_penalty = (2 / alpha) * (truths - upper) * (truths > upper)
    interval_score = interval_width + lower_penalty + upper_penalty
    mean_interval_score = np.mean(interval_score)
    
    # CRPS - Continuous Ranked Probability Score
    # Approximation using samples
    def crps_ensemble(truth, samples):
        """Calculate CRPS for ensemble predictions."""
        n = len(samples)
        # Term 1: mean absolute error between samples and truth
        term1 = np.mean(np.abs(samples - truth))
        # Term 2: mean absolute difference between samples
        term2 = np.sum(np.abs(samples[:, None] - samples[None, :])) / (2 * n * n)
        return term1 - term2
    
    crps_scores = [crps_ensemble(t, predictions[:, i]) for i, t in enumerate(truths)]
    mean_crps = np.mean(crps_scores)
    
    # Sharpness (average posterior std)
    sharpness = np.mean(pred_std)
    
    # Calibration error (how well do predicted uncertainties match actual errors)
    normalized_errors = (truths - pred_mean) / pred_std
    calibration_slope, _ = np.polyfit(pred_std, np.abs(truths - pred_mean), 1)
    
    return {
        'rmse': rmse,
        'bias': bias,
        'interval_score': mean_interval_score,
        'crps': mean_crps,
        'sharpness': sharpness,
        'calibration_slope': calibration_slope,
        'pred_std_mean': np.mean(pred_std),
        'pred_std_std': np.std(pred_std)
    }

def compute_binned_metrics(
    truths: np.ndarray,
    predictions: np.ndarray, 
    log_probs: np.ndarray,
    nbins: int = 20
) -> Dict:
    """Compute metrics in mass bins for detailed analysis."""
    
    bins = np.linspace(truths.min(), truths.max(), nbins + 1)
    bin_centers = (bins[1:] + bins[:-1]) / 2
    
    pred_mean = predictions.mean(axis=0)
    pred_std = predictions.std(axis=0)
    
    # Initialize arrays
    binned_rmse = np.zeros(nbins)
    binned_bias = np.zeros(nbins)
    binned_sharpness = np.zeros(nbins)
    binned_logprob = np.zeros(nbins)
    binned_counts = np.zeros(nbins)
    
    for i in range(nbins):
        mask = (truths >= bins[i]) & (truths < bins[i + 1])
        if mask.sum() > 0:
            binned_counts[i] = mask.sum()
            binned_rmse[i] = np.sqrt(np.mean((pred_mean[mask] - truths[mask]) ** 2))
            binned_bias[i] = np.mean(pred_mean[mask] - truths[mask])
            binned_sharpness[i] = np.mean(pred_std[mask])
            binned_logprob[i] = np.mean(log_probs[mask])
    
    return {
        'bin_centers': bin_centers,
        'rmse': binned_rmse,
        'bias': binned_bias,
        'sharpness': binned_sharpness,
        'log_prob': binned_logprob,
        'counts': binned_counts
    }

# ============================================
# Plotting Functions
# ============================================
def plot_comprehensive_analysis(
    cfg: Config,
    truths: np.ndarray,
    predictions: np.ndarray,
    log_probs: np.ndarray,
    levels: np.ndarray,
    coverages: np.ndarray,
    save_dir: Path
):
    """Create comprehensive analysis plots."""
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 1. Predictions vs Truth
    ax1 = fig.add_subplot(gs[0, 0])
    pred_mean = predictions.mean(axis=0)
    pred_std = predictions.std(axis=0)
    ax1.errorbar(truths, pred_mean, yerr=pred_std, fmt='.', alpha=0.3, markersize=1)
    ax1.plot([truths.min(), truths.max()], [truths.min(), truths.max()], 'k--')
    ax1.set_xlabel('True log(M)')
    ax1.set_ylabel('Predicted log(M)')
    ax1.set_title('Predictions vs Truth')
    
    # 2. Coverage plot
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(levels, coverages, 'o-', label='Model')
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')
    ax2.set_xlabel('Credibility Level')
    ax2.set_ylabel('Coverage')
    ax2.set_title('Calibration Plot')
    ax2.legend()
    
    # 3. HMF with uncertainties
    ax3 = fig.add_subplot(gs[0, 2])
    volume = cfg.box_size ** 3 / 1e9  # Convert to Gpc^3
    bin_centers, true_hmf, pred_hmf, pred_lower, pred_upper = compute_hmf_with_uncertainties(
        truths, predictions, volume
    )
    ax3.plot(bin_centers, true_hmf, 'k-', label='True', linewidth=2)
    ax3.plot(bin_centers, pred_hmf, 'b-', label='Predicted', linewidth=2)
    ax3.fill_between(bin_centers, pred_lower, pred_upper, alpha=0.3, color='blue')
    ax3.set_yscale('log')
    ax3.set_xlabel('log(M)')
    ax3.set_ylabel('dn/dlogM [(Gpc/h)^-3]')
    ax3.set_title('Halo Mass Function')
    ax3.legend()
    
    # 4. Residuals distribution
    ax4 = fig.add_subplot(gs[0, 3])
    residuals = pred_mean - truths
    ax4.hist(residuals, bins=50, density=True, alpha=0.7, edgecolor='black')
    x = np.linspace(residuals.min(), residuals.max(), 100)
    ax4.plot(x, stats.norm.pdf(x, residuals.mean(), residuals.std()), 'r-', label='Gaussian fit')
    ax4.set_xlabel('Residuals (Pred - True)')
    ax4.set_ylabel('Density')
    ax4.set_title(f'Residuals: μ={residuals.mean():.3f}, σ={residuals.std():.3f}')
    ax4.legend()
    
    # 5-8. Binned metrics
    binned = compute_binned_metrics(truths, predictions, log_probs)
    
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.plot(binned['bin_centers'], binned['rmse'], 'o-')
    ax5.set_xlabel('True log(M)')
    ax5.set_ylabel('RMSE')
    ax5.set_title('RMSE vs Mass')
    
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.plot(binned['bin_centers'], binned['bias'], 'o-')
    ax6.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax6.set_xlabel('True log(M)')
    ax6.set_ylabel('Bias')
    ax6.set_title('Bias vs Mass')
    
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.plot(binned['bin_centers'], binned['sharpness'], 'o-')
    ax7.set_xlabel('True log(M)')
    ax7.set_ylabel('Avg Posterior Std')
    ax7.set_title('Uncertainty vs Mass')
    
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.plot(binned['bin_centers'], binned['log_prob'], 'o-')
    ax8.set_xlabel('True log(M)')
    ax8.set_ylabel('Avg Log Probability')
    ax8.set_title('Log Probability vs Mass')
    
    # 9. Uncertainty calibration
    ax9 = fig.add_subplot(gs[2, 0])
    normalized_residuals = residuals / pred_std
    ax9.hist(normalized_residuals, bins=50, density=True, alpha=0.7, edgecolor='black')
    x = np.linspace(-4, 4, 100)
    ax9.plot(x, stats.norm.pdf(x, 0, 1), 'r-', label='N(0,1)')
    ax9.set_xlabel('Normalized Residuals')
    ax9.set_ylabel('Density')
    ax9.set_title('Uncertainty Calibration')
    ax9.legend()
    
    # 10. Predicted uncertainty vs actual error
    ax10 = fig.add_subplot(gs[2, 1])
    ax10.hexbin(pred_std, np.abs(residuals), gridsize=30, cmap='Blues')
    # Add diagonal line for perfect calibration
    max_val = max(pred_std.max(), np.abs(residuals).max())
    ax10.plot([0, max_val], [0, max_val], 'r--', alpha=0.5)
    ax10.set_xlabel('Predicted Uncertainty (std)')
    ax10.set_ylabel('Actual Error |Pred - True|')
    ax10.set_title('Predicted vs Actual Uncertainty')
    
    # 11. Q-Q plot
    ax11 = fig.add_subplot(gs[2, 2])
    stats.probplot(normalized_residuals, dist="norm", plot=ax11)
    ax11.set_title('Q-Q Plot (Normalized Residuals)')
    
    # 12. Metrics summary text
    ax12 = fig.add_subplot(gs[2, 3])
    ax12.axis('off')
    metrics = compute_calibration_metrics(truths, predictions)
    text = f"""Summary Metrics:
    RMSE: {metrics['rmse']:.4f}
    Bias: {metrics['bias']:.4f}
    Interval Score: {metrics['interval_score']:.4f}
    CRPS: {metrics['crps']:.4f}
    Sharpness: {metrics['sharpness']:.4f}
    Calibration Slope: {metrics['calibration_slope']:.4f}

    Coverage Error: {np.mean(np.abs(levels - coverages)):.4f}
    Mean Log Prob: {np.mean(log_probs):.4f}
    """
    ax12.text(0.1, 0.9, text, transform=ax12.transAxes, 
             fontsize=10, verticalalignment='top', family='monospace')
    
    # Overall title
    fig.suptitle(f'{cfg.model_type.value} - {cfg.sim.value} z={cfg.z}', fontsize=16)
    
    # Save
    plt.tight_layout()
    save_path = save_dir / f'comprehensive_analysis_{cfg.model_type.value}_{cfg.sim.value}_z{cfg.z}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    info(f"Saved analysis to {save_path}")
    
    return metrics

# ============================================
# Main Analysis Function
# ============================================
def run_complete_analysis(cfg: Config) -> Dict:
    """Run complete analysis pipeline with all metrics."""
    
    logger.info(f"Running complete analysis for {cfg.model_type.value} - {cfg.sim.value} z={cfg.z}")
    
    # Load test data and model
    test_loader = load_data(cfg, only_test=True, batch_size=64)
    model = load_model(cfg)
    
    # Get predictions
    truths, predictions, log_probs = get_predictions(cfg, test_loader, model)
    
    # Ensure numpy arrays
    truths = truths.cpu().numpy() if torch.is_tensor(truths) else truths
    predictions = predictions.cpu().numpy() if torch.is_tensor(predictions) else predictions
    log_probs = log_probs.cpu().numpy() if torch.is_tensor(log_probs) else log_probs
    
    # Get coverage
    levels, coverages = get_coverage(cfg)
    levels = levels.cpu().numpy() if torch.is_tensor(levels) else levels
    coverages = coverages.cpu().numpy() if torch.is_tensor(coverages) else coverages
    
    # Create output directories
    metrics_dir = cfg.results_dir / "metrics" / cfg.model_type.value
    figures_dir = cfg.results_dir / "figures" / cfg.model_type.value
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Save coverage data
    coverage_file = metrics_dir / f"coverage_{cfg.sim.value}_z{cfg.z}.csv"
    save_coverage_csv(levels, coverages, coverage_file)
    
    # Compute all metrics
    metrics = compute_calibration_metrics(truths, predictions)
    binned_metrics = compute_binned_metrics(truths, predictions, log_probs)
    
    # Add HMF metrics
    volume = cfg.box_size ** 3 / 1e9  # Convert to Gpc^3
    bin_centers, true_hmf, pred_hmf, _, _ = compute_hmf_with_uncertainties(
        truths, predictions, volume
    )
    hmf_error = np.mean(np.abs(np.log10(pred_hmf + 1e-10) - np.log10(true_hmf + 1e-10)))
    metrics['hmf_error'] = hmf_error
    
    # Create comprehensive plots
    plot_comprehensive_analysis(
        cfg, truths, predictions, log_probs, 
        levels, coverages, figures_dir
    )
    
    # Save detailed metrics
    detailed_metrics = {
        'summary': metrics,
        'binned': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                  for k, v in binned_metrics.items()},
        'coverage': {
            'levels': levels.tolist(),
            'coverages': coverages.tolist(),
            'mean_error': float(np.mean(np.abs(levels - coverages)))
        },
        'config': {
            'model': cfg.model_type.value,
            'sim': cfg.sim.value,
            'redshift': cfg.z,
            'observables_only': cfg.observables_only,
            'hm_present': cfg.hm_present
        }
    }
    
    # Save JSON report
    report_file = metrics_dir / f"detailed_report_{cfg.sim.value}_z{cfg.z}.json"
    with open(report_file, 'w') as f:
        json.dump(detailed_metrics, f, indent=2)
    
    logger.info(f"Analysis complete. Results saved to {cfg.results_dir}")
    
    return metrics

# ============================================
# Additional Analysis: Model Comparison
# ============================================
def compare_models(cfg_fcn: Config, cfg_gnn: Config):
    """Compare FCN and GNN models on the same data."""
    
    results = {}
    for cfg, name in [(cfg_fcn, 'FCN'), (cfg_gnn, 'GNN')]:
        cfg.observables_only = False  # Compare full feature models
        results[name] = run_complete_analysis(cfg)
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    metrics_to_compare = ['rmse', 'bias', 'interval_score', 'crps', 'sharpness', 'hmf_error']
    
    for ax, metric in zip(axes.flat, metrics_to_compare):
        fcn_val = results['FCN'][metric]
        gnn_val = results['GNN'][metric]
        
        ax.bar(['FCN', 'GNN'], [fcn_val, gnn_val])
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric}: FCN={fcn_val:.4f}, GNN={gnn_val:.4f}')
        
        # Add percentage difference
        pct_diff = 100 * (gnn_val - fcn_val) / fcn_val
        color = 'green' if pct_diff < 0 else 'red'
        ax.text(0.5, 0.95, f'Δ = {pct_diff:.1f}%', 
               transform=ax.transAxes, ha='center', color=color)
    
    plt.suptitle(f'Model Comparison - {cfg_fcn.sim.value} z={cfg_fcn.z}')
    plt.tight_layout()
    
    comparison_file = cfg_fcn.results_dir / 'figures' / f'model_comparison_{cfg_fcn.sim.value}_z{cfg_fcn.z}.png'
    plt.savefig(comparison_file, dpi=150)
    plt.close()
    
    logger.info(f"Model comparison saved to {comparison_file}")
    
    return results