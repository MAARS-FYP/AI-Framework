"""
Test script for bandwidth extraction from STFT data.

This script validates the neurosymbolic bandwidth extraction algorithm
by loading STFT spectrums from the dataset and comparing extracted
bandwidth classes against the ground truth labels in the CSV file.

Usage:
    python -m ai_framework.tests.test_bandwidth_extraction
    
    # Or with specific options:
    python -m ai_framework.tests.test_bandwidth_extraction --num-samples 20
    python -m ai_framework.tests.test_bandwidth_extraction --visualize
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ai_framework.core.dsp import (
    extract_bandwidth_from_stft,
    compute_psd_from_stft,
    classify_bandwidth,
    BandwidthConfig,
    BandwidthResult,
)


def bandwidth_hz_to_class_name(bandwidth_hz: float) -> str:
    bw = float(bandwidth_hz)
    if bw == 1_000_000.0:
        return "1MHz"
    if bw == 10_000_000.0:
        return "10MHz"
    if bw == 20_000_000.0:
        return "20MHz"
    raise ValueError(f"Unsupported Bandwidth_Hz value: {bandwidth_hz}")


def load_dataset_info(data_dir: Path) -> pd.DataFrame:
    """Load the optimal control dataset CSV."""
    csv_path = data_dir / "optimal_control_dataset.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


def load_stft_complex(data_dir: Path, filename: str) -> np.ndarray:
    """Load a complex STFT file from the stft_complex folder."""
    stft_path = data_dir / "stft_complex" / filename
    if not stft_path.exists():
        raise FileNotFoundError(f"STFT file not found: {stft_path}")
    return np.load(stft_path)


def run_bandwidth_test(
    num_samples: int = 10,
    sample_rate_hz: float = 125e6,
    n_fft: int = 2048,
    threshold_db: float = 3.0,
    verbose: bool = True,
    visualize: bool = False,
) -> Dict[str, any]:
    """
    Run bandwidth extraction test on STFT samples.
    
    Args:
        num_samples: Number of samples to test.
        sample_rate_hz: Sample rate assumption for frequency calculation.
        n_fft: FFT size assumption.
        threshold_db: Threshold in dB below peak for cutoff detection.
        verbose: Print detailed results.
        visualize: Generate visualization plots (requires matplotlib).
    
    Returns:
        Dictionary with test results including accuracy and per-sample details.
    """
    # Determine data directory
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "dataset" / "data"
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    print("=" * 70)
    print("BANDWIDTH EXTRACTION TEST - Neurosymbolic Feature Extraction")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Sample Rate: {sample_rate_hz / 1e6:.1f} MHz")
    print(f"  FFT Size: {n_fft}")
    print(f"  Threshold: -{threshold_db} dB (half-power point)")
    print(f"  Data Directory: {data_dir}")
    print()
    
    # Load dataset info
    df = load_dataset_info(data_dir)
    df = df.copy()
    df["Signal_BW_Class"] = df["Bandwidth_Hz"].apply(bandwidth_hz_to_class_name)
    print(f"Loaded dataset with {len(df)} samples")
    print(f"Bandwidth class distribution:")
    print(df['Signal_BW_Class'].value_counts().to_string())
    print()
    
    # Limit samples if needed
    test_samples = min(num_samples, len(df))
    
    # Create bandwidth config
    config = BandwidthConfig(
        sample_rate_hz=sample_rate_hz,
        n_fft=n_fft,
        threshold_db=threshold_db,
    )
    
    # Results storage
    results = []
    correct = 0
    total = 0
    
    # Class-wise tracking
    class_correct: Dict[str, int] = {'1MHz': 0, '10MHz': 0, '20MHz': 0}
    class_total: Dict[str, int] = {'1MHz': 0, '10MHz': 0, '20MHz': 0}
    
    print("-" * 70)
    print(f"Testing {test_samples} samples...")
    print("-" * 70)
    
    if verbose:
        print(f"\n{'Idx':<5} {'File':<25} {'True BW':<10} {'Pred BW':<10} "
              f"{'Low (MHz)':<12} {'High (MHz)':<12} {'BW (MHz)':<12} {'Match'}")
        print("-" * 100)
    
    visualization_data = []
    
    for idx in range(test_samples):
        row = df.iloc[idx]
        
        # Get ground truth
        true_bw_class = row['Signal_BW_Class']
        true_bw_hz = row['Bandwidth_Hz']
        stft_file = row['STFT_Complex_File']
        
        try:
            # Load STFT data
            stft_data = load_stft_complex(data_dir, stft_file)
            
            # Extract bandwidth
            result = extract_bandwidth_from_stft(
                stft_data,
                config=config,
                return_debug_info=visualize,
            )
            
            # Compare
            pred_bw_class = result.bandwidth_class
            is_correct = (pred_bw_class == true_bw_class)
            
            # Update counts
            total += 1
            if is_correct:
                correct += 1
            
            if true_bw_class in class_total:
                class_total[true_bw_class] += 1
                if is_correct:
                    class_correct[true_bw_class] += 1
            
            # Store result
            sample_result = {
                'index': idx,
                'file': stft_file,
                'true_bw_class': true_bw_class,
                'true_bw_hz': true_bw_hz,
                'pred_bw_class': pred_bw_class,
                'pred_low_hz': result.low_cutoff_hz,
                'pred_high_hz': result.high_cutoff_hz,
                'pred_bw_hz': result.bandwidth_hz,
                'is_correct': is_correct,
            }
            results.append(sample_result)
            
            if visualize and result.psd_db is not None:
                visualization_data.append({
                    'idx': idx,
                    'psd_db': result.psd_db,
                    'freq_hz': result.freq_axis_hz,
                    'true_class': true_bw_class,
                    'pred_class': pred_bw_class,
                    'low_cutoff': result.low_cutoff_hz,
                    'high_cutoff': result.high_cutoff_hz,
                })
            
            if verbose:
                match_str = "✓" if is_correct else "✗"
                print(f"{idx:<5} {stft_file:<25} {true_bw_class:<10} {pred_bw_class:<10} "
                      f"{result.low_cutoff_hz/1e6:<12.3f} {result.high_cutoff_hz/1e6:<12.3f} "
                      f"{result.bandwidth_hz/1e6:<12.3f} {match_str}")
                
        except Exception as e:
            print(f"Error processing sample {idx} ({stft_file}): {e}")
            continue
    
    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nOverall Accuracy: {correct}/{total} = {accuracy*100:.1f}%")
    print()
    print("Per-Class Accuracy:")
    for cls in ['1MHz', '10MHz', '20MHz']:
        cls_acc = class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0
        print(f"  {cls}: {class_correct[cls]}/{class_total[cls]} = {cls_acc*100:.1f}%")
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    print("                 Predicted")
    print("              1MHz   10MHz   20MHz")
    
    confusion = {
        '1MHz': {'1MHz': 0, '10MHz': 0, '20MHz': 0},
        '10MHz': {'1MHz': 0, '10MHz': 0, '20MHz': 0},
        '20MHz': {'1MHz': 0, '10MHz': 0, '20MHz': 0},
    }
    
    for r in results:
        true_cls = r['true_bw_class']
        pred_cls = r['pred_bw_class']
        if true_cls in confusion and pred_cls in confusion[true_cls]:
            confusion[true_cls][pred_cls] += 1
    
    for true_cls in ['1MHz', '10MHz', '20MHz']:
        row_str = f"  True {true_cls:>5}:"
        for pred_cls in ['1MHz', '10MHz', '20MHz']:
            row_str += f"  {confusion[true_cls][pred_cls]:>5}"
        print(row_str)
    
    # Visualize if requested
    if visualize and visualization_data:
        try:
            visualize_bandwidth_extraction(visualization_data)
        except ImportError:
            print("\nVisualization skipped: matplotlib not available")
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'class_accuracy': {
            cls: class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0
            for cls in class_correct
        },
        'results': results,
        'confusion_matrix': confusion,
    }


def visualize_bandwidth_extraction(
    visualization_data: List[Dict],
    num_plots: int = 6,
):
    """
    Create visualization plots for bandwidth extraction results.
    
    Args:
        visualization_data: List of dicts with PSD, frequency, and cutoff info.
        num_plots: Maximum number of plots to show.
    """
    import matplotlib.pyplot as plt
    
    num_plots = min(num_plots, len(visualization_data))
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (ax, data) in enumerate(zip(axes, visualization_data[:num_plots])):
        psd_db = data['psd_db'].numpy()
        freq_mhz = data['freq_hz'].numpy() / 1e6
        
        # Plot PSD
        ax.plot(freq_mhz, psd_db, 'b-', linewidth=1, label='PSD')
        
        # Mark cutoffs
        low_cutoff_mhz = data['low_cutoff'] / 1e6
        high_cutoff_mhz = data['high_cutoff'] / 1e6
        
        ax.axvline(low_cutoff_mhz, color='g', linestyle='--', label=f'Low: {low_cutoff_mhz:.2f} MHz')
        ax.axvline(high_cutoff_mhz, color='r', linestyle='--', label=f'High: {high_cutoff_mhz:.2f} MHz')
        
        # Title with results
        match_str = "✓" if data['true_class'] == data['pred_class'] else "✗"
        ax.set_title(f"Sample {data['idx']}: True={data['true_class']}, Pred={data['pred_class']} {match_str}")
        
        ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel('PSD (dB)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('Bandwidth Extraction from STFT - PSD Analysis', y=1.02)
    
    # Save figure
    output_path = Path(__file__).parent / "bandwidth_extraction_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    plt.show()


def test_parameter_sweep(
    num_samples: int = 50,
    verbose: bool = False,
):
    """
    Test different parameter configurations to find optimal settings.
    
    This helps tune the bandwidth extraction algorithm.
    """
    print("\n" + "=" * 70)
    print("PARAMETER SWEEP - Finding Optimal Configuration")
    print("=" * 70)
    
    # Parameters to test (125 MSPS is the actual hardware sample rate)
    sample_rates = [100e6, 125e6, 150e6]
    n_ffts = [1024, 2048, 4096]
    thresholds = [3.0, 6.0, 10.0]
    
    best_accuracy = 0
    best_config = {}
    
    results_table = []
    
    for sr in sample_rates:
        for nfft in n_ffts:
            for thresh in thresholds:
                try:
                    result = run_bandwidth_test(
                        num_samples=num_samples,
                        sample_rate_hz=sr,
                        n_fft=nfft,
                        threshold_db=thresh,
                        verbose=False,
                    )
                    
                    acc = result['accuracy']
                    results_table.append({
                        'sample_rate_mhz': sr / 1e6,
                        'n_fft': nfft,
                        'threshold_db': thresh,
                        'accuracy': acc,
                    })
                    
                    if acc > best_accuracy:
                        best_accuracy = acc
                        best_config = {
                            'sample_rate_hz': sr,
                            'n_fft': nfft,
                            'threshold_db': thresh,
                        }
                        
                except Exception as e:
                    print(f"Error with config ({sr/1e6}MHz, {nfft}, {thresh}dB): {e}")
    
    print("\n" + "-" * 70)
    print("Parameter Sweep Results:")
    print("-" * 70)
    print(f"{'Sample Rate (MHz)':<20} {'N_FFT':<10} {'Threshold (dB)':<15} {'Accuracy':<10}")
    print("-" * 55)
    
    for r in sorted(results_table, key=lambda x: -x['accuracy']):
        print(f"{r['sample_rate_mhz']:<20.0f} {r['n_fft']:<10} {r['threshold_db']:<15.1f} {r['accuracy']*100:<10.1f}%")
    
    print("\n" + "=" * 70)
    print(f"BEST CONFIGURATION: Accuracy = {best_accuracy*100:.1f}%")
    print(f"  Sample Rate: {best_config.get('sample_rate_hz', 0)/1e6:.0f} MHz")
    print(f"  N_FFT: {best_config.get('n_fft', 0)}")
    print(f"  Threshold: {best_config.get('threshold_db', 0)} dB")
    print("=" * 70)
    
    return best_config, results_table


def main():
    parser = argparse.ArgumentParser(
        description="Test bandwidth extraction from STFT data"
    )
    parser.add_argument(
        '--num-samples', '-n',
        type=int,
        default=30,
        help='Number of samples to test (default: 30)'
    )
    parser.add_argument(
        '--sample-rate',
        type=float,
        default=125e6,
        help='Sample rate in Hz (default: 125e6 = 125 MSPS)'
    )
    parser.add_argument(
        '--n-fft',
        type=int,
        default=2048,
        help='FFT size (default: 2048)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=3.0,
        help='Threshold in dB below peak (default: 3.0)'
    )
    parser.add_argument(
        '--visualize', '-v',
        action='store_true',
        help='Generate visualization plots'
    )
    parser.add_argument(
        '--sweep',
        action='store_true',
        help='Run parameter sweep to find optimal configuration'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress per-sample output'
    )
    
    args = parser.parse_args()
    
    if args.sweep:
        test_parameter_sweep(num_samples=args.num_samples, verbose=not args.quiet)
    else:
        run_bandwidth_test(
            num_samples=args.num_samples,
            sample_rate_hz=args.sample_rate,
            n_fft=args.n_fft,
            threshold_db=args.threshold,
            verbose=not args.quiet,
            visualize=args.visualize,
        )


if __name__ == "__main__":
    main()
