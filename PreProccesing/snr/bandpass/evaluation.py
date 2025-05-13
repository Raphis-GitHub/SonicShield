"""
Evaluation tools for measuring filter performance.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import sounddevice as sd
import time

from config import SAMPLE_RATE, SNR_WINDOW_SIZE
from filters import AudioFilter, MultiStageFilter
from signal_processing import calculate_snr


class FilterEvaluator:
    """Class for evaluating filter performance."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        """
        Initialize the evaluator.

        Args:
            sample_rate: Sample rate in Hz
        """
        self.sample_rate = sample_rate
        self.results = {}

    def evaluate_filter(self,
                        audio_filter: AudioFilter,
                        signal_data: np.ndarray,
                        noise_data: np.ndarray) -> Dict:
        """
        Evaluate a filter using signal and noise samples.

        Args:
            audio_filter: Filter to evaluate
            signal_data: Clean signal data (e.g., drone sounds)
            noise_data: Noise data (e.g., wind)

        Returns:
            Dict: Evaluation metrics
        """
        # Apply filter to both signal and noise
        filtered_signal = audio_filter.apply(signal_data)
        filtered_noise = audio_filter.apply(noise_data)

        # Calculate SNR before and after filtering
        original_snr = calculate_snr(signal_data, noise_data)
        filtered_snr = calculate_snr(filtered_signal, filtered_noise)

        # Measure processing time
        start_time = time.time()
        audio_filter.apply(np.concatenate([signal_data, noise_data]))
        processing_time = time.time() - start_time

        # Store and return results
        results = {
            'original_snr': original_snr,
            'filtered_snr': filtered_snr,
            'snr_improvement': filtered_snr - original_snr,
            'processing_time': processing_time
        }

        self.results[str(audio_filter.filter_type)] = results
        return results

    def compare_filters(self,
                        filters: List[AudioFilter],
                        signal_data: np.ndarray,
                        noise_data: np.ndarray) -> Dict:
        """
        Compare multiple filters.

        Args:
            filters: List of filters to compare
            signal_data: Clean signal data
            noise_data: Noise data

        Returns:
            Dict: Comparison results
        """
        results = {}

        for i, f in enumerate(filters):
            filter_name = f"{f.filter_type}_{i}"
            results[filter_name] = self.evaluate_filter(f, signal_data, noise_data)

        return results

    def plot_snr_comparison(self) -> None:
        """Plot SNR comparison of evaluated filters."""
        if not self.results:
            print("No evaluation results available. Run evaluate_filter first.")
            return

        filter_names = list(self.results.keys())
        original_snrs = [r['original_snr'] for r in self.results.values()]
        filtered_snrs = [r['filtered_snr'] for r in self.results.values()]

        x = np.arange(len(filter_names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width / 2, original_snrs, width, label='Original SNR')
        ax.bar(x + width / 2, filtered_snrs, width, label='Filtered SNR')

        ax.set_xlabel('Filter')
        ax.set_ylabel('SNR (dB)')
        ax.set_title('SNR Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(filter_names)
        ax.legend()

        plt.tight_layout()
        plt.show()

    def plot_processing_time(self) -> None:
        """Plot processing time of evaluated filters."""
        if not self.results:
            print("No evaluation results available. Run evaluate_filter first.")
            return

        filter_names = list(self.results.keys())
        times = [r['processing_time'] for r in self.results.values()]

        plt.figure(figsize=(10, 6))
        plt.bar(filter_names, times)
        plt.xlabel('Filter')
        plt.ylabel('Processing Time (seconds)')
        plt.title('Filter Processing Time Comparison')
        plt.tight_layout()
        plt.show()


def create_test_signals(duration: float = 5.0,
                        sample_rate: int = SAMPLE_RATE) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create test signals for filter evaluation.

    Args:
        duration: Duration in seconds
        sample_rate: Sample rate in Hz

    Returns:
        Tuple of (drone_signal, noise_signal)
    """
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)

    # Create simulated drone signal (mixture of frequencies in drone range)
    drone_signal = (
            0.5 * np.sin(2 * np.pi * 6500 * t) +  # 6.5 kHz
            0.3 * np.sin(2 * np.pi * 8000 * t) +  # 8 kHz
            0.2 * np.sin(2 * np.pi * 12000 * t)  # 12 kHz
    )

    # Create simulated wind noise (low frequency noise)
    noise_signal = (
            0.8 * np.sin(2 * np.pi * 100 * t) +  # 100 Hz
            0.6 * np.sin(2 * np.pi * 500 * t) +  # 500 Hz
            0.4 * np.sin(2 * np.pi * 1500 * t)  # 1.5 kHz
    )

    # Add some random noise to both
    drone_signal += 0.1 * np.random.normal(0, 1, len(t))
    noise_signal += 0.2 * np.random.normal(0, 1, len(t))

    return drone_signal, noise_signal


def load_test_files(drone_file: str, noise_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load test files for filter evaluation.

    Args:
        drone_file: Path to drone audio file
        noise_file: Path to noise audio file

    Returns:
        Tuple of (drone_signal, noise_signal)
    """
    # This is a placeholder for file loading functionality
    # In a real implementation, you would use soundfile or another library to load audio files
    print(f"Would load {drone_file} and {noise_file}")

    # For now, return synthetic signals
    return create_test_signals()