"""
Filter implementations for drone audio analysis.
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional

from config import (
    SAMPLE_RATE, FILTER_ORDER, FILTER_DESIGN,
    DEFAULT_DRONE_LOW_FREQ, DEFAULT_DRONE_HIGH_FREQ
)


class AudioFilter:
    """Base class for audio filters."""

    def __init__(self, filter_type: str = 'bandpass',
                 lowcut: float = DEFAULT_DRONE_LOW_FREQ,
                 highcut: float = DEFAULT_DRONE_HIGH_FREQ,
                 order: int = FILTER_ORDER):
        """
        Initialize the filter.

        Args:
            filter_type: Type of filter ('lowpass', 'highpass', 'bandpass', 'bandstop')
            lowcut: Lower cutoff frequency (Hz)
            highcut: Higher cutoff frequency (Hz)
            order: Filter order
        """
        self.filter_type = filter_type
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order
        self.b, self.a = self._design_filter()

    def _design_filter(self) -> Tuple[np.ndarray, np.ndarray]:
        """Design the filter and return coefficients."""
        nyquist = 0.5 * SAMPLE_RATE

        if self.filter_type == 'bandpass':
            low = self.lowcut / nyquist
            high = self.highcut / nyquist
            b, a = signal.butter(self.order, [low, high], btype='band')
        elif self.filter_type == 'lowpass':
            cutoff = self.highcut / nyquist
            b, a = signal.butter(self.order, cutoff, btype='low')
        elif self.filter_type == 'highpass':
            cutoff = self.lowcut / nyquist
            b, a = signal.butter(self.order, cutoff, btype='high')
        elif self.filter_type == 'bandstop':
            low = self.lowcut / nyquist
            high = self.highcut / nyquist
            b, a = signal.butter(self.order, [low, high], btype='stop')
        else:
            raise ValueError(f"Unknown filter type: {self.filter_type}")

        return b, a

    def apply(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply the filter to audio data."""
        return signal.lfilter(self.b, self.a, audio_data)

    def plot_frequency_response(self) -> None:
        """Plot the frequency response of the filter."""
        w, h = signal.freqz(self.b, self.a)
        plt.figure(figsize=(10, 6))
        plt.plot((SAMPLE_RATE * 0.5 / np.pi) * w, 20 * np.log10(abs(h)))
        plt.xscale('log')
        plt.title('Filter Frequency Response')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude [dB]')
        plt.grid(True)
        plt.axvline(self.lowcut, color='red')
        if self.highcut:
            plt.axvline(self.highcut, color='red')
        plt.show()


class MultiStageFilter:
    """Class for applying multiple filters in sequence."""

    def __init__(self, filters: Optional[List[AudioFilter]] = None):
        """
        Initialize with a list of filters.

        Args:
            filters: List of AudioFilter instances
        """
        self.filters = filters or []

    def add_filter(self, audio_filter: AudioFilter) -> None:
        """Add a filter to the pipeline."""
        self.filters.append(audio_filter)

    def apply(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply all filters in sequence."""
        filtered_data = audio_data.copy()
        for f in self.filters:
            filtered_data = f.apply(filtered_data)
        return filtered_data

    def plot_frequency_response(self) -> None:
        """Plot the frequency response of all filters."""
        plt.figure(figsize=(10, 6))

        # Plot individual filter responses
        for i, f in enumerate(self.filters):
            w, h = signal.freqz(f.b, f.a)
            plt.plot((SAMPLE_RATE * 0.5 / np.pi) * w, 20 * np.log10(abs(h)),
                     label=f'Filter {i + 1}: {f.filter_type}')

        plt.xscale('log')
        plt.title('Multi-stage Filter Frequency Response')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude [dB]')
        plt.grid(True)
        plt.legend()
        plt.show()


def create_drone_bandpass_filter(lowcut: float = DEFAULT_DRONE_LOW_FREQ,
                                 highcut: float = DEFAULT_DRONE_HIGH_FREQ,
                                 order: int = FILTER_ORDER) -> AudioFilter:
    """
    Create a bandpass filter optimized for drone detection.

    Args:
        lowcut: Lower cutoff frequency (Hz)
        highcut: Higher cutoff frequency (Hz)
        order: Filter order

    Returns:
        AudioFilter: Configured bandpass filter
    """
    return AudioFilter('bandpass', lowcut, highcut, order)


def create_wind_noise_filter(highcut: float = 3000,
                             order: int = FILTER_ORDER) -> AudioFilter:
    """
    Create a highpass filter to remove wind noise.

    Args:
        highcut: Cutoff frequency (Hz)
        order: Filter order

    Returns:
        AudioFilter: Configured highpass filter
    """
    return AudioFilter('highpass', highcut, None, order)