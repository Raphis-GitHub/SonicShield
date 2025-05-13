"""
Visualization tools for drone audio analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from typing import Tuple, Optional, Dict

from config import (
    SAMPLE_RATE, SPECTROGRAM_NPERSEG, SPECTROGRAM_NOVERLAP,
    SPECTROGRAM_CMAP, SPECTROGRAM_MAX_FREQ
)


def create_spectrogram(audio_data: np.ndarray,
                       sample_rate: int = SAMPLE_RATE,
                       nperseg: int = SPECTROGRAM_NPERSEG,
                       noverlap: int = SPECTROGRAM_NOVERLAP) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a spectrogram from audio data.

    Args:
        audio_data: Input audio data
        sample_rate: Sample rate in Hz
        nperseg: Length of each segment
        noverlap: Number of points to overlap between segments

    Returns:
        Tuple of (frequencies, times, spectrogram)
    """
    return signal.spectrogram(
        audio_data, fs=sample_rate, nperseg=nperseg, noverlap=noverlap
    )


def plot_spectrogram(frequencies: np.ndarray,
                     times: np.ndarray,
                     spectrogram: np.ndarray,
                     title: str = 'Audio Spectrogram',
                     max_freq: Optional[float] = SPECTROGRAM_MAX_FREQ,
                     cmap: str = SPECTROGRAM_CMAP) -> None:
    """
    Plot a spectrogram.

    Args:
        frequencies: Frequency array
        times: Time array
        spectrogram: Spectrogram data
        title: Plot title
        max_freq: Maximum frequency to display (Hz)
        cmap: Colormap name
    """
    plt.figure(figsize=(10, 6))

    # Convert to dB
    spec_db = 10 * np.log10(spectrogram + 1e-10)  # Add small value to avoid log(0)

    # Plot
    plt.pcolormesh(times, frequencies, spec_db, cmap=cmap, shading='gouraud')
    plt.title(title)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [seconds]')
    plt.colorbar(label='Intensity [dB]')

    if max_freq:
        plt.ylim(0, max_freq)

    plt.tight_layout()
    plt.draw()


def plot_spectrograms_comparison(audio_data: np.ndarray,
                                 filtered_data: np.ndarray,
                                 sample_rate: int = SAMPLE_RATE) -> None:
    """
    Plot original and filtered spectrograms side by side.

    Args:
        audio_data: Original audio data
        filtered_data: Filtered audio data
        sample_rate: Sample rate in Hz
    """
    # Create spectrograms
    f1, t1, spec1 = create_spectrogram(audio_data, sample_rate)
    f2, t2, spec2 = create_spectrogram(filtered_data, sample_rate)

    # Convert to dB
    spec1_db = 10 * np.log10(spec1 + 1e-10)
    spec2_db = 10 * np.log10(spec2 + 1e-10)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Original
    im1 = ax1.pcolormesh(t1, f1, spec1_db, cmap=SPECTROGRAM_CMAP, shading='gouraud')
    ax1.set_title('Original Audio')
    ax1.set_ylabel('Frequency [Hz]')
    ax1.set_xlabel('Time [seconds]')
    ax1.set_ylim(0, SPECTROGRAM_MAX_FREQ)
    fig.colorbar(im1, ax=ax1, label='Intensity [dB]')

    # Filtered
    im2 = ax2.pcolormesh(t2, f2, spec2_db, cmap=SPECTROGRAM_CMAP, shading='gouraud')
    ax2.set_title('Filtered Audio')
    ax2.set_ylabel('Frequency [Hz]')
    ax2.set_xlabel('Time [seconds]')
    ax2.set_ylim(0, SPECTROGRAM_MAX_FREQ)
    fig.colorbar(im2, ax=ax2, label='Intensity [dB]')

    plt.tight_layout()
    plt.show()


def plot_waveform_comparison(audio_data: np.ndarray,
                             filtered_data: np.ndarray,
                             sample_rate: int = SAMPLE_RATE) -> None:
    """
    Plot original and filtered waveforms.

    Args:
        audio_data: Original audio data
        filtered_data: Filtered audio data
        sample_rate: Sample rate in Hz
    """
    duration = len(audio_data) / sample_rate
    time = np.linspace(0, duration, len(audio_data))

    plt.figure(figsize=(10, 8))

    # Original
    plt.subplot(2, 1, 1)
    plt.plot(time, audio_data)
    plt.title('Original Audio Waveform')
    plt.ylabel('Amplitude')
    plt.grid(True)

    # Filtered
    plt.subplot(2, 1, 2)
    plt.plot(time, filtered_data)
    plt.title('Filtered Audio Waveform')
    plt.xlabel('Time [seconds]')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_fft_comparison(audio_data: np.ndarray,
                        filtered_data: np.ndarray,
                        sample_rate: int = SAMPLE_RATE,
                        max_freq: Optional[float] = SPECTROGRAM_MAX_FREQ) -> None:
    """
    Plot FFT comparison of original and filtered audio.

    Args:
        audio_data: Original audio data
        filtered_data: Filtered audio data
        sample_rate: Sample rate in Hz
        max_freq: Maximum frequency to display (Hz)
    """
    # Calculate FFT
    fft_orig = np.abs(np.fft.rfft(audio_data))
    fft_filt = np.abs(np.fft.rfft(filtered_data))

    # Frequency axis
    freq = np.fft.rfftfreq(len(audio_data), 1 / sample_rate)

    # Convert to dB
    fft_orig_db = 20 * np.log10(fft_orig + 1e-10)
    fft_filt_db = 20 * np.log10(fft_filt + 1e-10)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(freq, fft_orig_db, label='Original', alpha=0.7)
    plt.plot(freq, fft_filt_db, label='Filtered', alpha=0.7)

    if max_freq:
        plt.xlim(0, max_freq)

    plt.title('Frequency Response Comparison')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude [dB]')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()