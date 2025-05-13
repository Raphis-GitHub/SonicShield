"""
Signal processing utilities for drone audio analysis.
"""

import numpy as np
from typing import Tuple


def normalize_audio(audio_data: np.ndarray, target_level: float = 0.5) -> np.ndarray:
    """
    Apply automatic gain control to normalize audio levels.

    Args:
        audio_data: Input audio data
        target_level: Target peak level (0.0 to 1.0)

    Returns:
        np.ndarray: Normalized audio data
    """
    if np.max(np.abs(audio_data)) > 0:
        gain = target_level / np.max(np.abs(audio_data))
        return audio_data * gain
    return audio_data


def remove_dc_offset(audio_data: np.ndarray) -> np.ndarray:
    """
    Remove DC offset from audio data.

    Args:
        audio_data: Input audio data

    Returns:
        np.ndarray: Audio data with DC offset removed
    """
    return audio_data - np.mean(audio_data)


def preprocess_audio(audio_data: np.ndarray) -> np.ndarray:
    """
    Apply all preprocessing steps to audio data.

    Args:
        audio_data: Raw audio data

    Returns:
        np.ndarray: Preprocessed audio data
    """
    # Remove DC offset first
    audio_data = remove_dc_offset(audio_data)

    # Then normalize
    audio_data = normalize_audio(audio_data)

    return audio_data


def calculate_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    """
    Calculate Signal-to-Noise Ratio in dB.

    Args:
        signal: Signal audio data
        noise: Noise audio data

    Returns:
        float: SNR in dB
    """
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power == 0:
        return float('inf')

    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def split_frequency_bands(audio_data: np.ndarray,
                          sample_rate: int,
                          bands: list) -> Tuple[np.ndarray, ...]:
    """
    Split audio into multiple frequency bands using FFT.

    Args:
        audio_data: Input audio data
        sample_rate: Sample rate in Hz
        bands: List of frequency band tuples [(low1, high1), (low2, high2), ...]

    Returns:
        Tuple of np.ndarray, one for each frequency band
    """
    # Perform FFT
    fft_data = np.fft.rfft(audio_data)
    freq = np.fft.rfftfreq(len(audio_data), 1 / sample_rate)

    # Create output arrays for each band
    band_signals = []

    for low, high in bands:
        # Create a mask for this frequency band
        mask = (freq >= low) & (freq <= high)

        # Apply mask to FFT data
        band_fft = np.zeros_like(fft_data, dtype=complex)
        band_fft[mask] = fft_data[mask]

        # Convert back to time domain
        band_signal = np.fft.irfft(band_fft, len(audio_data))
        band_signals.append(band_signal)

    return tuple(band_signals)