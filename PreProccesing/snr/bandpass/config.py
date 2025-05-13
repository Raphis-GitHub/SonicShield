"""
Configuration settings for the drone audio analyzer.
"""

# Audio settings
SAMPLE_RATE = 44100  # Hz
DURATION = 5  # seconds
CHUNK_SIZE = int(SAMPLE_RATE * DURATION)

# Default filter settings
DEFAULT_FILTER_TYPE = 'bandpass'
# Based on the research paper, drone frequencies are primarily in 6kHz-20kHz range
DEFAULT_DRONE_LOW_FREQ = 6000  # Hz
DEFAULT_DRONE_HIGH_FREQ = 20000  # Hz
# Alternative drone frequency range (as mentioned in your notes)
ALT_DRONE_LOW_FREQ = 200  # Hz
ALT_DRONE_HIGH_FREQ = 8000  # Hz
# Wind noise is primarily in 0Hz-3kHz range according to the paper
WIND_NOISE_LOW_FREQ = 0  # Hz
WIND_NOISE_HIGH_FREQ = 3000  # Hz

# Filter design parameters
FILTER_ORDER = 4  # Default filter order
FILTER_DESIGN = 'butter'  # Default filter design (butterworth)

# Visualization settings
SPECTROGRAM_NPERSEG = 1024
SPECTROGRAM_NOVERLAP = 512
SPECTROGRAM_CMAP = 'hot'
SPECTROGRAM_MAX_FREQ = 22000  # Hz (maximum frequency to display)


# Performance metrics
SNR_WINDOW_SIZE = 1024  # Window size for SNR calculation