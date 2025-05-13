"""
Main application for drone audio analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import time
import keyboard  # You'll need to install this: pip install keyboard
from typing import Optional, Tuple, Dict, List

# Import from our modules
from config import (
    SAMPLE_RATE, DURATION, CHUNK_SIZE,
    DEFAULT_DRONE_LOW_FREQ, DEFAULT_DRONE_HIGH_FREQ
)
from filters import AudioFilter, MultiStageFilter, create_drone_bandpass_filter
from signal_processing import preprocess_audio, calculate_snr
from visualization import (
    create_spectrogram, plot_spectrogram,
    plot_spectrograms_comparison, plot_fft_comparison
)
from evaluation import FilterEvaluator, create_test_signals


class DroneAudioAnalyzer:
    """Main class for real-time drone audio analysis."""

    def __init__(self):
        """Initialize the analyzer."""
        self.sample_rate = SAMPLE_RATE
        self.duration = DURATION
        self.chunk_size = CHUNK_SIZE

        # Create default filter
        self.filter = create_drone_bandpass_filter()

        # Initialize plots
        plt.ion()  # Turn on interactive mode

    def update_filter(self,
                      filter_type: str = 'bandpass',
                      lowcut: float = DEFAULT_DRONE_LOW_FREQ,
                      highcut: float = DEFAULT_DRONE_HIGH_FREQ,
                      order: int = 4) -> None:
        """
        Update the current filter.

        Args:
            filter_type: Type of filter
            lowcut: Lower cutoff frequency (Hz)
            highcut: Higher cutoff frequency (Hz)
            order: Filter order
        """
        self.filter = AudioFilter(filter_type, lowcut, highcut, order)
        print(f"Filter updated: {filter_type}, {lowcut}-{highcut} Hz, order {order}")

    def record_audio(self) -> np.ndarray:
        """
        Record audio from the microphone.

        Returns:
            np.ndarray: Recorded audio data
        """
        print("Recording...")
        audio_data = sd.rec(self.chunk_size, samplerate=self.sample_rate,
                            channels=1, blocking=True)
        return audio_data.flatten()

    def process_audio(self, audio_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process audio data with preprocessing and filtering.

        Args:
            audio_data: Raw audio data

        Returns:
            Tuple of (preprocessed_data, filtered_data)
        """
        # Preprocess
        preprocessed_data = preprocess_audio(audio_data)

        # Apply filter
        filtered_data = self.filter.apply(preprocessed_data)

        return preprocessed_data, filtered_data

    def update_spectrogram(self, show_original: bool = True, show_filtered: bool = True) -> None:
        """
        Update the spectrogram display.

        Args:
            show_original: Whether to show the original spectrogram
            show_filtered: Whether to show the filtered spectrogram
        """
        # Record audio
        audio_data = self.record_audio()

        # Process audio
        preprocessed_data, filtered_data = self.process_audio(audio_data)

        # Clear any existing figures
        plt.figure(figsize=(15, 6))
        plt.clf()

        if show_original and show_filtered:
            # Create subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Original spectrogram
            f1, t1, spec1 = create_spectrogram(preprocessed_data, self.sample_rate)
            im1 = ax1.pcolormesh(t1, f1, 10 * np.log10(spec1 + 1e-10), cmap='hot', shading='gouraud')
            ax1.set_title('Original Audio')
            ax1.set_ylabel('Frequency [Hz]')
            ax1.set_xlabel('Time [seconds]')
            ax1.set_ylim(0, 20000)
            plt.colorbar(im1, ax=ax1, label='Intensity [dB]')

            # Filtered spectrogram
            f2, t2, spec2 = create_spectrogram(filtered_data, self.sample_rate)
            im2 = ax2.pcolormesh(t2, f2, 10 * np.log10(spec2 + 1e-10), cmap='hot', shading='gouraud')
            ax2.set_title('Filtered Audio')
            ax2.set_ylabel('Frequency [Hz]')
            ax2.set_xlabel('Time [seconds]')
            ax2.set_ylim(0, 20000)
            plt.colorbar(im2, ax=ax2, label='Intensity [dB]')

        elif show_original:
            # Only original spectrogram
            f, t, spec = create_spectrogram(preprocessed_data, self.sample_rate)
            im = plt.pcolormesh(t, f, 10 * np.log10(spec + 1e-10), cmap='hot', shading='gouraud')
            plt.title('Original Audio Spectrogram')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [seconds]')
            plt.ylim(0, 20000)
            plt.colorbar(im, label='Intensity [dB]')

        elif show_filtered:
            # Only filtered spectrogram
            f, t, spec = create_spectrogram(filtered_data, self.sample_rate)
            im = plt.pcolormesh(t, f, 10 * np.log10(spec + 1e-10), cmap='hot', shading='gouraud')
            plt.title('Filtered Audio Spectrogram')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [seconds]')
            plt.ylim(0, 20000)
            plt.colorbar(im, label='Intensity [dB]')

        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)  # Give matplotlib time to draw

    def run(self) -> None:
        """Run the analyzer in a loop."""
        print("Starting Drone Audio Analyzer")
        print("Press 'q' to quit, 'f' to show filter response, 's' to save spectrogram")

        try:
            while True:
                self.update_spectrogram()

                # Check for keyboard input
                if keyboard.is_pressed('q'):
                    print("Quitting...")
                    break

                if keyboard.is_pressed('f'):
                    print("Showing filter response...")
                    self.filter.plot_frequency_response()
                    time.sleep(0.5)  # Prevent multiple triggers

                if keyboard.is_pressed('s'):
                    plt.savefig(f'spectrogram_{time.strftime("%Y%m%d_%H%M%S")}.png')
                    print("Spectrogram saved!")
                    time.sleep(0.5)  # Prevent multiple triggers

                # Small pause between updates
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\nStopped by user")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            plt.close('all')

    def evaluate_filter(self) -> None:
        """Run a filter evaluation."""
        # Create evaluator
        evaluator = FilterEvaluator()

        # Create test signals
        drone_signal, noise_signal = create_test_signals()

        # Evaluate current filter
        results = evaluator.evaluate_filter(self.filter, drone_signal, noise_signal)

        print("\nFilter Evaluation Results:")
        print(f"Original SNR: {results['original_snr']:.2f} dB")
        print(f"Filtered SNR: {results['filtered_snr']:.2f} dB")
        print(f"SNR Improvement: {results['snr_improvement']:.2f} dB")
        print(f"Processing Time: {results['processing_time'] * 1000:.2f} ms")

        # Compare with different filter configurations
        print("\nComparing different filter configurations...")
        filters = [
            AudioFilter('bandpass', 6000, 20000, 4),  # Default from paper
            AudioFilter('bandpass', 200, 8000, 4),  # Alternative range
            AudioFilter('highpass', 3000, None, 4),  # Just wind removal
            AudioFilter('bandpass', 5000, 15000, 8)  # Higher order
        ]

        evaluator.compare_filters(filters, drone_signal, noise_signal)

        # Plot results
        evaluator.plot_snr_comparison()
        evaluator.plot_processing_time()


def main() -> None:
    """Main function."""
    analyzer = DroneAudioAnalyzer()

    # Run in interactive mode or evaluation mode
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'evaluate':
        analyzer.evaluate_filter()
    else:
        analyzer.run()


if __name__ == "__main__":
    main()