import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy import signal
import time
from matplotlib.animation import FuncAnimation  # Not using this yet but might switch later

# Audio settings - might need to tweak these depending on our hardware
SAMPLE_RATE = 44100  # My mic works best at this rate
DURATION = 5  # How many seconds to record at once - 5 seems like a good balance
CHUNK_SIZE = int(SAMPLE_RATE * DURATION)  # Total samples per chunk

# Setting up the visualization window
plt.figure(figsize=(10, 6))  # Looks good on my monitor, adjust if needed
plt.ion()  # Turn on interactive mode so we can update in real-time


# TODO: Fix the bandpass implementation - currently not working right
def apply_bandpass(audio_data, lowcut=5000, highcut=15000, sample_rate=SAMPLE_RATE):
    """
    Trying to implement a bandpass filter to clean up the audio.
    Currently disabled because it was causing weird artifacts.

    For some reason the butter filter was making everything sound muffled,
    need to revisit this later.
    """
   # This should work in theory:
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    filtered_data = signal.lfilter(b, a, audio_data)
    return filtered_data

   # Just returning the raw audio for now
    return audio_data


def update_spectrogram():
    """Grab audio and show the spectrogram - the heart of the program."""
    # Record from mic
    print("Recording...")  # Helps me see when it's actually recording
    audio_data = sd.rec(CHUNK_SIZE, samplerate=SAMPLE_RATE, channels=1, blocking=True)
    audio_data = audio_data.flatten()  # Convert to 1D array

    # I'll re-enable this when I fix the filter
    # audio_data = apply_bandpass(audio_data, lowcut=300, highcut=3000)

    # Clear previous plot - tried doing this other ways but this seems most reliable
    plt.clf()

    # Create the spectrogram
    # Played around with these parameters a lot - these settings seem decent
    frequencies, times, spectrogram = signal.spectrogram(
        audio_data, fs=SAMPLE_RATE, nperseg=1024, noverlap=512
    )

    # Plot it - the 10*log10 part converts to decibels which looks nicer
    plt.pcolormesh(times, frequencies, 10 * np.log10(spectrogram), cmap='hot', shading='gouraud')
    plt.title('My Real-time Audio Spectrogram')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [seconds]')
    plt.colorbar(label='Intensity [dB]')
    plt.ylim(0, 20000)

    # Update the plot
    plt.draw()
    plt.pause(0.1)  # Give matplotlib a moment to redraw


def main():
    print("Starting")
    try:
        while True:
            update_spectrogram()
            # Small pause between updates
            time.sleep(0.1)

          #  Could add a feature to save spectrograms here so
            if keyboard.is_pressed('s'):
               plt.savefig(f'spectrogram_{time.strftime("%Y%m%d_%H%M%S")}.png')
               print("Saved!")
    except KeyboardInterrupt:
        print("\nShutting down")
    except Exception as e:
        print(f"something went wrong: {e}")
    finally:
        plt.close()


if __name__ == "__main__":
    main()
    # For debugging specific functions:
    # update_spectrogram()