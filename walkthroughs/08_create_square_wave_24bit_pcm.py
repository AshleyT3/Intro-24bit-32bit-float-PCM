"""This example creates a 24-bit PCM .wav file with a square wave whose amplitude is 0.5.
Unlike earlier examples, it uses the sign of amplitude levels of a sine wave to determine
the square wave transitions.
"""

import os
import wave
import numpy as np


def generate_square_wave(frequency, duration_secs, sample_rate, amplitude):
    """Generates a square wave array."""
    num_frames = int(sample_rate * duration_secs)
    t = np.linspace(0, duration_secs, num_frames, endpoint=False)
    return np.sign(np.sin(2 * np.pi * frequency * t)) * amplitude


def convert_wave_data_to_24bit_pcm_samples(wave_data):
    # The max value of a 24-bit sample.
    max_24bit_sample_value = 2**23 - 1
    # The result array to hold the 24-bit PCM samples.
    data_24bit_pcm = b""
    # For each sample in the wave data...
    for sample in wave_data:
        # The sample is a Python float value.
        # Use that sample value to take a portion of max_24bit_sample_value as the sample to use.
        f = sample * max_24bit_sample_value
        # Round to integer, clamp to max 24-bit (0x7FFFFF).
        int_sample = min(int(f + 0.5) if f >= 0 else int(f - 0.5), 2**23 - 1)
        # Pack as 3-byte little-endian signed integer
        data_24bit_pcm += int_sample.to_bytes(3, byteorder="little", signed=True)
    return data_24bit_pcm


def _write_24bit_pcm(filename, data, sample_rate):
    # pylint: disable=no-member
    if (len(data) / 3) % 1 != 0:
        raise ValueError(
            f"24-bit sample data must be made up of 3-byte samples: len(data)={len(data)}"
        )
    with wave.open(filename, "w") as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(3)  # 24-bit (3 bytes per sample)
        wf.setframerate(int(sample_rate))
        wf.writeframes(data)


def main():

    # 48 kHz sample rate.
    sample_rate = 48000

    # A 400 Hz frequency.
    # If the amplitude low and high are distant, this will create a tone of that frequency.
    frequency = 400.0

    # How long should the .wav file be.
    duration_secs = 5.0

    amplitude = 0.5

    # The filename to use.
    example_num = os.path.basename(__file__)[:2]
    filename = (
        f"{example_num}_sine_square_24bit_pcm_{sample_rate//1000}kHz-"
        f"{frequency}Hz-{duration_secs}s-amp-{amplitude:.1f}.wav"
    )
    filename = os.path.join(
        os.path.dirname(__file__),
        filename,
    )

    # Create the wave data.
    square_wave = generate_square_wave(
        frequency=frequency,
        duration_secs=duration_secs,
        sample_rate=sample_rate,
        amplitude=amplitude,
    )

    # Convert the wave data into 24-bit PCM samples.
    data = convert_wave_data_to_24bit_pcm_samples(square_wave)

    # Write the 24-bit PCM samples to a .wav file.
    _write_24bit_pcm(filename, data, sample_rate)
    print(f"Generated: {filename}")


if __name__ == "__main__":
    main()
