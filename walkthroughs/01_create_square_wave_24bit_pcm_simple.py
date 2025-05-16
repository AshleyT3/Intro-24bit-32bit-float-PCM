"""This example creates a 24-bit PCM .wav file with a square wave whose cycle
goes from 0.5 to -0.5 to 0.5.
"""

import os
import wave
import numpy as np


def generate_square_wave_simple(
    frequency, duration_secs, sample_rate, amp_low, amp_high
):
    """Generates a wave by flucutating between amp_low and amp_high at the frequency."""

    # Number for samples in the resulting wave data.
    num_frames = int(sample_rate * duration_secs)

    # Create an array for the sample data.
    data = np.empty(num_frames)

    # How many samples represent half the frequency (half the cycle).
    half_cycle_frames = sample_rate / frequency / 2

    # How many frames remain in the current "half cycle" countdown.
    cur_count = half_cycle_frames

    # The current amplitude being written to the data array.
    # This changes every half cycle.
    cur_amp = amp_high

    # For each frame number...
    for fn in range(num_frames):

        # Set the sample.
        data[fn] = cur_amp

        # Decrement the countdown.
        cur_count -= 1

        # If the half cycle is complete, change the amplitude and reset the countdown.
        if cur_count == 0:
            cur_count = half_cycle_frames
            cur_amp = amp_high if cur_amp == amp_low else amp_low

    # Return the resulting sample data.
    return data


def convert_wave_data_to_24bit_pcm_samples(wave_data):
    # The max value of a 24-bit sample.
    max_24bit_sample_value = 2.0**23
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

    amp_high = 0.5
    amp_low = -amp_high

    # The .wav filename to use.
    example_num = os.path.basename(__file__)[:2]
    filename = (
        f"{example_num}_square_24bit_pcm_{sample_rate//1000}kHz-{frequency}Hz-{duration_secs}s"
        f"-amplow-{amp_low:.1f}-amphigh-{amp_high:.1f}.wav"
    )
    filename = os.path.join(
        os.path.dirname(__file__),
        filename,
    )

    # Create the wave data.
    square_wave = generate_square_wave_simple(
        frequency=frequency,
        duration_secs=duration_secs,
        sample_rate=sample_rate,
        amp_low=amp_low,
        amp_high=amp_high,
    )

    # Convert the wave data into 24-bit PCM samples.
    data = convert_wave_data_to_24bit_pcm_samples(square_wave)

    # Write the 24-bit PCM samples to a .wav file.
    _write_24bit_pcm(filename, data, sample_rate)
    print(f"Generated: {filename}")


if __name__ == "__main__":
    main()
