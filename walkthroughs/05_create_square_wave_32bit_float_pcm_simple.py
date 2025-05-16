"""This example creates a 32-bit PCM .wav file with a square wave whose cycle
goes from 1.0 to 0.5 to 1.0.
"""

import os
import struct
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


def write_wav_float_pcm(filename, sample_rate, data):
    """Writes raw float32 data to a WAV file with IEEE FLOAT format."""
    num_samples = len(data)
    bytes_per_sample = 4
    data_size = num_samples * bytes_per_sample
    frame_rate = sample_rate
    num_channels = 1
    bits_per_sample = 32

    fmt_chunk_size = 16
    riff_chunk_size = 4 + (8 + fmt_chunk_size) + (8 + data_size)

    riff_header = struct.pack("<4sI4s", b"RIFF", riff_chunk_size, b"WAVE")
    fmt_header = struct.pack(
        "<4sIHHIIHH",
        b"fmt ",
        fmt_chunk_size,
        3,  # Wave format IEEE-754 Float
        num_channels,
        sample_rate,
        frame_rate * num_channels * bytes_per_sample,
        num_channels * bytes_per_sample,
        bits_per_sample,
    )
    data_header = struct.pack("<4sI", b"data", data_size)
    audio_data = b"".join(struct.pack("<f", sample) for sample in data)

    with open(filename, "wb") as f:
        f.write(riff_header)
        f.write(fmt_header)
        f.write(data_header)
        f.write(audio_data)


def main():

    # 48 kHz sample rate.
    sample_rate = 48000

    # A 400 Hz frequency.
    # If the amplitude low and high are distant, this will create a tone of that frequency.
    frequency = 400.0

    # How long should the .wav file be.
    duration_secs = 5.0

    amp_high = 1.0
    amp_low = 0.5

    # The filename to use.
    example_num = os.path.basename(__file__)[:2]
    filename = (
        f"{example_num}_square_32bit_float_pcm_{sample_rate//1000}kHz-"
        f"{frequency}Hz-{duration_secs}s-amplow-{amp_low:.1f}-amphigh-{amp_high:.1f}.wav"
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

    # Write the 24-bit PCM samples to a .wav file.
    write_wav_float_pcm(filename, sample_rate, square_wave)
    print(f"Generated: {filename}")


if __name__ == "__main__":
    main()
