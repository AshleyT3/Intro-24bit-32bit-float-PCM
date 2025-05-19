"""This example creates a 24-bit PCM .wav file with a sine wave whose amplitude is
increased in steps."""

import os
import wave
import numpy as np


def generate_sine_wave_stepped_amplitude(
    frequency,
    duration_secs,
    sample_rate,
    start_amplitude,
    amplitude_inc,
    step_duration,
):
    """Generates an array with a stepped amplitude sine wave."""
    if frequency <= 0:
        raise ValueError("frequency")
    if duration_secs <= 0:
        raise ValueError("duration_secs")
    if sample_rate <= 0:
        raise ValueError("sample_rate")
    if start_amplitude < 0:
        raise ValueError("start_amplitude")
    if amplitude_inc < 0:
        raise ValueError("amplitude_inc")
    if step_duration < 0 or step_duration > duration_secs:
        raise ValueError("step_duration")

    num_samples = int(sample_rate * duration_secs)
    num_samples_per_sec = int(sample_rate * step_duration)
    num_sections = int(num_samples // num_samples_per_sec)

    t = np.linspace(0, duration_secs, num_samples, endpoint=False)
    stepped_sine_wave = np.sin(2 * np.pi * frequency * t)

    amplitude = start_amplitude
    for i in range(num_sections + 1):
        start_index = i * num_samples_per_sec
        end_index = min((i + 1) * num_samples_per_sec, num_samples)
        stepped_sine_wave[start_index:end_index] *= amplitude
        amplitude = amplitude_inc * (i+1)

    return stepped_sine_wave


def convert_wave_data_to_24bit_pcm_samples(wave_data):
    # The max value of a 24-bit sample.
    max_24bit_sample_value = 2**23 - 1
    # The result array to hold the 24-bit PCM samples.
    data_24bit_pcm = bytearray(len(wave_data))
    dest_idx = 0
    # For each sample in the wave data...
    for sample in wave_data:
        # The sample is a Python float value.
        # Use that sample value to take a portion of max_24bit_sample_value as the sample to use.
        f = sample * max_24bit_sample_value
        # Round to integer, clamp to max 24-bit (0x7FFFFF).
        int_sample = min(int(f + 0.5) if f >= 0 else int(f - 0.5), 2**23 - 1)
        # Pack as 3-byte little-endian signed integer
        data_24bit_pcm[dest_idx:dest_idx+3] = int_sample.to_bytes(
            length=3, byteorder="little", signed=True
        )
        dest_idx += 3
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
    duration_secs = 33.0
    start_amplitude = 0.0
    amplitude_inc = 0.1
    step_duration = 3.0

    # The .wav filename to use.
    example_num = os.path.basename(__file__)[:2]
    filename = (
        f"{example_num}_sine_24bit_pcm_{sample_rate//1000}kHz-"
        f"{frequency}Hz-{duration_secs}s-"
        f"amp{start_amplitude:.1f}-"
        f"ampinc{amplitude_inc:.1f}-"
        f"step{step_duration:.1f}.wav"
    )
    filename = os.path.join(
        os.path.dirname(__file__),
        filename,
    )

    # Create the wave data.
    stepped_sine_wave = generate_sine_wave_stepped_amplitude(
        frequency=frequency,
        duration_secs=duration_secs,
        sample_rate=sample_rate,
        start_amplitude=start_amplitude,
        amplitude_inc=amplitude_inc,
        step_duration=step_duration,
    )

    # Convert the wave data into 24-bit PCM samples.
    data = convert_wave_data_to_24bit_pcm_samples(stepped_sine_wave)

    # Write the 24-bit PCM samples to a .wav file.
    _write_24bit_pcm(filename, data, sample_rate)
    print(f"Generated: {filename}")


if __name__ == "__main__":
    main()
