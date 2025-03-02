import csv
from ctypes import c_float
from enum import Enum
import glob
import os
from dataclasses import dataclass
import argparse
import sys
from matplotlib.ticker import FuncFormatter, MultipleLocator, StrMethodFormatter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import librosa
from pydub import AudioSegment
import tinytag
from atbu.common.exception import (
    InvalidStateError,
    exc_to_string,
)

from ricochet_tech.audio.float32_helpers import (
    FloatLogLevel,
    float_to_24bit_no_round,
    fltu,
    float_to_24bit,
    get_float_ranges_csv_output,
    get_float_ranges_output,
    get_fltu_log_str,
    get_float_highest_24bit_quant,
    get_i24bit_equal_over_float,
    i24bit_to_float,
)


def wait_debugger():
    """Call this from wherever you would like to begin waiting for remote debugger attach.
    """
    debug_server_port = 7777
    try:
        import debugpy  # pylint: disable=import-outside-toplevel

        debugpy.listen(debug_server_port)
        print(f"Waiting for the debugger to attach via port {debug_server_port}...")
        debugpy.wait_for_client()
        debugpy.breakpoint()
        print(f"Debugger connected.")
    except ModuleNotFoundError as ex:
        raise InvalidStateError(
            f"Cannot 'import debugpy'. Either ensure vscode debugpy is available."
        ) from ex
    except Exception as ex:
        raise InvalidStateError(
            f"Unexpected error. Cannot wait for debugger attach. {exc_to_string(ex)}"
        ) from ex


_LOAD_NON_FLOAT_AS_INT = True


def load_24bit_pcm_with_pydub(file_path, mono: bool):
    audio = AudioSegment.from_wav(file_path)
    if mono and audio.channels > 1:
        audio.set_channels(1)
    audio = audio.set_frame_rate(audio.frame_rate).set_channels(
        audio.channels
    )  # Ensure correct sample rate and channels
    samples = np.array(audio.get_array_of_samples())
    return samples, audio.frame_rate


@dataclass
class AudioInfo:
    data: np.array
    sr: int
    duration: float
    bitdepth: int
    channels: int
    fn: str


def load_audio_files(filenames: str | list[str]) -> list[AudioInfo]:
    if isinstance(filenames, str):
        filenames = list[filenames]
    audio_info: list[AudioInfo] = []
    for spec in filenames:
        found_files = glob.glob(spec)
        for fn in found_files:
            if not os.path.isfile(fn):
                print(f"WARNING: Skipping non-file: {fn}")
                continue
            mdata = tinytag.TinyTag.get(fn)
            if _LOAD_NON_FLOAT_AS_INT and mdata.bitdepth == 24:
                # data, sr = sf.read(fn, dtype='int32')
                # data = data.reshape(-1, 3)  # Reshape into 24-bit samples
                data, sr = load_24bit_pcm_with_pydub(fn, mono=True)
            else:
                data, sr = librosa.load(fn, mono=True, sr=None)
            audio_info.append(
                AudioInfo(
                    data=data,
                    sr=sr,
                    duration=mdata.duration,
                    bitdepth=mdata.bitdepth,
                    channels=mdata.channels,
                    fn=fn,
                )
            )
    return audio_info


@dataclass
class SteadyLevelSegment:
    start_second: float
    end_second: float
    start_sample: int
    end_sample: int
    peak_amplitude: float
    mean_amplitude: float
    peak_dbfs: float
    rms_dbfs: float
    avg_log_mel_db: float


def find_steady_levels(
    samples, sample_rate, min_duration=2.0, threshold_db=0.2
) -> list[SteadyLevelSegment]:
    """
    Identify areas of a floating point audio samples where the audio level is
    steady for a specified duration.

    Args:
        samples: Audio samples loaded from a wav file (all samples normalized to -1.0 to 1.0).
        sample_rate: The sample rate of the `sample` (i.e., 44100, 48000, etc.).
        min_duration (float): Minimum duration (in seconds) of a steady level.
        threshold_db: What level deviation ends, and perhaps starts a steady level.

    Returns:
        list[SteadyLevelInfo]: A list of SteadyLevelInfo instances, one for each
        identified steady level.
    """

    stft = librosa.stft(samples)
    magnitude_spectrogram = np.abs(stft)

    log_mel_spectrogram = librosa.amplitude_to_db(magnitude_spectrogram, ref=np.max)

    steady_segments: list[SteadyLevelSegment] = []
    current_segment_start = None
    current_level_sum = 0
    current_segment_samples = 0

    frame_length = librosa.time_to_frames([min_duration], sr=sample_rate)[0]

    def capture_steady_level():
        start_time = librosa.frames_to_time(current_segment_start, sr=sample_rate)
        end_time = librosa.frames_to_time(
            current_segment_start + current_segment_samples, sr=sample_rate
        )
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        seg_samples = samples[start_sample:end_sample]
        seg_samples_abs = np.abs(seg_samples)

        steady_segments.append(
            SteadyLevelSegment(
                start_second=start_time,
                end_second=end_time,
                start_sample=start_sample,
                end_sample=end_sample,
                peak_amplitude=np.max(np.abs(seg_samples)),
                mean_amplitude=np.mean(np.abs(seg_samples)),
                peak_dbfs=20 * np.log10(np.max(seg_samples_abs)),
                rms_dbfs=20 * np.log10(np.sqrt(np.mean(seg_samples_abs**2))),
                avg_log_mel_db=current_level_sum / current_segment_samples,
            )
        )

    for i in range(log_mel_spectrogram.shape[1]):
        current_frame_level = np.mean(log_mel_spectrogram[:, i])

        if current_segment_start is None:
            current_segment_start = i
            current_level_sum = current_frame_level
            current_segment_samples = 1
            continue

        level_difference = abs(
            current_frame_level - (current_level_sum / current_segment_samples)
        )
        if level_difference < threshold_db:
            current_level_sum += current_frame_level
            current_segment_samples += 1
            continue

        if current_segment_samples >= frame_length:
            capture_steady_level()

        current_segment_start = i
        current_level_sum = current_frame_level
        current_segment_samples = 1

    if current_segment_start is not None and current_segment_samples >= frame_length:
        capture_steady_level()

    if not steady_segments:
        samples_abs = np.abs(samples)
        steady_segments.append(
            SteadyLevelSegment(
                start_second=0.0,
                end_second=librosa.get_duration(y=samples, sr=sample_rate),
                start_sample=0,
                end_sample=len(samples),
                peak_amplitude=np.max(samples_abs),
                mean_amplitude=np.mean(samples_abs),
                peak_dbfs=20 * np.log10(np.max(samples_abs)),
                rms_dbfs=20 * np.log10(np.sqrt(np.mean(samples_abs**2))),
                avg_log_mel_db=np.mean(log_mel_spectrogram),
            )
        )

    return steady_segments


def get_audio_segments(
    samples: np.array,
    sample_rate: int,
    trunc_segments: int = None,
    threshold_db: float = None,
    min_segment_seconds: float = None,
) -> list[SteadyLevelSegment]:
    segments = find_steady_levels(
        samples=samples,
        sample_rate=sample_rate,
        threshold_db=threshold_db,
        min_duration=min_segment_seconds,
    )
    if trunc_segments != -1:
        segments = segments[:trunc_segments]
        if len(segments) > 0:
            samples = samples[: int(segments[-1].end_sample)]
    return segments, samples


def get_target_samples(
    audio_info: AudioInfo,
    start_at_seconds: float = None,
    stop_at_seconds: float = None,
) -> np.array:
    if audio_info.channels == 1:
        audio = audio_info.data
    else:
        audio = audio_info.data[0]
    sr = audio_info.sr

    start_trim_count = 0
    end_trim_count = 0

    if stop_at_seconds is not None:
        stop_at_sample = int(stop_at_seconds * sr)
        end_trim_count = max(len(audio) - stop_at_sample, 0)
        audio = audio[:stop_at_sample]

    if start_at_seconds is not None:
        start_at_sample = int(start_at_seconds * sr)
        start_trim_count = min(len(audio), start_at_sample)
        audio = audio[start_at_sample:]

    if len(audio) == 0:
        raise ValueError("No audio to process.")

    return audio, start_trim_count, end_trim_count


def create_audio_figure_subplots(
    audio_info: AudioInfo | list[AudioInfo],
    start_at_seconds: float = None,
    stop_at_seconds: float = None,
    trunc_segments: int = None,
    threshold_db: float = None,
    min_segment_seconds: float = None,
) -> tuple[Figure, list[Axes]]:

    axs: list[Axes]
    fig, axs = plt.subplots(nrows=len(audio_info), sharex=True, figsize=(14, 8))
    if not isinstance(axs, (list, np.ndarray)):
        axs = [axs]

    for i, ai in enumerate(audio_info):
        audio, start_trim_count, end_trim_count = get_target_samples(
            audio_info=ai,
            start_at_seconds=start_at_seconds,
            stop_at_seconds=stop_at_seconds,
        )

        if len(audio) == 0:
            raise ValueError("No audio to process.")

        segments: list[SteadyLevelSegment] = None
        if trunc_segments is not None:
            segments, audio = get_audio_segments(
                samples=audio,
                sample_rate=ai.sr,
                trunc_segments=trunc_segments,
                threshold_db=threshold_db,
                min_segment_seconds=min_segment_seconds,
            )

        # print("Audio file:", ai.fn)
        # print("Sample rate:", ai.sr)
        # print("Sample data (first 10):", audio[:10])
        # print(f"Samples: {len(audio)}")
        # print(f"Seconds:{len(audio) / ai.sr}")

        ax: Axes = axs[i]
        target_seconds = np.arange(len(audio)) / ai.sr
        ax.plot(target_seconds, audio)

        ax.set_ylim(ymin=-1.0, ymax=1.0)

        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        start_second = start_trim_count / ai.sr

        def x_label_fmt_func(secs, pos):
            targ_time = f"{int(secs // 60)}:{int(secs % 60):02d}"
            if start_second != 0:
                real_time = f"{int((start_second + secs) // 60)}:{int((start_second + secs) % 60):02d}"
                return f"{real_time}\n({targ_time})"
            return f"{targ_time}"

        ax.xaxis.set_major_formatter(FuncFormatter(func=x_label_fmt_func))
        for mtick_text in ax.xaxis.get_majorticklabels():
            mtick_text.set_fontsize(mtick_text.get_fontsize() - 1)
        for mtick_text in ax.xaxis.get_minorticklabels():
            mtick_text.set_fontsize(mtick_text.get_fontsize() - 1)
        ax.set_xlabel("Time (minutes:seconds)")
        ax.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)

        ax.set_title(
            f"{i}: {os.path.basename(ai.fn)}   (bits/sample={ai.bitdepth} rate={ai.sr})"
        )
        ax.set_ylabel("Amplitude")
        ax.grid(which="both", linestyle="--", linewidth=0.5)
        ax.minorticks_on()

        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:+.3f}"))
        ax.yaxis.set_minor_formatter(StrMethodFormatter("{x:+.3f}"))
        for mtick_text in ax.yaxis.get_minorticklabels():
            mtick_text.set_fontsize(mtick_text.get_fontsize() - 2)

        ax2 = ax.twinx()
        ax2.set_yticks(ax.get_yticks())
        ax2.set_yticklabels(ax.get_yticklabels())
        ax2.set_yticks(ax.get_yticks(minor=True), minor=True)
        ax2.set_yticklabels(ax.get_yticklabels(minor=True), minor=True)
        ax2.set_ylim(ax.get_ylim())

        for mtick_text in ax2.get_yminorticklabels():
            mtick_text.set_fontsize(mtick_text.get_fontsize() - 2)

        if segments is not None and i == 0:
            level_annot_inf = []
            for level, segment in enumerate(segments):
                y_value = segment.peak_amplitude
                segment_dbfs = 20 * np.log10(y_value)
                level_annot_inf.append(
                    (
                        f"Level {level} ({y_value:.3f})",
                        segment.start_second,
                        y_value,
                        65,
                        segment.start_second - 1,
                        y_value + 0.1,
                    )
                )
                level_annot_inf.append(
                    (
                        f"{segment_dbfs:.1f} dBFS",
                        segment.start_second,
                        -y_value,
                        45,
                        segment.start_second - 12,
                        -(y_value + 0.22),
                    )
                )

            for next_idx, tup in enumerate(level_annot_inf):
                lbl, lx, ly, rot, tx, ty = tup
                ax.annotate(
                    text=lbl,
                    fontsize="small",
                    rotation=rot,
                    xy=(lx, ly),
                    xytext=(tx, ty),
                    arrowprops=dict(
                        facecolor="black",
                        headwidth=5,
                        headlength=5,
                        width=1,
                        shrink=0.05,
                    ),
                )
    return fig, axs


def plot_audio_files(args):
    audio_info = load_audio_files(filenames=args.filename)
    if args.boost_factor != 1.0:
        for ai in audio_info:
            ai.data *= args.boost_factor
    fig, axs = create_audio_figure_subplots(
        audio_info=audio_info,
        start_at_seconds=args.start_at,
        stop_at_seconds=args.stop_at,
    )
    plt.tight_layout()
    plt.show(block=True)


def plot_audio_file_levels(args):
    audio_info = load_audio_files(filenames=args.filename)
    if args.boost_factor != 1.0:
        for ai in audio_info:
            ai.data *= args.boost_factor
    fig, axs = create_audio_figure_subplots(
        audio_info=audio_info,
        start_at_seconds=args.start_at,
        stop_at_seconds=args.stop_at,
        trunc_segments=args.max_segments,
        threshold_db=args.threshold_db,
        min_segment_seconds=args.min_seconds,
    )
    plt.tight_layout()
    plt.show(block=True)


def handle_levels(args):
    csv_writer = None
    if args.csv:
        csvfile = sys.stdout
        if args.o is not None:
            csvfile = open(args.o, "wt", newline="")
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(
            [
                "#",
                "StartSecond",
                "StartSecondSrc",
                "EndSecond",
                "EndSecondSrc",
                "Peak_dBFS",
                "RMS_dBFS",
                "Peak_Amplitude",
                "Mean_Amplitude",
                "SampleRate",
                "AvgLogMel_dB",
                "Filename",
            ]
        )

    audio_info = load_audio_files(filenames=args.filename)
    if args.boost_factor != 1.0:
        for ai in audio_info:
            ai.data *= args.boost_factor
    for i, ai in enumerate(audio_info):
        audio, start_trim_count, _ = get_target_samples(
            audio_info=ai,
            start_at_seconds=args.start_at,
            stop_at_seconds=args.stop_at,
        )

        if len(audio) == 0:
            raise ValueError("No audio to process.")

        segments, audio = get_audio_segments(
            samples=audio,
            sample_rate=ai.sr,
            trunc_segments=args.max_segments,
            threshold_db=args.threshold_db,
            min_segment_seconds=args.min_seconds,
        )

        if len(segments) == 0:
            raise ValueError("No segments to process.")

        total_seconds = len(audio) / ai.sr
        start_second = start_trim_count / ai.sr
        if not args.csv:
            print(f"Target samples duration: {total_seconds}s")
            print(f"Target samples start: {start_second}s")

        seg: SteadyLevelSegment
        for level, seg in enumerate(segments):
            if args.csv:
                csv_writer.writerow(
                    [
                        f"{level}",
                        f"{seg.start_second}",
                        f"{start_second + seg.start_second}",
                        f"{seg.end_second}",
                        f"{start_second + seg.end_second}",
                        f"{seg.peak_dbfs}",
                        f"{seg.rms_dbfs}",
                        f"{seg.peak_amplitude}",
                        f"{seg.mean_amplitude}",
                        f"{ai.sr}",
                        f"{seg.avg_log_mel_db }",
                        f"{os.path.basename(ai.fn)}",
                    ]
                )
            else:
                print(
                    f"Seg#={level} "
                    f"StartSec={seg.start_second:.3f} "
                    f"StartSecSrc={start_second + seg.start_second:.3f} "
                    f"EndSec={seg.end_second:.3f} "
                    f"EndSecSrc={start_second + seg.end_second:.3f} "
                    f"Peak_dBFS={seg.peak_dbfs} "
                    f"RMS_dBFS={seg.rms_dbfs} "
                    f"Peak_Amplitude={seg.peak_amplitude:.6f} "
                    f"Mean_Amplitude={seg.peak_amplitude:.6f} "
                    f"SR={ai.sr} "
                    f"AvgLogMel_dB={seg.avg_log_mel_db :.6f} "
                    f"FN={os.path.basename(ai.fn)}"
                )


def range1_0_800000_brk_10th_float():
    f_range_start = fltu(f=0.0)
    i24bit_range_start = 0x000000
    for i24bit_cur in range(0, 0x800000, 1):
        f = i24bit_to_float(i24bit=i24bit_cur)
        if f >= f_range_start.f + 0.1:
            i24bit_range_end = i24bit_cur
            f_range_end = fltu(f=f)
            print(
                f"0x{i24bit_range_start:06x} to 0x{i24bit_range_end:06x}: "
                f"{get_fltu_log_str(f_range_start)} to {get_fltu_log_str(f_range_end)}"
            )
            i24bit_range_start = i24bit_cur + 1
            f_range_start = fltu(f=i24bit_to_float(i24bit=i24bit_range_start))
    pass


def range2_0_to_1_inc_10th_float():
    f_range_start = fltu(f=0.0)
    i24bit_range_start = 0x000000
    f_cur = 0.1
    while f_cur <= 1.0:
        f_range_end = fltu(f=f_cur)
        i24bit_range_end = float_to_24bit(f=f_range_end.f)
        print(
            f"0x{i24bit_range_start:06x} to 0x{i24bit_range_end:06x}: "
            f"{get_fltu_log_str(f_range_start)} to {get_fltu_log_str(f_range_end)}"
        )
        f_range_start = f_range_end
        f_range_start.i += 1
        i24bit_range_start = float_to_24bit(f=f_range_start.f)
        f_cur += 0.1
    pass


def range3_0_to_1_inc_10th_float_hquant():
    f_range_start = fltu(f=0.0)
    i24bit_range_start = 0x000000
    f_cur = 0.1
    while f_cur <= 1.0:
        f_range_end = fltu(f=get_float_highest_24bit_quant(f_cur))
        i24bit_range_end = float_to_24bit(f=f_range_end.f)
        ifloat_range_size = f_range_end.i - f_range_start.i
        i24bit_range_size = i24bit_range_end - i24bit_range_start
        print(
            f"i24_size={i24bit_range_size:06x} ({i24bit_range_size:7,}) "
            f"flt_size={ifloat_range_size:08x} ({ifloat_range_size:13,}): "
            f"0x{i24bit_range_start:06x} to 0x{i24bit_range_end:06x}: "
            f"{f_range_start.f:.9e} to {f_range_end.f:.9e}   "
            f"{get_fltu_log_str(f_range_start)} to {get_fltu_log_str(f_range_end)}"
        )
        f_range_start = f_range_end
        f_range_start.i += 1
        i24bit_range_start = float_to_24bit(f=f_range_start.f)
        f_cur += 0.1
    pass


def range_i24_0_800000_inc_10th():
    f_range_start = fltu(f=0.0)
    i24bit_range_start = 0x000000
    f_cur = 0.1
    while f_cur <= 1.0:
        i24bit_range_end = float_to_24bit(f_cur)
        i24bit_range_end = get_i24bit_equal_over_float(
            i24bit=i24bit_range_end, f_limit=f_cur
        )
        f_act = fltu(f=i24bit_to_float(i24bit=i24bit_range_end))
        ifloat_range_size = f_act.i - f_range_start.i
        i24bit_range_size = i24bit_range_end - i24bit_range_start
        print(
            f"i24_size={i24bit_range_size:06x} ({i24bit_range_size:7,}) "
            f"flt_size={ifloat_range_size:08x} ({ifloat_range_size:13,}): "
            f"0x{i24bit_range_start:06x} to 0x{i24bit_range_end:06x}: "
            f"{f_range_start.f:.9e} to {f_act.f:.9e}   "
            f"{get_fltu_log_str(f_range_start)} to {get_fltu_log_str(f_act)}"
        )
        i24bit_range_start = i24bit_range_end
        f_range_start = fltu(f=f_cur)
        f_cur += 0.1


def get_range_output(
    line_num: int,
    i24bit_range_size: int,
    ifloat_range_size: int,
    i24bit_range_start: int,
    i24bit_range_end: int,
    f_range_start: fltu,
    f_range_end: fltu,
):
    return(
        f"{line_num:3}: "
        f"i24_size={i24bit_range_size:06x} ({i24bit_range_size:9,}) "
        f"flt_size={ifloat_range_size:08x} ({ifloat_range_size:13,}): "
        f"0x{i24bit_range_start:06x} to 0x{i24bit_range_end:06x}: "
        f"{f_range_start.f:.9e} to {f_range_end.f:.9e}   "
        f"{get_fltu_log_str(f_range_start)} to {get_fltu_log_str(f_range_end)}"
        "\n"
    )


class IncrementType(Enum):
    FLOAT = 0
    EXP = 1


def handle_range(
    incf: fltu, inc_type: IncrementType, use_csv: bool = False, output_fn: str = None
):

    def inc_value(range_num: int, f: fltu, inc_val: fltu, inc_type: IncrementType):
        if inc_type == IncrementType.FLOAT:
            f.f = inc_val.f * range_num
        elif inc_type == IncrementType.EXP:
            f.p.biased_exp += int(inc_val.f)
        else:
            raise ValueError(f"inc_type is unknown: {inc_type}")

    output_file = sys.stdout
    if output_fn is not None:
        output_file = open(output_fn, "wt", newline="")

    csv_writer = None
    if use_csv:
        csv_writer = csv.writer(output_file)
        csv_writer.writerow(
            [
                "#",
                "i24bit_size_hex",
                "i24bit_size_dec",
                "ifloat_size_hex",
                "ifloat_size_dec",
                "i24bit_start",
                "i24bit_end",
                "f32_start",
                "f32_start_sign",
                "f32_start_bexp",
                "f32_start_exp",
                "f32_start_man",
                "f32_start_raw",
                "f32_end",
                "f32_end_sign",
                "f32_end_bexp",
                "f32_end_exp",
                "f32_end_man",
                "f32_end_raw",
            ]
        )
    f_range_start = fltu(f=0.0)
    f_cur = fltu(f=f_range_start)
    i24bit_range_start = 0x000000
    range_num = 1
    inc_value(range_num, f=f_cur, inc_val=incf, inc_type=inc_type)
    while f_cur.f <= c_float(1.0).value:
        i24bit_range_end = float_to_24bit(f_cur.f)
        if inc_type == IncrementType.FLOAT:
            f_range_end = fltu(f=i24bit_to_float(i24bit=i24bit_range_end))
            i24bit_range_end = get_i24bit_equal_over_float(
                i24bit=i24bit_range_end, f_limit=f_cur.f
            )
        elif inc_type == IncrementType.EXP:
            f_range_end = fltu(f=f_cur)
        else:
            raise ValueError()
        ifloat_range_size = f_range_end.i - f_range_start.i
        i24bit_range_size = i24bit_range_end - i24bit_range_start
        if use_csv:
            csv_writer.writerow(
                [
                    f"{range_num}",
                    f"0x{i24bit_range_size:06x}",
                    f"{i24bit_range_size}",
                    f"0x{ifloat_range_size:08x}",
                    f"{ifloat_range_size}",
                    f"0x{i24bit_range_start:06x}",
                    f"0x{i24bit_range_end:06x}",
                    f"{f_range_start.f:.9e}",
                    f"{f_range_start.p.sign}",
                    f"{f_range_start.p.biased_exp}",
                    f"{f_range_start.exp}",
                    f"0x{f_range_start.p.man:06x}",
                    f"0x{f_range_start.i:08x}",
                    f"{f_range_end.f:.9e}",
                    f"{f_range_end.p.sign}",
                    f"{f_range_end.p.biased_exp}",
                    f"{f_range_end.exp}",
                    f"0x{f_range_end.p.man:06x}",
                    f"0x{f_range_end.i:08x}",
                ]
            )
        else:
            range_output = get_range_output(
                range_num,
                i24bit_range_size=i24bit_range_size,
                ifloat_range_size=ifloat_range_size,
                i24bit_range_start=i24bit_range_start,
                i24bit_range_end=i24bit_range_end,
                f_range_start=f_range_start,
                f_range_end=f_range_end,
            )
            output_file.write(range_output)

        i24bit_range_start = i24bit_range_end
        f_range_start = fltu(f=f_cur)
        range_num += 1
        inc_value(range_num, f=f_cur, inc_val=incf, inc_type=inc_type)


def show_ranges_simple(args):
    inc_type = IncrementType[args.inc_type.upper()]
    handle_range(
        incf=fltu(f=args.inc), inc_type=inc_type, use_csv=args.csv, output_fn=args.o
    )


def show_ranges_detailed(args):
    output_file = sys.stdout
    if args.o is not None:
        output_file = open(args.o, "wt", newline="")
    if args.csv:
        output_file.writelines(get_float_ranges_csv_output())
    else:
        output_file.writelines(get_float_ranges_output())


def interactive_prompt(args):
    while True:
        float_value = input("Enter float value or 'exit':")
        if float_value.lower() == 'exit':
            break
        try:
            float_value = float(float_value)
            print(get_fltu_log_str(u=fltu(f=float_value), level=FloatLogLevel.DETAILED2))
        except ValueError:
            print("Invalid value, try again.")


def main(argv=None):

    parser = argparse.ArgumentParser(
        prog="audio_util",
        description="Audio Utility v0.01",
    )

    parser_common_filenames = argparse.ArgumentParser(add_help=False)
    parser_common_filenames.add_argument(
        "filename", nargs="+", help="One or more audio filenames."
    )

    parser_common_samples = argparse.ArgumentParser(add_help=False)
    parser_common_samples.add_argument(
        "--boost-factor",
        help="The amount by which to boost the .wav samples.",
        type=float,
        default=1.0,
    )

    subparsers = parser.add_subparsers(
        title="Subcommands",
        description="Subcommands description",
        help="Subcommands help",
        required=True,
    )

    parser_common_start_stop = argparse.ArgumentParser(add_help=False)
    parser_common_start_stop.add_argument(
        "--stop-at",
        help="Point in time (in seconds) where audio file processing should stop.",
        type=float,
        default=None,
    )
    parser_common_start_stop.add_argument(
        "--start-at",
        help="Point in time (in seconds) where audio file processing should start.",
        type=float,
        default=None,
    )

    parser_common_find_levels = argparse.ArgumentParser(add_help=False)
    parser_common_find_levels.add_argument(
        "-m",
        "--max-segments",
        help="The maximum segments of steady audio to process.",
        default=-1,
        type=int,
    )
    parser_common_find_levels.add_argument(
        "-t",
        "--threshold-db",
        help="The dB threshold defining the end of, and possibly the start of a segment.",
        default=0.05,
        type=float,
    )
    parser_common_find_levels.add_argument(
        "-s",
        "--min-seconds",
        help="The minimum number of seconds within dB threshold of a valid segement.",
        default=2.0,
        type=float,
    )

    parser_common_csv = argparse.ArgumentParser(add_help=False)
    parser_common_csv.add_argument(
        "-c",
        "--csv",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Output a CSV to either the terminal or a specified file.",
    )
    parser_common_csv.add_argument(
        "-o",
        default=None,
        help="Output CSV file name.",
    )

    subparser_plot = subparsers.add_parser(
        "plot",
        help="Peform graph plots on audio files.",
    )

    plot_subcmd = subparser_plot.add_subparsers(
        dest="plot_subcmd",
    )

    plot_samples = plot_subcmd.add_parser(
        name="samples",
        help=f"Plot the audio file samples.",
        parents=[
            parser_common_filenames,
            parser_common_samples,
            parser_common_start_stop,
        ],
    )
    plot_samples.set_defaults(func=plot_audio_files)

    plot_levels = plot_subcmd.add_parser(
        name="levels",
        help=f"Plot the audio file samples with level annotations.",
        parents=[
            parser_common_filenames,
            parser_common_samples,
            parser_common_start_stop,
            parser_common_find_levels,
        ],
    )
    plot_levels.set_defaults(func=plot_audio_file_levels)

    subparser_levels = subparsers.add_parser(
        "levels",
        help=f"Show audio file levels.",
        parents=[
            parser_common_filenames,
            parser_common_csv,
            parser_common_samples,
            parser_common_start_stop,
            parser_common_find_levels,
        ],
    )
    subparser_levels.set_defaults(func=handle_levels)

    subparser_range = subparsers.add_parser(
        "range",
        help="Show ranges.",
    )

    range_subcmd = subparser_range.add_subparsers(
        dest="range_subcmd",
    )
    range_simple = range_subcmd.add_parser(
        name="simple",
        help=f"Display the float ranges from 0.0 to 1.0 using a relatively simple format.",
        parents=[
            parser_common_csv,
        ],
    )
    inc_types = [e.name.lower() for e in IncrementType]
    range_simple.add_argument(
        "-i",
        "--inc",
        help=f"""The float step increment to use when showing ranges from 0.0 to 1.0. This value determines the size
of each range. It defaults to 0.1 when --inc-type is '{inc_types[IncrementType.FLOAT.value]}', which makes each range
approximately 0.1 (1/10th of 1.0) in size. It defaults to 1.0 when --inc-type is '{inc_types[IncrementType.EXP.value]}',
which makes each range exactly the size of one expoonent's range.
""",
        default=None,
        type=float,
    )
    range_simple.add_argument(
        "--inc-type",
        help=f"""The type of increment to use when walking through the ranges. This can be one of {inc_types}.
This switch's meaning is tied to the value of --inc. If --inc-type is '{inc_types[IncrementType.FLOAT.value]}', the next
range is range_num*inc. If --inc-type is '{inc_types[IncrementType.EXP.value]}', the next range is the biased exponent
plus <inc>.
""",
        choices=inc_types,
        default=inc_types[0],
        type=str,
    )
    range_simple.set_defaults(func=show_ranges_simple)

    range_detailed = range_subcmd.add_parser(
        name="detailed",
        help=f"Display each exponent's range with detail.",
        parents=[
            parser_common_csv,
        ],
    )
    range_detailed.set_defaults(func=show_ranges_detailed)

    subparser_interactive = subparsers.add_parser(
        "interactive",
        help="Interactive prompt to show float details on-demand.",
    )
    subparser_interactive.set_defaults(func=interactive_prompt)

    args = parser.parse_args()

    if hasattr(args, 'inc') and args.inc is None:
        args.inc = 0.1 if args.inc_type == inc_types[IncrementType.FLOAT.value] else 1.0

    args.func(args)


if __name__ == "__main__":
    main()
