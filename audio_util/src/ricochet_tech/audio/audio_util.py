"""audioutil, a utility to access/examine audio 24-bit and 32-bit float audio samples."""

# pylint: disable=unsupported-binary-operation

import sys
import os
import glob
import argparse
from dataclasses import dataclass
from enum import Enum
from ctypes import c_float
import csv


from atbu.common.exception import (
    InvalidStateError,
    exc_to_string,
)
from matplotlib import ticker
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator, StrMethodFormatter
import numpy as np
import librosa
from pydub import AudioSegment
import tinytag


from ricochet_tech.audio.float32_helpers import (
    FloatLogLevel,
    fltu,
    float_to_24bit,
    get_float_inc,
    get_float_ranges_csv_output,
    get_float_ranges_output,
    get_fltu_log_str,
    get_float_highest_24bit_quant,
    get_i24bit_equal_over_float,
    i24bit_to_float,
)


class AudioUtilException(Exception):
    def __init__(self, *args):
        super().__init__(*args)


def wait_debugger():
    """Call this from wherever you would like to begin waiting for remote debugger attach."""
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


def load_24bit_pcm_with_pydub(file_path, mono: bool):
    audio = AudioSegment.from_wav(file_path)
    if mono and audio.channels > 1:
        audio = audio.set_channels(1)
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


def load_audio_files(
    filenames: str | list[str],
    float_only: bool = True,
    boost_factor: float = None,
    shift_32bit_samples: bool = True,
) -> list[AudioInfo]:
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
            if not float_only and mdata.bitdepth == 24:
                data, sr = load_24bit_pcm_with_pydub(fn, mono=True)
                if (
                    shift_32bit_samples
                    and data.size != 0
                    and isinstance(data[0], np.intc)
                    and len(data[0].tobytes()) == 4
                ):
                    data = np.right_shift(data, 8)
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
    if not audio_info:
        raise AudioUtilException(f"No files found: {filenames}")
    if boost_factor is not None and boost_factor != 1.0:
        for i in audio_info:
            i.data *= boost_factor
    return audio_info


@dataclass
class SteadyLevelSegment:
    start_second: float
    end_second: float
    start_sample: int
    end_sample: int
    min_amplitude: float
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

        max_seg_samples_abs = np.max(seg_samples_abs)
        peak_dbfs = np.float32(-np.inf)
        rms_dbfs = np.float32(-np.inf)
        if max_seg_samples_abs != 0:
            peak_dbfs = 20 * np.log10(max_seg_samples_abs)
        if not np.all(seg_samples_abs == 0):
            rms_dbfs = 20 * np.log10(np.sqrt(np.mean(seg_samples_abs**2)))

        steady_segments.append(
            SteadyLevelSegment(
                start_second=start_time,
                end_second=end_time,
                start_sample=start_sample,
                end_sample=end_sample,
                min_amplitude=np.min(seg_samples_abs),
                peak_amplitude=np.max(seg_samples_abs),
                mean_amplitude=np.mean(seg_samples_abs),
                peak_dbfs=peak_dbfs,
                rms_dbfs=rms_dbfs,
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
                min_amplitude=np.min(samples_abs),
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
    if len(audio_info.data.shape) == 1:
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
    auto_adjust_y_axis: bool = True,
    fixed_notation: bool = False,
    y_font_adjust: float = 0.0,
    titles: str = None,
) -> tuple[Figure, list[Axes]]:

    axs: list[Axes]
    fig, axs = plt.subplots(nrows=len(audio_info), sharex=True, figsize=(14, 8))
    if not isinstance(axs, (list, np.ndarray)):
        axs = [axs]

    foreground_color = 'cyan'
    background_color = 'black'
    plot_color = '#008000'
    grid_color = '#404040'

    fig.patch.set_facecolor(background_color)

    def set_axis_color(ax: Axes):
        ax.set_facecolor(background_color)
        ax.spines['bottom'].set_color(foreground_color)
        ax.spines['top'].set_color(foreground_color)
        ax.spines['left'].set_color(foreground_color)
        ax.spines['right'].set_color(foreground_color)
        for tick in ax.get_xticklabels():
            tick.set_color(foreground_color)
        for tick in ax.get_yticklabels():
            tick.set_color(foreground_color)

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

        total_seconds = len(audio) / ai.sr
        target_seconds = np.arange(len(audio)) / ai.sr
        ax.plot(target_seconds, audio, color=plot_color)

        ymin = -1.0
        ymax = 1.0
        if auto_adjust_y_axis:
            max_sample = np.max(np.abs(audio))
            if max_sample > ymax:
                max_sample *= 1.1
                ymin = -max_sample
                ymax = max_sample
        ax.set_ylim(ymin=ymin, ymax=ymax)

        x_maj_multiple = 10
        if total_seconds < 10:
            x_maj_multiple = 0.2
        ax.xaxis.set_major_locator(MultipleLocator(x_maj_multiple))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        start_second = start_trim_count / ai.sr

        def make_x_label_fmt_func(start_second):
            def x_label_fmt_func(secs, pos):
                # pylint: disable=unused-argument
                if secs < 0:
                    return ""
                targ_time = f"{int(secs // 60)}:{int(secs % 60):02d}"
                if start_second != 0:
                    label_minutes = int((start_second + secs) // 60)
                    label_seconds = int((start_second + secs) % 60)
                    real_time = f"{label_minutes}:{label_seconds:02d}"
                    return f"{real_time}\n({targ_time})"
                return f"{targ_time}"
            return x_label_fmt_func

        ax.xaxis.set_major_formatter(
            FuncFormatter(func=make_x_label_fmt_func(start_second=start_second))
        )
        for mtick_text in ax.xaxis.get_majorticklabels():
            mtick_text.set_fontsize(mtick_text.get_fontsize() - 2)
        for mtick_text in ax.xaxis.get_minorticklabels():
            mtick_text.set_fontsize(mtick_text.get_fontsize() - 2)
        ax.set_xlabel("Time (minutes:seconds)", color=foreground_color)
        ax.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)

        if titles is None or len(titles) <= i:
            cur_title = f"{os.path.basename(ai.fn)}   (bits/sample={ai.bitdepth} rate={ai.sr})"
            if len(audio_info) > 1:
                cur_title = f"Graph #{i}: " + cur_title
        else:
            cur_title = titles[i]
        ax.set_title(cur_title, color=foreground_color)
        ax.set_ylabel("Amplitude", color=foreground_color)
        ax.grid(which="both", linestyle="--", linewidth=0.5, color=grid_color)
        ax.minorticks_on()
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(3))

        label_fmt = "{x:+.3e}"
        if fixed_notation:
            label_fmt = "{x:+.3f}"
        ax.yaxis.set_major_formatter(StrMethodFormatter(label_fmt))
        ax.yaxis.set_minor_formatter(StrMethodFormatter(label_fmt))
        for mtick_text in ax.yaxis.get_majorticklabels():
            mtick_text.set_fontsize(mtick_text.get_fontsize() + y_font_adjust)
        for mtick_text in ax.yaxis.get_minorticklabels():
            mtick_text.set_fontsize(mtick_text.get_fontsize() - 1 + y_font_adjust)

        ax2 = ax.twinx()
        ax2.set_yticks(ax.get_yticks())
        ax2.set_yticklabels(ax.get_yticklabels())
        ax2.set_yticks(ax.get_yticks(minor=True), minor=True)
        ax2.set_yticklabels(ax.get_yticklabels(minor=True), minor=True)
        ax2.set_ylim(ax.get_ylim())

        for mtick_text in ax2.get_ymajorticklabels():
            mtick_text.set_fontsize(mtick_text.get_fontsize() + y_font_adjust)
        for mtick_text in ax2.get_yminorticklabels():
            mtick_text.set_fontsize(mtick_text.get_fontsize() - 1 + y_font_adjust)

        set_axis_color(ax)
        set_axis_color(ax2)

        if segments is not None and i == 0:
            level_annot_inf = []
            for level, segment in enumerate(segments):
                y_value = segment.peak_amplitude
                segment_dbfs = np.float32(-np.inf) if y_value == 0 else 20 * np.log10(y_value)
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
                    color=foreground_color,
                    rotation=rot,
                    xy=(lx, ly),
                    xytext=(tx, ty),
                    arrowprops=dict(
                        facecolor=foreground_color,
                        edgecolor=foreground_color,
                        headwidth=5,
                        headlength=5,
                        width=1,
                        shrink=0.05,
                    ),
                )
    return fig, axs


def plot_audio_files(args):
    audio_info = load_audio_files(
        filenames=args.filename, boost_factor=args.boost_factor
    )
    fig, axs = create_audio_figure_subplots(
        audio_info=audio_info,
        start_at_seconds=args.start_at,
        stop_at_seconds=args.stop_at,
        auto_adjust_y_axis=args.auto_adjust,
        fixed_notation=args.fixed,
        y_font_adjust=args.yfont_adjust,
        titles=args.titles,
    )
    plt.tight_layout()
    plt.show(block=True)


def plot_audio_file_levels(args):
    audio_info = load_audio_files(
        filenames=args.filename, boost_factor=args.boost_factor
    )
    fig, axs = create_audio_figure_subplots(
        audio_info=audio_info,
        start_at_seconds=args.start_at,
        stop_at_seconds=args.stop_at,
        trunc_segments=args.max_segments,
        threshold_db=args.threshold_db,
        min_segment_seconds=args.min_seconds,
        auto_adjust_y_axis=args.auto_adjust,
        fixed_notation=args.fixed,
        titles=args.titles,
    )
    plt.tight_layout()
    plt.show(block=True)


def handle_steadylevels(args):
    csv_writer = None
    if args.csv:
        csvfile = sys.stdout
        if args.o is not None:
            csvfile = open(args.o, "wt", newline="", encoding="utf-8")
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
                "Min_Amplitude",
                "Peak_Amplitude",
                "Mean_Amplitude",
                "AvgLogMel_dB",
                "SampleRate",
                "BitDepth",
                "Channels",
                "Filename",
            ]
        )

    audio_info = load_audio_files(
        filenames=args.filename, boost_factor=args.boost_factor
    )
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
                        f"{seg.min_amplitude}",
                        f"{seg.peak_amplitude}",
                        f"{seg.mean_amplitude}",
                        f"{seg.avg_log_mel_db }",
                        f"{ai.sr}",
                        f"{ai.bitdepth}",
                        f"{ai.channels}",
                        f"{os.path.basename(ai.fn)}",
                    ]
                )
            else:
                sep = " " if not args.multi_line else os.linesep + "    "
                print(
                    f"Seg#={level}{sep}"
                    f"StartSec={seg.start_second:.3f}{sep}"
                    f"StartSecSrc={start_second + seg.start_second:.3f}{sep}"
                    f"EndSec={seg.end_second:.3f}{sep}"
                    f"EndSecSrc={start_second + seg.end_second:.3f}{sep}"
                    f"Peak_dBFS={seg.peak_dbfs}{sep}"
                    f"RMS_dBFS={seg.rms_dbfs}{sep}"
                    f"Min_Amplitude={seg.min_amplitude:.9e}{sep}"
                    f"Peak_Amplitude={seg.peak_amplitude:.9e}{sep}"
                    f"Mean_Amplitude={seg.peak_amplitude:.9e}{sep}"
                    f"AvgLogMel_dB={seg.avg_log_mel_db :.6f}{sep}"
                    f"SampleRate={ai.sr}{sep}"
                    f"BitDepth={ai.bitdepth}{sep}"
                    f"Channels={ai.channels}{sep}"
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
    f_range_start_inc = get_float_inc(f_range_start)
    f_range_end_inc = get_float_inc(f_range_end)
    return (
        f"{line_num:3}: "
        f"i24_size={i24bit_range_size:06x} ({i24bit_range_size:9,}) "
        f"flt_size={ifloat_range_size:08x} ({ifloat_range_size:13,}): "
        f"0x{i24bit_range_start:06x} to 0x{i24bit_range_end:06x}: "
        f"{f_range_start.f:.9e} to {f_range_end.f:.9e}   "
        f"{get_fltu_log_str(f_range_start)} to {get_fltu_log_str(f_range_end)}   "
        f"(flt_start_inc={f_range_start_inc:.9e} flt_end_inc={f_range_end_inc:.9e})"
        "\n"
    )


class IncrementType(Enum):
    FLOAT = 0
    EXP = 1


INC_TYPE_NAMES = [e.name.lower() for e in IncrementType]


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
        output_file = open(output_fn, "wt", newline="", encoding="utf-8")

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
                "f32_start_inc",
                "f32_end",
                "f32_end_sign",
                "f32_end_bexp",
                "f32_end_exp",
                "f32_end_man",
                "f32_end_raw",
                "f32_end_inc",
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
        f_range_start_inc = get_float_inc(f_range_start)
        f_range_end_inc = get_float_inc(f_range_end)
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
                    f"{f_range_start_inc:.9e}",
                    f"{f_range_end.f:.9e}",
                    f"{f_range_end.p.sign}",
                    f"{f_range_end.p.biased_exp}",
                    f"{f_range_end.exp}",
                    f"0x{f_range_end.p.man:06x}",
                    f"0x{f_range_end.i:08x}",
                    f"{f_range_end_inc:.9e}",
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
        output_file = open(args.o, "wt", newline="", encoding="utf-8")
    if args.csv:
        output_file.writelines(get_float_ranges_csv_output())
    else:
        output_file.writelines(get_float_ranges_output())


def interactive_prompt(args):
    # pylint: disable=unused-argument,line-too-long
    while True:
        cmd = input("Enter value (? for help):")
        if cmd.lower() in ["exit", "quit"]:
            break
        if cmd == "?":
            print(
                """
Display the IEEE-754 Binary32 details for a value entered in one of the following formats:

    <float_value>: A floating point value (i.e., 1.0, 1.5e+00, -1.5e+00, 1.0e-37).

    0x<hex_value>: A raw hex IEEE-754 Binary32 value (i.e., "0x3fc00000" is 1.5e+00).

    <sign>,<bexp>,<mantissa>: Combine each into a single IEEE-754 Binary value. Each part is as follows:
        <sign> is 0 for positive, 1 for negative.
        <bexp> is the biased exponent (i.e., biased exponent 128 is an exponent of 1).
        <mantissa> is the mantissa.
        For example, entering "1,127,0x400000" shows details for -1.5e+00).

    'exit', 'quit': exit the program.
"""
            )
            continue
        try:
            cmd_list = cmd.split(",")
            if len(cmd_list) == 3:
                cmd_vals = [int(s, 0) for s in cmd_list]
                f = fltu(sign=cmd_vals[0], biased_exp=cmd_vals[1], man=cmd_vals[2])
            elif cmd[:2] == "0x":
                f = fltu()
                f.i = int(cmd, 16)
            else:
                f = fltu(f=float(cmd))
            print(get_fltu_log_str(u=f, level=FloatLogLevel.DETAILED2))
        except ValueError:
            print("Invalid value, try again.")


def handle_dump(args):
    verbosity = args.verbose
    verbosity = min(verbosity, FloatLogLevel.DETAILED2.value)
    audio_info = load_audio_files(
        filenames=args.filename,
        float_only=False,
        boost_factor=args.boost_factor,
        shift_32bit_samples=args.shift_32bit,
    )
    for i, ai in enumerate(audio_info):
        audio, start_trim_count, _ = get_target_samples(
            audio_info=ai,
            start_at_seconds=args.start_at,
            stop_at_seconds=args.stop_at,
        )

        if len(audio) == 0:
            raise ValueError("No audio to process.")

        total_seconds = len(audio) / ai.sr
        start_second = start_trim_count / ai.sr
        print(
            "-----------------------------------------------------------------------------------"
        )
        print(f"Filename={ai.fn}")
        print(f"Target samples duration: {total_seconds}s")
        print(f"Target samples start: {start_second}s")

        num_bytes_sample = int(ai.bitdepth / 8)
        bytes_fmt = f"0x{{:0{int(num_bytes_sample*2)}x}}"
        for i, sample in enumerate(audio):
            cur_time = start_second + (i * (1.0 / ai.sr))
            print(f"{i:7}", end="")
            print(f": t={cur_time:.9f}s ", end="")
            if ai.bitdepth == 32:
                print(f"sample={sample:.9e} ({sample:.9f})", end="")
                sample_f = sample
            else:
                raw_bytes = sample.tobytes()
                if np.little_endian:
                    raw_bytes = raw_bytes[::-1]
                if len(raw_bytes) == 4 and args.shift_32bit:
                    raw_bytes = raw_bytes[1:]
                raw_hex = "".join(f"{byte:02x}" for byte in raw_bytes)
                byte_delta = int(len(raw_bytes) - (ai.bitdepth / 8))
                print(f"sample=0x{raw_hex}", end="")
                sample_f = i24bit_to_float(i24bit=sample)
            if verbosity != 0:
                flt_log_str = get_fltu_log_str(u=fltu(f=sample_f), level=FloatLogLevel(verbosity))
                print(f" {flt_log_str}", end="")
            print()


def create_args_parser() -> argparse.ArgumentParser:
    # pylint: disable=line-too-long

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
        help="The amount by which to boost the .wav samples (default is 1.0, no adjustment).",
        type=float,
        default=1.0,
    )
    parser_common_samples.add_argument(
        "--shift-32bit",
        help="""When 24-bit PCM integer samples are used directly (currently only used by 'dump'), they
are loaded into a high bytes of a 32-bit integer. By default, when dumping those samples,
they are shifted back to the right by 8 bits so a normal 24-bit sample is presented
as part of dump output. If you wish to see the 32-bit sample loaded by the python
package, you can use this option to disable that shift normalizing effect. Generally,
you can ignore this option.""",
        action=argparse.BooleanOptionalAction,
        default=True,
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

    parser_common_plot = argparse.ArgumentParser(add_help=False)
    parser_common_plot.add_argument(
        "--auto-adjust",
        help=(
            "Adjust the y-axis (amplitude) to fix the maximum amplitude sample. "
            "If disabled, force -1.0 to 1.0 (default is enabled)."
        ),
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser_common_plot.add_argument(
        "--fixed",
        help=(
            "Use fixed format notation for the y-axis (amplitude) instead of scientific notation "
            "(default is scientific notation)."
        ),
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser_common_plot.add_argument(
        "--yfont-adjust",
        help="Apply a delta to the y-axis font size (default is 0.0, no adjustment).",
        type=float,
        default=0.0,
    )
    parser_common_plot.add_argument(
        "--titles",
        nargs="+",
        help=(
            "One or more titles for graphs (by default, titles are auto-generated using file name "
            "and metadata). If multiple graphs are produced (multiple files specified), "
            "auto-generated titles will be used if enough titles are not specified for all graphs. "
            "Specifying more titles than needed simply does not use the additional titles."
        ),
        type=str,
        default=None,
    )

    parser_common_find_levels = argparse.ArgumentParser(add_help=False)
    parser_common_find_levels.add_argument(
        "-m",
        "--max-segments",
        help="The maximum segments of steady audio to process (default is unlimited).",
        default=-1,
        type=int,
    )
    parser_common_find_levels.add_argument(
        "-t",
        "--threshold-db",
        help="""The dB threshold that, if exceeded, defines the end of, and possibly the
start of a segment (default=0.05).
""",
        default=0.05,
        type=float,
    )
    parser_common_find_levels.add_argument(
        "-s",
        "--min-seconds",
        help="""The minimum number of seconds required to remain within dB threshold
in order for a segment to be observed (default=2.0 seconds).
""",
        default=2.0,
        type=float,
    )

    parser_common_output = argparse.ArgumentParser(add_help=False)
    parser_common_output.add_argument(
        "--multi-line",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="For non-CSV output, use a multi-line format instead of a single line per item.",
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
            parser_common_plot,
            parser_common_start_stop,
        ],
    )
    plot_samples.set_defaults(func=plot_audio_files)

    plot_levels = plot_subcmd.add_parser(
        name="steadylevels",
        description="""Identifies segments within an audio file where the amplitude remains at a
particular level for some duration. The resulting segments are then graphed
with annotations. For example, a test tone could be played while adjusting a
gain knob to different steps at some selected interval where this command
will graph and annoatate the levels at the different steps. This command should
only be used with files that have periods of largly unchanged amplitude.""",
        help="Plot the steady levels of an audio file annotations.",
        parents=[
            parser_common_filenames,
            parser_common_samples,
            parser_common_plot,
            parser_common_start_stop,
            parser_common_find_levels,
        ],
    )
    plot_levels.set_defaults(func=plot_audio_file_levels)

    subparser_levels = subparsers.add_parser(
        "steadylevels",
        description="""Identifies segments within an audio file where the amplitude remains at a
particular level for some duration. The resulting segments are then listed in
human-readable or csv format. For example, a test tone could be played while
adjusting a gain knob to different steps at some selected interval where this
command will list information about the levels at the different steps. This command
should only be used with files that have periods of largly unchanged amplitude.""",
        help="List the steady levels of an audio file annotations.",
        parents=[
            parser_common_filenames,
            parser_common_output,
            parser_common_csv,
            parser_common_samples,
            parser_common_start_stop,
            parser_common_find_levels,
        ],
    )
    subparser_levels.set_defaults(func=handle_steadylevels)

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
    range_simple.add_argument(
        "-i",
        "--inc",
        help=f"""The float step increment to use when showing ranges from 0.0 to 1.0. This value determines the size
of each range. It defaults to 0.1 when --inc-type is '{INC_TYPE_NAMES[IncrementType.FLOAT.value]}', which makes each range
approximately 0.1 (1/10th of 1.0) in size. It defaults to 1.0 when --inc-type is '{INC_TYPE_NAMES[IncrementType.EXP.value]}',
which makes each range exactly the size of one expoonent's range.
""",
        default=None,
        type=float,
    )
    range_simple.add_argument(
        "--inc-type",
        help=f"""The type of increment to use when walking through the ranges. This can be one of {INC_TYPE_NAMES}.
This switch's meaning is tied to the value of --inc. If --inc-type is '{INC_TYPE_NAMES[IncrementType.FLOAT.value]}', the next
range is range_num*inc. If --inc-type is '{INC_TYPE_NAMES[IncrementType.EXP.value]}', the next range is the biased exponent
plus <inc>. (The default is {INC_TYPE_NAMES[0]}.)
""",
        choices=INC_TYPE_NAMES,
        default=INC_TYPE_NAMES[0],
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

    subparser_dump = subparsers.add_parser(
        "dump",
        help="Dump audio file samples.",
        description="Dumps the audio file names.",
        parents=[
            parser_common_filenames,
            parser_common_output,
            parser_common_samples,
            parser_common_start_stop,
        ],
    )
    subparser_dump.add_argument(
        "-v",
        "--verbose",
        help="Verbose. This can be specified multiple times (i.e., -vv is more verbose than -v).",
        action='count',
        default=0,
    )
    subparser_dump.set_defaults(func=handle_dump)

    return parser


def main(argv=None):
    # pylint: disable=unused-argument

    parser = create_args_parser()
    args = parser.parse_args()

    if hasattr(args, "inc") and args.inc is None:
        args.inc = (
            0.1 if args.inc_type == INC_TYPE_NAMES[IncrementType.FLOAT.value] else 1.0
        )

    try:
        args.func(args)
    except AudioUtilException as e:
        print(e)
        exit_code = 1 if len(e.args) == 1 else int(e.args[1])
        exit(exit_code)


if __name__ == "__main__":
    main()
