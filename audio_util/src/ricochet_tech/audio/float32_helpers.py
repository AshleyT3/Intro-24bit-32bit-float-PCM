"""Helper classes, unions, functions to examine/manipulate the 32-bit float format.
"""

# pylint: disable=unsupported-binary-operation

from ctypes import Structure, Union, c_uint32, c_float
from enum import Enum
import math

OUT_PRECISION = 9
FLOAT_FORMAT_SCI = f".{OUT_PRECISION}e"
FLOAT_FORMAT_FXD = f".{OUT_PRECISION}f"

I24BIT_MAX = int(0x800000)
F_TO_I24BIT = float(I24BIT_MAX)
I24BIT_TO_F = 1.0 / float(I24BIT_MAX)


def round_float_to_int(f: float) -> int:
    return int(math.copysign(math.floor(math.fabs(f) + 0.5), f))


def float_to_24bit(f: float) -> int:
    r = round_float_to_int(f * F_TO_I24BIT)
    if f >= 0 and r == I24BIT_MAX:
        return I24BIT_MAX - 1
    return r


def float_to_24bit_no_round(f: float) -> int:
    return int(f)


def i24bit_to_float(i24bit: int) -> float:
    i24bit = min(max(i24bit, -I24BIT_MAX), I24BIT_MAX - 1)
    return i24bit * I24BIT_TO_F


def get_i24bit_equal_over_float(i24bit: int, f_limit: float) -> int:
    while i24bit_to_float(i24bit=i24bit) > f_limit:
        i24bit -= 1
    while i24bit_to_float(i24bit=i24bit) <= f_limit and i24bit < I24BIT_MAX - 1:
        i24bit += 1
    return i24bit


def get_i24bit_equal_under_float(i24bit: int, f_limit: float) -> int:
    while i24bit_to_float(i24bit=i24bit) < f_limit and i24bit < I24BIT_MAX - 1:
        i24bit += 1
    while i24bit_to_float(i24bit=i24bit) > f_limit:
        i24bit -= 1

    return i24bit


class fltu_fields(Structure):
    _fields_ = [
        ("man", c_uint32, 23),
        ("biased_exp", c_uint32, 8),
        ("sign", c_uint32, 1),
    ]


class fltu(Union):

    _fields_ = [
        ("p", fltu_fields),
        ("f", c_float),
        ("i", c_uint32),
    ]

    def __init__(self, *args, **kw):
        """Initialize fltu Union.

        Args:
            f (float): Initialize the union with a float. If this is specified,
                none of sign, exp, and man should be specified. This can be
                either a float or fltu.

            sign (int): The IEEE-754 Float32 1-bit sign. If this is specified,
                both exp and man, but not f, should also be specified.

            biased_exp (int): The IEEE-754 Float32 biased exponent. If this is
                specified, both sign and man, but not f, should also be specified.

            man (int): The IEEE-754 Float32 mantissa. If this is specified,
                both sign and exp, but not f, should also be specified.
        """
        self.i = c_uint32(0)
        self.f = c_float(0)
        f = kw.pop("f", None)
        sign = kw.pop("sign", None)
        biased_exp = kw.pop("biased_exp", None)
        man = kw.pop("man", None)
        super().__init__(*args, **kw)
        if f is not None:
            if sign is not None or biased_exp is not None or man is not None:
                raise ValueError(
                    "Cannot specify both f and parts (sign, exp, and man)."
                )
            if isinstance(f, fltu):
                f = c_float(f.f)
            if not isinstance(f, c_float):
                f = c_float(f)
            self.f = f
        if sign is not None or biased_exp is not None or man is not None:
            if sign is None or biased_exp is None or man is None:
                raise ValueError(
                    "If specifying any part (sign, exp, and man), all are required."
                )
            self.p.sign = sign
            self.p.biased_exp = biased_exp
            self.p.man = man

    @property
    def exp(self) -> int:
        exp = int(self.p.biased_exp) - 127
        if exp == -127:
            exp = -126
        return exp


def get_float_inc(f: float | fltu) -> float:
    if isinstance(f, float):
        f = fltu(f=f)
    f2 = fltu(
        sign=f.p.sign,
        biased_exp=f.p.biased_exp,
        man=f.p.man - 1 if f.p.man != 0 else f.p.man + 1,
    )
    return abs(f.f - f2.f)


def get_float_lowest_24bit_quant(f: float | fltu) -> float:
    f = fltu(f=f)
    start_24bit = float_to_24bit(f.f)
    while float_to_24bit(f.f) == start_24bit:
        f.i -= 1
    if float_to_24bit(f.f) != start_24bit:
        f.i += 1
    return f.f


def get_float_highest_24bit_quant(f: float | fltu) -> float:
    f = fltu(f=f)
    start_24bit = float_to_24bit(f.f)
    while float_to_24bit(f.f) == start_24bit:
        f.i += 1
    if float_to_24bit(f.f) != start_24bit:
        f.i -= 1
    return f.f


class FloatLogLevel(Enum):
    NORMAL = 1
    WITH_24BIT = 2
    DETAILED = 3
    DETAILED2 = 4


def get_fltu_log_str(u: fltu, level: FloatLogLevel = FloatLogLevel.NORMAL) -> str:
    s = f"{u.f:{FLOAT_FORMAT_SCI}} "
    if level == FloatLogLevel.DETAILED:
        s += f"({u.f:{FLOAT_FORMAT_FXD}}) "
    s += (
        f"(sign={u.p.sign} "
        f"bexp={u.p.biased_exp} "
        f"exp={u.exp} "
        f"man=0x{u.p.man:06x} "
        f"raw=0x{u.i:08x})"
    )
    if level.value >= FloatLogLevel.WITH_24BIT.value:
        if u.f < I24BIT_MAX and u.f >= -I24BIT_MAX:
            f_24bit = float_to_24bit(u.f)
            s += f" (24bit: 0x{f_24bit:06x}"
            if level == FloatLogLevel.DETAILED:
                s += f" dec={f_24bit}"
            if level == FloatLogLevel.DETAILED2:
                s += f" f_from={i24bit_to_float(f_24bit):{FLOAT_FORMAT_SCI}}"
            s += ")"
        else:
            s += f" (24bit: N/A)"
    return s


def get_float_from_to_log_str(f_from: float, f_to: float) -> str:
    return f"{f_from:{FLOAT_FORMAT_SCI}} to {f_to:{FLOAT_FORMAT_SCI}}"


class FloatRange:
    def __init__(self, biased_exp: int):
        self.low_end = fltu(sign=0, biased_exp=biased_exp, man=0)
        self.low_end_plus_1 = fltu(sign=0, biased_exp=biased_exp, man=1)
        self.high_end_minus_1 = fltu(sign=0, biased_exp=biased_exp, man=-2)
        self.high_end = fltu(sign=0, biased_exp=biased_exp, man=-1)


def get_float_ranges() -> list[FloatRange]:
    fr_list = []
    for exp in range(-127, 1):
        biased_exp = 127 + exp
        fr_list.append(FloatRange(biased_exp=biased_exp))
    return fr_list


def get_fltu_csv_heaader_part(name_pfx: str) -> str:
    return (
        f"{name_pfx},"
        f"{name_pfx}_raw,"
        f"{name_pfx}_sign,"
        f"{name_pfx}_biased_exp,"
        f"{name_pfx}_mantissa,"
        f"{name_pfx}_24bit,"
        f"{name_pfx}_24bit_hex"
    )


def get_fltu_csv_part(u: fltu) -> str:

    return (
        f"{u.f:{FLOAT_FORMAT_SCI}},"
        f"0x{u.i:08x},"
        f"{u.p.sign},"
        f"{u.p.biased_exp},"
        f"{u.p.man},"
        f"{float_to_24bit(f=u.f)},"
        f"0x{float_to_24bit(f=u.f):06x}"
    )


def get_float_ranges_output() -> list[str]:
    # pylint: disable=line-too-long
    padding = 14
    output = []
    for fr in get_float_ranges():
        output.extend(
            [
                f"{get_float_from_to_log_str(fr.low_end.f, fr.high_end.f)}\n",
                f"     {'low_end ':.<{padding}} {get_fltu_log_str(u=fr.low_end, level=FloatLogLevel.DETAILED)}\n",
                f"     {'low_end+1 ':.<{padding}} {get_fltu_log_str(u=fr.low_end_plus_1, level=FloatLogLevel.DETAILED)}\n",
                f"     ...\n",
                f"     {'high_end-1 ':.<{padding}} {get_fltu_log_str(u=fr.high_end_minus_1, level=FloatLogLevel.DETAILED)}\n",
                f"     {'high_end ':.<{padding}} {get_fltu_log_str(u=fr.high_end, level=FloatLogLevel.DETAILED)}\n",
                "\n",
            ]
        )
    return output


def get_float_ranges_csv_output() -> list[str]:
    output = []
    output.append(
        f"{get_fltu_csv_heaader_part("low_end")},"
        f"{get_fltu_csv_heaader_part("low_end_plus_1")},"
        f"{get_fltu_csv_heaader_part("high_end_minus_1")},"
        f"{get_fltu_csv_heaader_part("high_end")}"
        "\n"
    )
    for fr in get_float_ranges():
        output.append(
            f"{get_fltu_csv_part(fr.low_end)},"
            f"{get_fltu_csv_part(fr.low_end_plus_1)},"
            f"{get_fltu_csv_part(fr.high_end_minus_1)},"
            f"{get_fltu_csv_part(fr.high_end)}"
            "\n"
        )
    return output


def output_float_ranges():
    for line in get_float_ranges_output():
        print(line, end="")


def output_float_ranges_as_csv():
    for line in get_float_ranges_csv_output():
        print(line, end="")


if __name__ == "__main__":
    output_float_ranges()
    pass
