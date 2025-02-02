from ctypes import (
    Structure,
    Union,
    c_uint32,
    c_float
)
import math

OUT_PRECISION = 9
FLOAT_FORMAT_SCI = f".{OUT_PRECISION}e"
FLOAT_FORMAT_FXD = f".{OUT_PRECISION}f"

F_TO_I24BIT = float(0x800000)
I24BIT_TO_F = 1.0 / float(0x800000)

def round_float_to_int(f: float) -> int:
    return int(math.copysign(math.floor(math.fabs(f) + 0.5), f))


def float_to_24bit(f: float) -> int:
    return round_float_to_int(f * F_TO_I24BIT)


def float_to_24bit_no_round(f: float) -> int:
    return int(f)


def i24bit_to_float(i24bit: int) -> float:
    return i24bit * I24BIT_TO_F


def get_i24bit_equal_over_float(i24bit: int, f_limit: float) -> int:
    while i24bit_to_float(i24bit=i24bit) > f_limit:
        i24bit -= 1
    while i24bit_to_float(i24bit=i24bit) <= f_limit:
        i24bit += 1
    return i24bit


def get_i24bit_equal_under_float(i24bit: int, f_limit: float) -> int:
    while i24bit_to_float(i24bit=i24bit) < f_limit:
        i24bit += 1
    while i24bit_to_float(i24bit=i24bit) > f_limit:
        i24bit -= 1

    return i24bit

class fltu_fields(Structure):
    _fields_ = [
        ("man", c_uint32, 23),
        ("raw_exp", c_uint32, 8),
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

            exp (int): The IEEE-754 Float32 1-bit sign. If this is specified,
                both sign and man, but not f, should also be specified.

            man (int): The IEEE-754 Float32 1-bit sign. If this is specified,
                both sign and exp, but not f, should also be specified.
        """
        self.i = c_uint32(0)
        self.f = c_float(0)
        f = kw.pop("f", None)
        sign = kw.pop("sign", None)
        exp = kw.pop("exp", None)
        man = kw.pop("man", None)
        super().__init__(*args, **kw)
        if f is not None:
            if sign is not None or exp is not None or man is not None:
                raise ValueError("Cannot specify both f and parts (sign, exp, and man).")
            if isinstance(f, fltu):
                f = c_float(f.f)
            if not isinstance(f, c_float):
                f = c_float(f)
            self.f = f
        if sign is not None or exp is not None or man is not None:
            if sign is None or exp is None or man is None:
                raise ValueError("If specifying any part (sign, exp, and man), all are required.")
            self.p.sign = sign
            self.p.exp = exp
            self.p.man = man

    @property
    def exp(self) -> int:
        exp = int(self.p.raw_exp) - 127
        if exp == -127:
            exp = -126
        return exp

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


def get_fltu_log_str_detail(u: fltu):
    return (
        f"{{0:{FLOAT_FORMAT_SCI}}} "
        f"({{0:{FLOAT_FORMAT_FXD}}}) "
        "("
        "sign={1} "
        "rexp={2:3} (exp={3:4}) "
        "man=0x{4:06x} "
        "raw=0x{5:08x}"
        ") ("
        "24bit={6:07} "
        "0x{6:06x}"
        ")"
    ).format(
        u.f,
        u.p.sign,
        u.p.raw_exp,
        u.exp,
        u.p.man,
        u.i,
        float_to_24bit(u.f),
    )


def get_fltu_log_str_with_24bit(u: fltu):
    return (
        f"{{0:{FLOAT_FORMAT_SCI}}} "
        "("
        "sign={1} "
        "rexp={2:3} (exp={3:4}) "
        "man=0x{4:06x} "
        "raw=0x{5:08x}"
        ") ("
        "24bit=0x{6:06x}"
        ")"
    ).format(
        u.f,
        u.p.sign,
        u.p.raw_exp,
        u.exp,
        u.p.man,
        u.i,
        float_to_24bit(u.f),
    )


def get_fltu_log_str(u: fltu):
    return (
        f"{{0:{FLOAT_FORMAT_SCI}}} "
        "("
        "sign={1} "
        "rexp={2:3} (exp={3:4}) "
        "man=0x{4:06x} "
        "raw=0x{5:08x}"
        ")"
    ).format(
        u.f,
        u.p.sign,
        u.p.raw_exp,
        u.exp,
        u.p.man,
        u.i,
    )


if __name__ == "__main__":
    fu = fltu(f=3.0)
    print(get_fltu_log_str(fu))
    pass