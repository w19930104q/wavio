"""
The wavio module defines the functions:

read(file)
    Read a WAV file and return a `wavio.Wav` object, with attributes
    `data`, `rate`, `sampwidth` and `chunks`.  `chunks` is a tuple of
    `wavio.Chunk` objects that exposes the raw RIFF chunks of the file
    (e.g. 'cue ', 'LIST', 'smpl', ...) in the order in which they occur
    in the file.

write(filename, data, rate, scale=None, sampwidth=None, chunks=None)
    Write a numpy array to a WAV file.  If `chunks` is given, the extra
    RIFF chunks are merged into the output; the chunk offsets and sizes
    are recomputed in coordination with the 'data' chunk.


-----
Author: Warren Weckesser
License: BSD 2-Clause:
Copyright (c) 2015-2022, Warren Weckesser
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

from typing import BinaryIO, Iterable, Literal, cast
import io as _io
import os as _os
import struct as _struct
import warnings as _warnings
from dataclasses import dataclass
import wave as _wave
import numpy as _np
from numpy.typing import ArrayLike


__version__ = "0.0.10dev2"


class ClippedDataWarning(UserWarning):

    def __init__(self, message=None):
        if message is None:
            message = ('Some data values were clipped when converted to the '
                       'output format.')
        self.args = (message,)


class ClippedDataError(RuntimeError):

    def __init__(self, message=None):
        if message is None:
            message = ('Some data values were clipped when converted to the '
                       'output format.')
        self.args = (message,)


class WavChunkError(_wave.Error):
    """
    Exception raised when the RIFF chunk structure of a WAV file (or of
    the `chunks` argument of `wavio.write`) is corrupt or invalid, e.g.
    a truncated chunk, or a duplicate of a chunk that may occur at most
    once.  This is a subclass of `wave.Error`.
    """


@dataclass(frozen=True, kw_only=True)
class Chunk:
    """
    A raw RIFF chunk of a WAV file.

    Attributes
    ----------
    id : bytes
        The 4-byte chunk identifier (e.g. ``b'fmt '``, ``b'cue '``,
        ``b'LIST'``).
    data : bytes
        The raw payload of the chunk, not including the 8-byte chunk
        header and not including the padding byte.
    pad : bool
        Whether a padding byte follows the chunk payload in the file.
        By default (``pad=None``), the standard convention is used: the
        padding byte is present if and only if ``len(data)`` is odd.
    """
    id: bytes
    data: bytes
    pad: bool | None = None

    def __post_init__(self):
        id_ = bytes(self.id)
        if len(id_) != 4:
            raise ValueError('The chunk id must be exactly 4 bytes, '
                             f'got {id_!r}.')
        data = bytes(self.data)
        object.__setattr__(self, 'id', id_)
        object.__setattr__(self, 'data', data)
        if self.pad is None:
            object.__setattr__(self, 'pad', bool(len(data) & 1))

    def __repr__(self):
        return f"Chunk(id={self.id!r}, len(data)={len(self.data)})"


# Chunk ids that may occur at most once in a standard WAV file.  'LIST'
# is deliberately not included: a file may contain several LIST chunks
# (e.g. an 'INFO' list and an 'adtl' list).  Duplicates of unknown chunk
# ids are also allowed, since their semantics are not known.
_singleton_chunk_ids = {b'fmt ', b'data', b'cue ', b'fact', b'smpl',
                        b'plst', b'inst', b'bext'}


def _read_chunks(f):
    """
    Walk the RIFF chunk structure of the file object `f`, starting at the
    current position, and return the chunks as a tuple of `Chunk` objects
    in the order in which they occur in the file.  Raises `WavChunkError`
    if the structure is corrupt.
    """
    header = f.read(12)
    if len(header) < 12:
        raise WavChunkError('not a RIFF/WAVE file: the file is too short '
                            f'({len(header)} bytes) to hold a RIFF header.')
    riff_id, _, wave_id = header[:4], header[4:8], header[8:12]
    if riff_id != b'RIFF':
        raise WavChunkError("not a RIFF file: the file starts with "
                            f"{riff_id!r}, not b'RIFF'.")
    if wave_id != b'WAVE':
        raise WavChunkError("not a WAVE file: the RIFF form type is "
                            f"{wave_id!r}, not b'WAVE'.")
    chunks = []
    seen_singletons = set()
    offset = 12
    while True:
        hdr = f.read(8)
        if len(hdr) == 0:
            break
        if len(hdr) < 8:
            raise WavChunkError('corrupt WAV file: truncated chunk header '
                                f'at offset {offset} (expected 8 bytes, '
                                f'got {len(hdr)}).')
        cid = hdr[:4]
        size = _struct.unpack('<I', hdr[4:])[0]
        data = f.read(size)
        if len(data) < size:
            raise WavChunkError(f'corrupt WAV file: chunk {cid!r} at offset '
                                f'{offset} declares {size} data bytes, but '
                                f'the file ends after {len(data)}.')
        offset += 8 + size
        pad = False
        if size & 1:
            # A missing pad byte at the end of the file is tolerated
            # (the `wave` module itself does not write a pad byte after
            # an odd-sized 'data' chunk).
            pad = len(f.read(1)) == 1
            if pad:
                offset += 1
        if cid in _singleton_chunk_ids:
            if cid in seen_singletons:
                raise WavChunkError('corrupt WAV file: duplicate chunk '
                                    f'{cid!r} at offset {offset - 8 - size}.')
            seen_singletons.add(cid)
        chunks.append(Chunk(id=cid, data=data, pad=pad))
    return tuple(chunks)


def _wav2array(nchannels, sampwidth, data):
    """data must be the string containing the bytes from the wav file."""
    num_samples, remainder = divmod(len(data), sampwidth * nchannels)
    if remainder > 0:
        raise ValueError(f'The length of data ({len(data)}) is not a multiple '
                         'of sampwidth * num_channels.')
    if sampwidth > 4:
        raise ValueError("sampwidth must not be greater than 4.")

    if sampwidth == 3:
        a = _np.empty((num_samples, nchannels, 4), dtype=_np.uint8)
        raw_bytes = _np.frombuffer(data, dtype=_np.uint8)
        a[:, :, :sampwidth] = raw_bytes.reshape(-1, nchannels, sampwidth)
        a[:, :, sampwidth:] = (a[:, :, sampwidth - 1:sampwidth] >> 7) * 255
        result = a.view('<i4').reshape(a.shape[:-1])
    else:
        # 8 bit samples are stored as unsigned ints; others as signed ints.
        dt_char = 'u' if sampwidth == 1 else 'i'
        a = _np.frombuffer(data, dtype=f'<{dt_char}{sampwidth}')
        result = a.reshape(-1, nchannels)
    return result


def _array2wav(a, sampwidth):
    """
    Convert the input array `a` to a string of WAV data.

    a.dtype must be one of uint8, int16 or int32.  Allowed sampwidth
    values are:
        dtype    sampwidth
        uint8        1
        int16        2
        int32      3 or 4
    When sampwidth is 3, the *low* bytes of `a` are assumed to contain
    the values to include in the string.
    """
    if sampwidth == 3:
        # `a` must have dtype int32
        if a.ndim == 1:
            # Convert to a 2D array with a single column.
            a = a.reshape(-1, 1)
        # By shifting first 0 bits, then 8, then 16, the resulting output
        # is 24 bit little-endian.
        a8 = (a.reshape(a.shape + (1,)) >> _np.array([0, 8, 16])) & 255
        wavdata = a8.astype(_np.uint8).tobytes()
    else:
        # Make sure the array is little-endian, and then convert using
        # tobytes().
        a = a.astype('<' + a.dtype.str[1:], copy=False)
        wavdata = a.tobytes()
    return wavdata


@dataclass(frozen=True, kw_only=True)
class Wav:
    """
    Object returned by `wavio.read`.  Attributes are:

    data : numpy array
        The array of data read from the WAV file. The shape of the array
        is (num_samples, num_channels).  num_channels is the number of audio
        channels (1 for mono, 2 for stereo).   The data type of the array
        (i.e. data.dtype) is determined by `sampwidth`::

                sampwidth      dtype
                    1          numpy.uint8
                    2          numpy.int16
                    3          numpy.int32
                    4          numpy.int32

    rate : int
        The sampling frequency (i.e. frame rate or sample rate) of the
        WAV file.
    sampwidth : int
        The sample width (i.e. number of bytes per sample) of the WAV file.
        For example, `sampwidth == 3` is a 24 bit WAV file.
    chunks : tuple of `Chunk`
        The raw RIFF chunks of the file, in the order in which they occur
        in the file.  This includes the 'fmt ' and 'data' chunks, so the
        positions of the other chunks (e.g. 'cue ', 'LIST', 'smpl')
        relative to the audio data are preserved.  Passing this tuple to
        `wavio.write` via its `chunks` parameter preserves the extended
        chunks of the file; for a standard PCM file, the output of such
        a read/write round trip is byte-for-byte identical to the input.

    """
    data: _np.ndarray
    rate: int
    sampwidth: int
    chunks: tuple = ()

    def __repr__(self):
        return (f"Wav(data.shape={self.data.shape}, "
                f"data.dtype={self.data.dtype}, "
                f"rate={self.rate}, "
                f"sampwidth={self.sampwidth})")


def read(file: BinaryIO | str) -> Wav:
    """
    Read a WAV file.

    Parameters
    ----------
    file : string or file object
        Either the name of a file or an open file pointer.

    Returns
    -------
    wav : wavio.Wav() instance
        The return value is an instance of the class `wavio.Wav`,
        with the following attributes:

            data : numpy array
                The array of data read from the WAV file.  The shape of the
                array is (num_samples, num_channels).  num_channels is the
                number of audio channels (1 for mono, 2 for stereo).  The
                data type of the array (i.e. data.dtype) is determined by
                `sampwidth`::

                    sampwidth      dtype
                        1          numpy.uint8
                        2          numpy.int16
                        3          numpy.int32
                        4          numpy.int32

            rate : int
                The sampling frequency (i.e. frame rate or sample rate) of the
                WAV file.
            sampwidth : int
                The sample width (i.e. number of bytes per sample) of the
                WAV file.  For example, `sampwidth == 3` is a 24 bit WAV file.
            chunks : tuple of `wavio.Chunk`
                The raw RIFF chunks of the file, in file order.  See the
                docstring of `wavio.Wav` for details.

    Notes
    -----
    This function uses the `wave` module of the Python standard libary
    to read the WAV file, so it has the same limitations as that library.
    In particular, the function does not read compressed WAV files, and
    it does not read files with floating point data.

    In addition to decoding the audio data, the RIFF chunk structure of
    the file is validated while the raw chunks are collected.  A corrupt
    chunk structure (e.g. a truncated chunk) or a duplicate of a chunk
    that may occur at most once (such as 'fmt ', 'data' or 'cue ')
    raises `wavio.WavChunkError` (a subclass of `wave.Error`).

    The array returned by `wavio.read` is always two-dimensional.  If the
    WAV data is mono, the array will have shape (num_samples, 1).

    `wavio.read()` does not scale or normalize the data.  The data in the
    array `wav.data` is the data that was in the file.  When the file
    contains 24 bit samples, the resulting numpy array is 32 bit integers,
    with values that have been sign-extended.
    """
    if isinstance(file, (str, _os.PathLike)):
        with open(file, 'rb') as f:
            return _read_wav(f)
    return _read_wav(file)


def _read_wav(f):
    try:
        pos = f.tell()
    except (AttributeError, OSError):
        # The file object is not seekable; slurp it into a seekable
        # buffer so the chunk walk can be followed by the `wave` read.
        f = _io.BytesIO(f.read())
        pos = 0
    chunks = _read_chunks(f)
    f.seek(pos)
    with _wave.open(f) as wav:
        rate = wav.getframerate()
        nchannels = wav.getnchannels()
        sampwidth = wav.getsampwidth()
        nframes = wav.getnframes()
        data = wav.readframes(nframes)
    array = _wav2array(nchannels, sampwidth, data)
    return Wav(data=array, rate=rate, sampwidth=sampwidth, chunks=chunks)


_sampwidth_dtypes = {1: _np.uint8,
                     2: _np.int16,
                     3: _np.int32,
                     4: _np.int32}
_sampwidth_minmax = {1: (0, 255),
                     2: (-2**15, 2**15 - 1),
                     3: (-2**23, 2**23 - 1),
                     4: (-2**31, 2**31 - 1)}


def _float_to_integer(x, sampwidth, scale=None, clip="warn"):

    # For a given sampwidth and scale, the actual allowed
    # interval for float input is [-(1 + 1/c)*scale, scale],
    # where c = 2**(8*sampwidth - 1) - 0.5.  Values outside
    # that interval will result in clipping.

    nbits = 8*sampwidth
    c = 2**(nbits - 1) - 0.5

    if scale == "auto":
        scale = max(_np.max(_np.r_[x[x > 0], 0]),
                    _np.max(_np.r_[-x[x < 0], 0])/(1 + 1/c))
    elif scale is None:
        scale = 1.0

    if sampwidth == 1:
        int_min = 0
        midpoint = 128
        int_max = 255
    else:
        int_min = -2**(nbits - 1)
        midpoint = 0
        int_max = 2**(nbits - 1) - 1

    scaled_x = x / scale
    if _np.any(scaled_x > 1) or _np.any(scaled_x < -1 - 1/c):
        msg = (f'Some data values have been clipped.  With scale={scale}, the '
               'interval of input values that will not be clipped '
               f'is [{-(1 + 1/c)*scale}, {scale}]')
        if clip == "warn":
            _warnings.warn(ClippedDataWarning(msg))
        elif clip == "raise":
            raise ClippedDataError(msg)

    y = midpoint + _round_with_half_towards_zero(x/scale*c)
    return _np.clip(y, int_min, int_max).astype(_sampwidth_dtypes[sampwidth])


def _round_with_half_towards_zero(x):
    s = _np.sign(x)
    return s * _np.ceil(_np.abs(x) - 0.5)


_sampwidth_t = Literal[1, 2, 3, 4]


def _as_chunk(obj):
    """Coerce `obj` to a `Chunk`, with a clear error if that fails."""
    if isinstance(obj, Chunk):
        return obj
    try:
        return Chunk(id=obj.id, data=obj.data, pad=getattr(obj, 'pad', None))
    except AttributeError:
        raise TypeError('each item of `chunks` must be a wavio.Chunk (or '
                        'an object with `id` and `data` attributes), got '
                        f'{obj!r}.')


def _fmt_chunk_data(nchannels, sampwidth, rate):
    """Generate the 16 byte PCM 'fmt ' chunk payload.

    This is exactly the payload written by the `wave` module of the
    standard library.
    """
    return _struct.pack('<HHIIHH', 1, nchannels, rate,
                        rate*nchannels*sampwidth,
                        nchannels*sampwidth, 8*sampwidth)


def _fmt_matches(fmt_data, nchannels, sampwidth, rate):
    """
    Return True if the 'fmt ' chunk payload `fmt_data` describes the same
    PCM format as the given parameters.  This allows the original 'fmt '
    bytes (which might use a nonminimal encoding such as 18 bytes with a
    cbSize field, or WAVE_FORMAT_EXTENSIBLE) to be passed through on a
    read/write round trip.
    """
    if len(fmt_data) < 16:
        return False
    tag, ch, r, byterate, blockalign, bits = _struct.unpack('<HHIIHH',
                                                            fmt_data[:16])
    if tag not in (1, 0xFFFE):  # WAVE_FORMAT_PCM, WAVE_FORMAT_EXTENSIBLE
        return False
    return (ch == nchannels and r == rate and bits == 8*sampwidth
            and blockalign == nchannels*sampwidth
            and byterate == rate*blockalign)


def _merge_chunks(chunks, nchannels, sampwidth, rate, wavdata):
    """
    Merge `chunks` with the 'fmt ' and 'data' chunks generated from the
    other arguments, and return a list of (id, payload, pad) tuples in
    output order.  'fmt ' and 'data' entries in `chunks` are used as
    position markers only; their payloads are regenerated (the original
    'fmt ' payload is kept if it matches the output format).  If `chunks`
    contains no 'fmt ' or 'data' entry, the 'fmt ' chunk is placed first
    and the 'data' chunk last.
    """
    out = []  # items are ('fmt', src), ('data', src) or ('raw', chunk)
    fmt_src = None
    data_src = None
    seen_singletons = set()
    for obj in chunks:
        ch = _as_chunk(obj)
        if ch.id in _singleton_chunk_ids:
            if ch.id in seen_singletons:
                raise WavChunkError(f'duplicate chunk {ch.id!r} in `chunks`.')
            seen_singletons.add(ch.id)
        if ch.id == b'fmt ':
            fmt_src = ch
            out.append(('fmt', ch))
        elif ch.id == b'data':
            data_src = ch
            out.append(('data', ch))
        else:
            out.append(('raw', ch))
    if fmt_src is None:
        out.insert(0, ('fmt', None))
    if data_src is None:
        out.append(('data', None))

    fmt_data = _fmt_chunk_data(nchannels, sampwidth, rate)
    fmt_pad = False
    if fmt_src is not None and _fmt_matches(fmt_src.data, nchannels,
                                            sampwidth, rate):
        # The original 'fmt ' payload describes the same format; keep its
        # exact bytes so a read/write round trip is byte-for-byte exact.
        fmt_data = fmt_src.data
        fmt_pad = fmt_src.pad
    data_pad = data_src.pad if data_src is not None else False

    merged = []
    for kind, ch in out:
        if kind == 'fmt':
            merged.append((b'fmt ', fmt_data, fmt_pad))
        elif kind == 'data':
            merged.append((b'data', wavdata, data_pad))
        else:
            merged.append((ch.id, ch.data, ch.pad))
    return merged


def _build_riff(chunks, nchannels, sampwidth, rate, wavdata):
    """
    Build the complete RIFF/WAVE byte string from `chunks` merged with
    the generated 'fmt ' and 'data' chunks.  All chunk sizes, padding and
    the RIFF size field are recomputed from the actual payloads, so the
    chunk offsets are consistent with the final position of the 'data'
    chunk.
    """
    merged = _merge_chunks(chunks, nchannels, sampwidth, rate, wavdata)
    riff_size = 4  # 4 bytes for the 'WAVE' form type
    for _, payload, pad in merged:
        riff_size += 8 + len(payload) + (1 if pad else 0)
    out = bytearray()
    out += b'RIFF' + _struct.pack('<I', riff_size) + b'WAVE'
    for cid, payload, pad in merged:
        out += cid + _struct.pack('<I', len(payload)) + payload
        if pad:
            out += b'\x00'
    return bytes(out)


def write(file: BinaryIO | str,
          data: ArrayLike,
          rate: int,
          scale: float | Literal["auto"] | None = None,
          sampwidth: _sampwidth_t | None = None,
          clip: Literal["ignore", "warn", "raise"] = "warn",
          chunks: Iterable[Chunk] | None = None) -> None:
    """
    Write the numpy array `data` to a WAV file.

    The Python standard library "wave" is used to write the data to the
    file, so this function has the same limitations as that module.  In
    particular, the Python library does not support floating point data,
    so this function must convert floating point input to integers before
    writing the data to the file.  See below for the conversion rules.

    *Important notes*

    * If `data` has an *integer* data type, signed or unsigned and any bit
      depth, the values are never scaled or shifted.  The only possible
      changes to the values that can occur is if the data must be clipped
      to fit the desired output sample width.  It is an error to give a
      value for the `scale` parameter if `data` has an integer data type.
    * If `data` is a floating point type, `sampwidth` must be given.  The
      default behavior is to scale input values in the range [-1, 1] to
      the output range [min_int+1, max_int], where min_int and max_int are
      the minimum and maximum values of the output data type determined by
      `sampwidth`.  See the description of `scale` below for more options.

    Parameters
    ----------
    file : string, or file object open for writing in binary mode
        Either the name of a file or an open file pointer.
    data : numpy array, 1- or 2-dimensional, integer or floating point
        If it is 2-d, the rows are the frames (i.e. samples) and the
        columns are the channels.
    rate : int
        The sampling frequency (i.e. frame rate) of the data.
    sampwidth : int, optional
        The sample width, in bytes, of the output file.
        If `sampwidth` is not given, it is inferred (if possible) from
        the data type of `data`, as follows::

            data.dtype     sampwidth
            ----------     ---------
            uint8, int8        1
            uint16, int16      2
            uint32, int32      4

        For any other data types, or to write a 24 bit file, `sampwidth`
        must be given.
    scale : float | Literal["auto"], optional
        This controls the output range when the input is floating point.
        `scale` must not be given when the input data has integer data
        type.

        if `scale` is a numeric value, then input values in the range
        `[-scale, scale]` are mapped to the integer output range centered
        at the midpoint of the output range.  For 8 bit unsigned integer
        output (i.e. `sampwdith=1`), the midpoint is 128. For `sampwidth`
        2, 3 or 4 (corresponding to signed integer output), the midpoint
        is 0.

        If `scale` is the string `"auto"`, the data written to the file is
        scaled up or down to occupy the full range of the output data type.
        For example, with `sampwidth=2` the input
        `data=[-0.5, -1.0, 0.0, 0.25, 1.0]` would result in
        the 16 signed integers [-16384, -32767, 0, 8192, 32767] being
        written to the WAV file. By setting `scale=2`, the output values
        would be [-8192, -16384, 0, 4096, 16384].

        When the input is floating point and `scale` is not given, the
        default value `scale=1.0` is used.
    clip : str, optional
        If "warn" (the default), the function will generate a warning if
        any of the data values must be clipped when written to the format
        of the output array.  If "raise", the function will raise an
        exception if clipping occurs.  If "ignore", no warning or
        exception is generated if clipping occurs.
    chunks : sequence of `wavio.Chunk`, optional
        Additional RIFF chunks to merge into the output file, in the
        given order.  Unknown chunks are passed through unchanged.  The
        chunk sizes, padding and the RIFF size field are recomputed when
        the chunks are merged, so all chunk offsets are consistent with
        the final position of the 'data' chunk.

        Entries with the id ``b'fmt '`` or ``b'data'`` are used as
        position markers only; their contents are generated from the
        other parameters (the original 'fmt ' payload is kept if it
        describes the same PCM format).  If `chunks` contains no 'fmt '
        or 'data' entry, the 'fmt ' chunk is written first and the
        'data' chunk last.

        Passing the `chunks` attribute of the `wavio.Wav` object returned
        by `wavio.read` preserves the extended chunks of the original
        file (cue points, loop points, 'LIST'/'INFO' metadata, etc.);
        for a standard PCM file, the output of such a read/write round
        trip is byte-for-byte identical to the input.  A duplicate of a
        chunk that may occur at most once (such as 'fmt ', 'data' or
        'cue ') raises `wavio.WavChunkError`.

    Examples
    --------
    Create a 3 second 440 Hz sine wave, and save it in a 24-bit WAV file.

    >>> import numpy as np
    >>> import wavio

    >>> rate = 22050           # samples per second
    >>> T = 3                  # sample duration (seconds)
    >>> n = int(rate*T)        # number of samples
    >>> t = np.arange(n)/rate  # grid of time values

    >>> f = 440.0              # sound frequency (Hz)
    >>> x = np.sin(2*np.pi * f * t)

    `x` is a single sine wave with amplitude 1, so we can use the default
    `scale`.

    >>> wavio.write("sine24.wav", x, rate, sampwidth=3)

    Create a file that contains the 16 bit integer values -10000 and 10000
    repeated 100 times.  Use a sample rate of 8000.

    >>> x = np.empty(200, dtype=np.int16)
    >>> x[::2] = -10000
    >>> x[1::2] = 10000
    >>> wavio.write("foo.wav", x, 8000)

    Check that the file contains what we expect.  The values are checked
    for exact equality.  The input was an integer array, so the values are
    not scaled.

    >>> w = wavio.read("foo.wav")
    >>> np.all(w.data[:, 0] == x)
    True

    Write floating point data to a 16 bit WAV file.  The floating point
    values are assumed to be within the range [-2, 2], and we want the
    values 2 and -2 to correspond to the full output range, even if the
    actual values in the data do not fill this range.  We do that by
    specifying `scale=2`.

    `T`, `rate` and `t` are from above.  The data is the sum of two
    sinusoids, with frequencies 440 and 880 Hz, modulated by a parabolic
    curve that is zero at the start and end of the data.

    >>> envelope = (4/T**2)*(t * (T - t))
    >>> omega1 = 2*np.pi*440
    >>> omega2 = 2*np.pi*880
    >>> y = envelope*(np.sin(omega1*t) + 0.3*np.sin(omega2*t + 0.2))
    >>> y.min(), y.max()
    (-1.1745469775555515, 1.093833464065767)

    Write the WAV file, with `scale=2`.

    >>> wavio.write('harmonic.wav', y, rate, sampwidth=2, scale=2)

    Check the minimum and maximum integers that were actually written
    to the file:

    >>> w = wavio.read("harmonic.wav")
    >>> w.data.min(), w.data.max()
    (-19243, 17921)

    If we want the WAV file to use as much of the range of the output
    integer type as possible (while still mapping 0.0 in the input to 0 in
    the output), we set `scale="auto"`.

    >>> wavio.write('harmonic_full.wav', y, rate, sampwidth=2, scale="auto")

    >>> w = wavio.read('harmonic_full.wav')
    >>> w.data.min(), w.data.max()
    (-32768, 30517)

    """
    if clip not in ["ignore", "warn", "raise"]:
        raise ValueError('clip must be one of "ignore", "warn" or "raise".')

    data_arr = _np.asarray(data)

    if sampwidth is None:
        if (not _np.issubdtype(data_arr.dtype, _np.integer)
                or data_arr.itemsize) > 4:
            raise ValueError('when data.dtype is not an 8-, 16-, or 32-bit '
                             'integer type, sampwidth must be specified.')
        sampwidth = cast(_sampwidth_t, data_arr.itemsize)
    else:
        if sampwidth not in [1, 2, 3, 4]:
            raise ValueError('sampwidth must be 1, 2, 3 or 4.')

    outdtype = _sampwidth_dtypes[sampwidth]
    outmin, outmax = _sampwidth_minmax[sampwidth]

    if _np.issubdtype(data_arr.dtype, _np.integer):
        if scale is not None:
            raise ValueError('The scale parameter must not be set when the '
                             'input is an integer array.  No shifting or '
                             'scaling is done to integer input values.')
        if (data_arr.min() < outmin or data_arr.max() > outmax):
            if clip == "warn":
                _warnings.warn(ClippedDataWarning())
            elif clip == "raise":
                raise ClippedDataError()
        clip_min = max(outmin, _np.iinfo(data_arr.dtype).min)
        clip_max = min(outmax, _np.iinfo(data_arr.dtype).max)
        data_arr = data_arr.clip(clip_min, clip_max).astype(outdtype)
    elif _np.issubdtype(data_arr.dtype, _np.floating):
        data_arr = _float_to_integer(data_arr, sampwidth,
                                     scale=scale, clip=clip)
    else:
        raise TypeError(f'unsupported input array data type: {data_arr.dtype}')

    # At this point, `data_arr` has been converted to have one of the
    # following dtypes:
    #
    #    sampwidth   dtype
    #    ---------   -----
    #        1       uint8
    #        2       int16
    #        3       int32
    #        4       int32
    #
    # The values in `data_arr` are in the form in which they will be saved;
    # no more scaling will take place.

    if data_arr.ndim == 1:
        data_arr = data_arr.reshape(-1, 1)

    wavdata = _array2wav(data_arr, sampwidth)

    if chunks is not None:
        payload = _build_riff(chunks, data_arr.shape[1], sampwidth, rate,
                              wavdata)
        if isinstance(file, (str, _os.PathLike)):
            with open(file, 'wb') as f:
                f.write(payload)
        else:
            file.write(payload)
        return

    with _wave.open(file, 'wb') as w:
        w.setnchannels(data_arr.shape[1])
        w.setsampwidth(sampwidth)
        w.setframerate(rate)
        w.writeframes(wavdata)
