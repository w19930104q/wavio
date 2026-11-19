"""
The wavio module defines the functions:

read(file)
    Read a WAV file and return a `wavio.Wav` object, with attributes
    `data`, `rate` and `sampwidth`.

write(filename, data, rate, scale=None, sampwidth=None)
    Write a numpy array to a WAV file.


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

from typing import BinaryIO, Literal, cast
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


class WavReadError(ValueError):
    """
    Error raised when `wavio.read` encounters a malformed WAV/RIFF file.

    The message always identifies the offending RIFF chunk (its four-byte
    chunk ID) and the absolute byte offset in the file at which the problem
    was detected, so that bad files can be located during batch processing.

    Attributes
    ----------
    chunk : bytes or None
        The four-byte ID of the offending RIFF chunk (e.g. ``b'data'``).
        It is ``None`` when the problem is in the outermost RIFF structure,
        where no subchunk exists yet (e.g. a file that does not begin with
        a RIFF header).
    offset : int or None
        The absolute byte offset in the file at which the problem was
        detected.
    """

    def __init__(self, message, chunk=None, offset=None):
        self.chunk = chunk
        self.offset = offset
        super().__init__(message)


def _riff_error(message, chunk, offset):
    if chunk is None:
        where = "RIFF structure"
    else:
        where = f"chunk {chunk!r}"
    if offset is None:
        return WavReadError(f"malformed WAV file in {where}: {message}",
                            chunk=chunk, offset=offset)
    return WavReadError(
        f"malformed WAV file in {where} at byte offset {offset}: {message}",
        chunk=chunk, offset=offset)


_WAVE_FORMAT_PCM = 0x0001
_WAVE_FORMAT_EXTENSIBLE = 0xFFFE
# Derived from uuid.UUID("00000001-0000-0010-8000-00aa00389b71").bytes_le
_KSDATAFORMAT_SUBTYPE_PCM = (b'\x01\x00\x00\x00\x00\x00\x10\x00'
                             b'\x80\x00\x00\xaa\x008\x9bq')


class _RiffReader:
    """
    Thin wrapper around a binary file that tracks the absolute byte offset.

    Works with both seekable files and non-seekable streams; `skip` falls
    back to reading and discarding data when the file cannot seek.
    """

    def __init__(self, file):
        self.file = file
        try:
            start = file.tell()
        except (AttributeError, OSError):
            self.offset = 0
            self._seekable = False
            self._file_size = None
        else:
            self.offset = start
            try:
                file.seek(0, 2)
                self._file_size = file.tell()
                file.seek(start, 0)
            except OSError:
                # seek/tell exist but the stream is not truly seekable.
                self._seekable = False
                self._file_size = None
            else:
                self._seekable = True

    def read(self, size):
        data = self.file.read(size)
        self.offset += len(data)
        return data

    def skip(self, size, chunk=None):
        """Skip exactly `size` bytes; raise WavReadError on a short read."""
        if size <= 0:
            return
        if self._seekable:
            if self.offset + size > self._file_size:
                raise _riff_error(
                    f"file is truncated: expected {size} more byte(s) but "
                    f"only {max(0, self._file_size - self.offset)} are "
                    "available",
                    chunk=chunk, offset=self.offset)
            try:
                self.file.seek(size, 1)
            except OSError:
                self._seekable = False
            else:
                self.offset += size
                return
        remaining = size
        while remaining > 0:
            data = self.file.read(min(remaining, 1 << 16))
            if not data:
                raise _riff_error(
                    f"file is truncated: expected {size} more byte(s), only "
                    f"{size - remaining} available",
                    chunk=chunk, offset=self.offset)
            self.offset += len(data)
            remaining -= len(data)


def _read_exact(reader, size, chunk, offset, what):
    """Read exactly `size` bytes, reporting chunk/offset on truncation."""
    data = reader.read(size)
    if len(data) < size:
        raise _riff_error(
            f"file is truncated while reading {what}: expected {size} "
            f"byte(s) at offset {offset}, only {len(data)} available",
            chunk=chunk, offset=offset)
    return data


def _skip_chunk_body(reader, chunk_id, chunk_size, body_offset, riff_end):
    """Skip the body (and padding) of a chunk that is not needed."""
    body_end = body_offset + chunk_size
    if body_end > riff_end:
        raise _riff_error(
            f"chunk size ({chunk_size} bytes) extends past the end of the "
            f"RIFF container (payload ends at byte {riff_end}); declared "
            f"body would end at byte {body_end}",
            chunk=chunk_id, offset=body_offset - 8)
    reader.skip(chunk_size + (chunk_size & 1), chunk=chunk_id)


def _read_fmt_chunk(reader, chunk_id, chunk_size, body_offset):
    """Parse a 'fmt ' chunk.  Consumes the whole body plus padding."""
    if chunk_size < 16:
        raise _riff_error(
            f"chunk is too small for a PCM format chunk: size is "
            f"{chunk_size} bytes, at least 16 are required",
            chunk=chunk_id, offset=body_offset - 8)
    # wFormatTag, nchannels, nSamplesPerSec, nAvgBytesPerSec, nBlockAlign.
    # Only the format tag, channel count and rate are needed; the other two
    # fields are consumed by the fixed 14-byte unpack but not validated.
    fmt_head = _read_exact(reader, 14, chunk_id, body_offset,
                           "the format fields of the 'fmt ' chunk")
    format_tag, nchannels, rate, _avg_bytes_per_sec, _block_align = (
        _struct.unpack_from('<HHLLH', fmt_head))
    # wBitsPerSample, at body offset 14.
    bits = _struct.unpack_from(
        '<H', _read_exact(reader, 2, chunk_id, body_offset + 14,
                          "wBitsPerSample in the 'fmt ' chunk"))[0]
    if format_tag not in (_WAVE_FORMAT_PCM, _WAVE_FORMAT_EXTENSIBLE):
        raise _riff_error(
            f"unsupported format tag 0x{format_tag:04x}; only uncompressed "
            "PCM (0x0001) and PCM in a WAVEFORMATEXTENSIBLE chunk "
            "(0xfffe) are supported",
            chunk=chunk_id, offset=body_offset)
    if format_tag == _WAVE_FORMAT_EXTENSIBLE:
        # The extension adds cbSize(2) + wValidBitsPerSample(2) +
        # dwChannelMask(4) + SubFormat(16) = 24 bytes.
        if chunk_size < 40:
            raise _riff_error(
                f"WAVEFORMATEXTENSIBLE format chunk is truncated: size is "
                f"{chunk_size} bytes, 40 are required",
                chunk=chunk_id, offset=body_offset - 8)
        # cbSize, wValidBitsPerSample and dwChannelMask: 8 bytes that are
        # consumed but not validated, followed by the 16-byte SubFormat.
        _read_exact(reader, 8, chunk_id, body_offset + 16,
                    "the WAVEFORMATEXTENSIBLE extension header")
        subformat = _read_exact(reader, 16, chunk_id, body_offset + 24,
                                "the SubFormat GUID of the "
                                "WAVEFORMATEXTENSIBLE extension")
        if subformat != _KSDATAFORMAT_SUBTYPE_PCM:
            raise _riff_error(
                "WAVEFORMATEXTENSIBLE SubFormat is not PCM "
                f"(got {subformat.hex('-')}); only uncompressed PCM data "
                "is supported",
                chunk=chunk_id, offset=body_offset + 24)
    # Any remaining (extension) bytes, followed by the word-alignment pad.
    reader.skip(body_offset + chunk_size - reader.offset, chunk=chunk_id)
    reader.skip(chunk_size & 1, chunk=chunk_id)

    sampwidth = (bits + 7) // 8
    if sampwidth == 0:
        raise _riff_error(
            f"bad sample width: wBitsPerSample={bits} gives a sample width "
            "of 0 bytes",
            chunk=chunk_id, offset=body_offset + 14)
    if sampwidth > 4:
        raise _riff_error(
            f"unsupported sample width: wBitsPerSample={bits} gives a "
            f"sample width of {sampwidth} bytes, at most 4 bytes are "
            "supported",
            chunk=chunk_id, offset=body_offset + 14)
    if nchannels == 0:
        raise _riff_error(
            "bad number of channels: nchannels is 0",
            chunk=chunk_id, offset=body_offset + 2)
    return rate, nchannels, sampwidth


def _parse_riff(file):
    """
    Parse the RIFF/WAVE structure of an open binary file.

    Returns
    -------
    rate, nchannels, sampwidth, data_bytes : tuple
        Every structural problem raises WavReadError naming the offending
        chunk and the absolute byte offset where the problem was detected.
    """
    reader = _RiffReader(file)

    header_offset = reader.offset
    header = reader.read(12)
    if len(header) < 4:
        raise _riff_error(
            f"file is truncated while reading the RIFF header: expected 4 "
            f"bytes at offset {header_offset}, only {len(header)} available",
            chunk=None, offset=header_offset)
    if header[:4] != b'RIFF':
        raise _riff_error(
            f"file does not start with a RIFF id; got {header[:4]!r}",
            chunk=None, offset=header_offset)
    if len(header) < 8:
        raise _riff_error(
            f"file is truncated while reading the RIFF size field: "
            f"expected 4 bytes at offset {header_offset + 4}, only "
            f"{len(header) - 4} available",
            chunk=None, offset=header_offset + 4)
    if len(header) < 12:
        raise _riff_error(
            f"file is truncated while reading the RIFF form type: expected "
            f"4 bytes at offset {header_offset + 8}, only "
            f"{len(header) - 8} available",
            chunk=None, offset=header_offset + 8)
    if header[8:12] != b'WAVE':
        raise _riff_error(
            f"not a WAVE file: RIFF form type is {header[8:12]!r}, "
            "expected b'WAVE'",
            chunk=None, offset=header_offset + 8)

    riff_size = _struct.unpack_from('<L', header, 4)[0]
    # The RIFF payload (form type + subchunks) has `riff_size` bytes; the
    # whole container occupies 8 + riff_size bytes.
    riff_end = header_offset + 8 + riff_size

    fmt = None
    data_info = None
    while True:
        chunk_offset = reader.offset
        chunk_header = reader.read(8)
        if len(chunk_header) == 0:
            break
        if len(chunk_header) < 8:
            raise _riff_error(
                f"file is truncated while reading a chunk header: expected "
                f"8 bytes at offset {chunk_offset}, only "
                f"{len(chunk_header)} available",
                chunk=None, offset=chunk_offset)
        chunk_id = chunk_header[:4]
        chunk_size = _struct.unpack_from('<L', chunk_header, 4)[0]
        body_offset = chunk_offset + 8
        body_end = body_offset + chunk_size
        if body_end > riff_end:
            raise _riff_error(
                f"declared chunk size ({chunk_size} bytes) extends past "
                f"the end of the RIFF container (payload ends at byte "
                f"{riff_end}); the chunk body would end at byte {body_end}",
                chunk=chunk_id, offset=chunk_offset)

        if chunk_id == b'fmt ':
            fmt = _read_fmt_chunk(reader, chunk_id, chunk_size, body_offset)
        elif chunk_id == b'data':
            if fmt is None:
                raise _riff_error(
                    "data chunk encountered before the 'fmt ' chunk; the "
                    "format chunk must precede the audio data",
                    chunk=chunk_id, offset=chunk_offset)
            data_info = (chunk_id, chunk_size, body_offset)
            break
        else:
            _skip_chunk_body(reader, chunk_id, chunk_size, body_offset,
                             riff_end)

    if fmt is None:
        raise _riff_error(
            "required 'fmt ' chunk is missing from the RIFF container",
            chunk=None, offset=header_offset)
    if data_info is None:
        raise _riff_error(
            "required 'data' chunk is missing from the RIFF container",
            chunk=None, offset=header_offset)

    rate, nchannels, sampwidth = fmt
    data_id, data_size, data_body_offset = data_info
    data = _read_exact(reader, data_size, data_id, data_body_offset,
                       "the audio samples of the 'data' chunk")
    # RIFF chunks are word-aligned, but the standard library wave writer
    # omits the padding byte of an odd-sized final data chunk, so a missing
    # pad at EOF is tolerated just like wave._Chunk.read does.
    if data_size & 1:
        reader.read(1)
    return rate, nchannels, sampwidth, data


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

    """
    data: _np.ndarray
    rate: int
    sampwidth: int

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

    Notes
    -----
    The RIFF/WAVE structure of the file is parsed by this module rather
    than the standard library ``wave`` module.  Errors raised while parsing
    a malformed file are `wavio.WavReadError` exceptions whose message names
    the offending RIFF chunk and the byte offset where the problem was
    found.  Like the ``wave`` module, this function does not read compressed
    WAV files, and it does not read files with floating point data.

    The array returned by `wavio.read` is always two-dimensional.  If the
    WAV data is mono, the array will have shape (num_samples, 1).

    `wavio.read()` does not scale or normalize the data.  The data in the
    array `wav.data` is the data that was in the file.  When the file
    contains 24 bit samples, the resulting numpy array is 32 bit integers,
    with values that have been sign-extended.
    """
    opened_file = None
    if isinstance(file, (str, _os.PathLike)):
        opened_file = open(file, 'rb')
        fileobj = opened_file
    elif isinstance(file, (bytes, bytearray)):
        # Treat raw bytes as in-memory file contents.
        fileobj = _io.BytesIO(file)
    else:
        # Assume it is already an open binary file object.
        fileobj = file
    try:
        rate, nchannels, sampwidth, data_bytes = _parse_riff(fileobj)
        # Match the standard library wave module: nframes is the declared
        # data size divided by the frame size, so a partial frame left over
        # at the end of the data chunk is ignored rather than rejected.
        frame_size = nchannels * sampwidth
        usable_len = len(data_bytes) - (len(data_bytes) % frame_size)
        if usable_len != len(data_bytes):
            data_bytes = data_bytes[:usable_len]
        array = _wav2array(nchannels, sampwidth, data_bytes)
    finally:
        if opened_file is not None:
            opened_file.close()
    return Wav(data=array, rate=rate, sampwidth=sampwidth)


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


def write(file: BinaryIO | str,
          data: ArrayLike,
          rate: int,
          scale: float | Literal["auto"] | None = None,
          sampwidth: _sampwidth_t | None = None,
          clip: Literal["ignore", "warn", "raise"] = "warn") -> None:
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

    with _wave.open(file, 'wb') as w:
        w.setnchannels(data_arr.shape[1])
        w.setsampwidth(sampwidth)
        w.setframerate(rate)
        w.writeframes(wavdata)
