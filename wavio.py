"""
The wavio module defines the functions and classes:

read(file)
    Read a WAV file and return a `wavio.Wav` object, with attributes
    `data`, `rate` and `sampwidth`.

read_chunks(file, chunk_size)
    Read a WAV file incrementally, yielding chunks of at most
    `chunk_size` frames at a time.

write(filename, data, rate, scale=None, sampwidth=None)
    Write a numpy array to a WAV file.

WavWriter(file, rate, sampwidth, nchannels, scale=None, clip="warn")
    Stream chunks of frames to a WAV file without holding the complete
    data set in memory.  Use it as a context manager and call
    `write_chunk` for each chunk of frames.

write_chunks(chunks, file, rate, scale=None, sampwidth=None, clip="warn")
    Write an iterable of array chunks to a WAV file; a convenience
    wrapper around `WavWriter`.


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
    This function uses the `wave` module of the Python standard libary
    to read the WAV file, so it has the same limitations as that library.
    In particular, the function does not read compressed WAV files, and
    it does not read files with floating point data.

    The array returned by `wavio.read` is always two-dimensional.  If the
    WAV data is mono, the array will have shape (num_samples, 1).

    `wavio.read()` does not scale or normalize the data.  The data in the
    array `wav.data` is the data that was in the file.  When the file
    contains 24 bit samples, the resulting numpy array is 32 bit integers,
    with values that have been sign-extended.
    """
    with _wave.open(file) as wav:
        rate = wav.getframerate()
        nchannels = wav.getnchannels()
        sampwidth = wav.getsampwidth()
        nframes = wav.getnframes()
        data = wav.readframes(nframes)
    array = _wav2array(nchannels, sampwidth, data)
    return Wav(data=array, rate=rate, sampwidth=sampwidth)


def read_chunks(file: BinaryIO | str, chunk_size: int):
    """
    Read a WAV file incrementally, one chunk of frames at a time.

    This is a generator that yields two-dimensional numpy arrays, each
    containing at most `chunk_size` frames (rows) of the WAV data.  In
    contrast to `read`, the complete data set is never held in memory;
    the working set is bounded by `chunk_size` and the sample width.
    The conversion of each chunk (including the sign extension of 24 bit
    data to 32 bit integers) is identical to that performed by `read`,
    so concatenating the yielded chunks gives exactly the same array as
    `read` would return::

        w = wavio.read(filename)
        chunks = list(wavio.read_chunks(filename, chunk_size=1000))
        assert np.array_equal(np.concatenate(chunks, axis=0), w.data)

    Parameters
    ----------
    file : string or file object
        Either the name of a file or an open file pointer.
    chunk_size : int
        The maximum number of frames (rows) included in each yielded
        array.  Must be a positive integer.  The final chunk (and only
        the final chunk) may contain fewer frames.

    Yields
    ------
    chunk : numpy array, shape (n, nchannels)
        A chunk of the data read from the WAV file.  The data type is
        determined by `sampwidth`, exactly as in `read` (uint8 for
        8 bit, int16 for 16 bit, int32 for 24 and 32 bit).  Every
        yielded array is two-dimensional, even for mono data.

    Notes
    -----
    Like `read`, this function relies on the standard library `wave`
    module and therefore shares its limitations (no compressed or
    floating point WAV data).
    """
    if not isinstance(chunk_size, (int, _np.integer)) or chunk_size <= 0:
        raise ValueError('chunk_size must be a positive integer.')

    with _wave.open(file) as wav:
        nchannels = wav.getnchannels()
        sampwidth = wav.getsampwidth()
        while True:
            data = wav.readframes(chunk_size)
            if not data:
                break
            yield _wav2array(nchannels, sampwidth, data)


_sampwidth_dtypes = {1: _np.uint8,
                     2: _np.int16,
                     3: _np.int32,
                     4: _np.int32}
_sampwidth_minmax = {1: (0, 255),
                     2: (-2**15, 2**15 - 1),
                     3: (-2**23, 2**23 - 1),
                     4: (-2**31, 2**31 - 1)}


def _auto_scale(x, c):
    # The scale used by `write` when scale="auto".
    return max(_np.max(_np.r_[x[x > 0], 0]),
               _np.max(_np.r_[-x[x < 0], 0])/(1 + 1/c))


def _float_to_integer_clip_message(scale, c):
    return (f'Some data values have been clipped.  With scale={scale}, the '
            'interval of input values that will not be clipped '
            f'is [{-(1 + 1/c)*scale}, {scale}]')


def _float_to_integer_chunk(x, sampwidth, scale=None, clip="warn"):
    """
    Convert one chunk of floating point data to the output integer type.

    This function contains the actual scale/round/clip arithmetic.  It is
    written so that applying it independently to the chunks of an array
    produces exactly the same values as applying it to the complete
    array, provided `scale` is not "auto" (the auto scale depends on the
    extrema of the complete data set and cannot be computed from a
    single chunk).

    Returns
    -------
    y : numpy array
        The converted chunk, with the output integer dtype.
    clipped : bool
        Whether at least one value in the chunk was clipped.
    msg : str or None
        The clipping message for this chunk, or None when no clipping
        occurred.
    """
    # For a given sampwidth and scale, the actual allowed
    # interval for float input is [-(1 + 1/c)*scale, scale],
    # where c = 2**(8*sampwidth - 1) - 0.5.  Values outside
    # that interval will result in clipping.

    nbits = 8*sampwidth
    c = 2**(nbits - 1) - 0.5

    if scale is None:
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
    clipped = bool(_np.any(scaled_x > 1)
                   or _np.any(scaled_x < -1 - 1/c))
    msg = _float_to_integer_clip_message(scale, c) if clipped else None
    if clipped:
        if clip == "warn":
            _warnings.warn(ClippedDataWarning(msg))
        elif clip == "raise":
            raise ClippedDataError(msg)

    y = midpoint + _round_with_half_towards_zero(x/scale*c)
    return _np.clip(y, int_min, int_max).astype(_sampwidth_dtypes[sampwidth]), \
        clipped, msg


def _float_to_integer(x, sampwidth, scale=None, clip="warn"):
    nbits = 8*sampwidth
    c = 2**(nbits - 1) - 0.5

    if scale == "auto":
        scale = _auto_scale(x, c)

    y, _, _ = _float_to_integer_chunk(x, sampwidth, scale=scale, clip=clip)
    return y


def _integer_to_integer(x, sampwidth, clip="warn"):
    """
    Clip (if needed) and convert one chunk of integer data to the
    output integer type.  Integer values are never scaled or shifted.

    Returns
    -------
    y : numpy array
        The converted chunk, with the output integer dtype.
    clipped : bool
        Whether at least one value in the chunk was clipped.
    """
    outmin, outmax = _sampwidth_minmax[sampwidth]
    clipped = bool(x.min() < outmin or x.max() > outmax)
    if clipped:
        if clip == "warn":
            _warnings.warn(ClippedDataWarning())
        elif clip == "raise":
            raise ClippedDataError()
    clip_min = max(outmin, _np.iinfo(x.dtype).min)
    clip_max = min(outmax, _np.iinfo(x.dtype).max)
    return x.clip(clip_min, clip_max).astype(_sampwidth_dtypes[sampwidth]), \
        clipped


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

    if _np.issubdtype(data_arr.dtype, _np.integer):
        if scale is not None:
            raise ValueError('The scale parameter must not be set when the '
                             'input is an integer array.  No shifting or '
                             'scaling is done to integer input values.')
        data_arr, _ = _integer_to_integer(data_arr, sampwidth, clip=clip)
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


class WavWriter:
    """
    Write a WAV file one chunk of frames at a time.

    `WavWriter` is a streaming counterpart of `write`: instead of holding
    the complete data set in memory, it converts and writes each chunk
    as it is received.  The peak working set is therefore bounded by the
    size of the chunks passed to `write_chunk`, even when writing 24 bit
    data (which requires an intermediate 32 bit representation).

    The scaling, rounding and clipping rules are the same as in `write`;
    each chunk is converted independently with the same arithmetic, so a
    file written in chunks is byte-for-byte identical to a file written
    with `write` from the concatenated chunks.

    `WavWriter` is intended to be used as a context manager::

        with wavio.WavWriter("large.wav", rate=44100, sampwidth=3,
                             nchannels=2) as w:
            for chunk in wavio.read_chunks("source.wav",
                                           chunk_size=10000):
                w.write_chunk(chunk)

    Parameters
    ----------
    file : string or file object open for writing in binary mode
        Either the name of a file or an open file pointer.
    rate : int
        The sampling frequency (i.e. frame rate) of the data.
    sampwidth : int
        The sample width, in bytes, of the output file.  Must be 1, 2,
        3 or 4.
    nchannels : int
        The number of audio channels.  Every chunk written with
        `write_chunk` must have this many columns.
    scale : float, optional
        Controls the output range when floating point chunks are
        written, with the same meaning as in `write`.  It must not be
        set when the chunks have an integer data type.  The string
        ``"auto"`` is not supported by the streaming interface, because
        the auto scale requires the extrema of the complete data set;
        compute the desired scale up front and pass it as a number
        instead.
    clip : str, optional
        ``"ignore"``, ``"warn"`` (the default) or ``"raise"``, with the
        same meaning as in `write`.  With ``"warn"``, clipping detected
        in any chunk is collected and reported with a single
        `ClippedDataWarning` when the writer is closed, instead of one
        warning per chunk.
    """

    def __init__(self,
                 file: BinaryIO | str,
                 rate: int,
                 sampwidth: _sampwidth_t,
                 nchannels: int,
                 scale: float | None = None,
                 clip: Literal["ignore", "warn", "raise"] = "warn"):
        if clip not in ["ignore", "warn", "raise"]:
            raise ValueError('clip must be one of "ignore", "warn" or '
                             '"raise".')
        if sampwidth not in [1, 2, 3, 4]:
            raise ValueError('sampwidth must be 1, 2, 3 or 4.')
        if scale == "auto":
            raise ValueError('scale="auto" is not supported by the '
                             'streaming WavWriter, because it requires the '
                             'extrema of the complete data set.  Compute a '
                             'numeric scale up front and pass that instead.')

        self.file = file
        self.rate = rate
        self.sampwidth = sampwidth
        self.nchannels = nchannels
        self.scale = scale
        self.clip = clip
        self._kind = None
        self._clipped = False
        self._clip_message = None
        self._closed = False
        # The file is created up front, so a writer that is closed
        # without any `write_chunk` calls still leaves a valid, empty
        # WAV file on disk.
        self._wav = _wave.open(file, 'wb')
        self._wav.setnchannels(nchannels)
        self._wav.setsampwidth(sampwidth)
        self._wav.setframerate(rate)

    def write_chunk(self, chunk: ArrayLike) -> None:
        """
        Convert and write one chunk of frames.

        Parameters
        ----------
        chunk : numpy array, 1- or 2-dimensional, integer or floating point
            The frames to append to the file.  A one-dimensional array
            is treated as a mono chunk (one channel).  All chunks must
            have the same number of channels and the same data type
            kind (all integer or all floating point).
        """
        if self._closed:
            raise ValueError('write_chunk called after the WavWriter was '
                             'closed.')
        a = _np.asarray(chunk)
        if a.ndim == 1:
            a = a.reshape(-1, 1)
        elif a.ndim != 2:
            raise ValueError('each chunk must be a 1- or 2-dimensional '
                             'array.')
        if a.shape[1] != self.nchannels:
            raise ValueError(f'the chunk has {a.shape[1]} channels, but the '
                             f'writer was created with nchannels='
                             f'{self.nchannels}.')

        kind = ('i' if _np.issubdtype(a.dtype, _np.integer)
                else 'f' if _np.issubdtype(a.dtype, _np.floating)
                else None)
        if kind is None:
            raise TypeError(f'unsupported input array data type: {a.dtype}')
        if self._kind is not None and kind != self._kind:
            raise ValueError('all chunks must have the same data type kind '
                             '(all integer or all floating point).')
        self._kind = kind

        if kind == 'i':
            if self.scale is not None:
                raise ValueError('The scale parameter must not be set when '
                                 'the input is an integer array.  No '
                                 'shifting or scaling is done to integer '
                                 'input values.')
            # Detect clipping without emitting per-chunk warnings; the
            # warning (if any) is emitted once, at close.
            converted, clipped = _integer_to_integer(
                a, self.sampwidth,
                clip="raise" if self.clip == "raise" else "ignore")
        else:
            converted, clipped, msg = _float_to_integer_chunk(
                a, self.sampwidth, scale=self.scale,
                clip="raise" if self.clip == "raise" else "ignore")
            if clipped:
                self._clip_message = msg

        if clipped:
            self._clipped = True

        wavdata = _array2wav(converted, self.sampwidth)
        self._wav.writeframes(wavdata)

    def close(self) -> None:
        """
        Flush and close the underlying WAV file.

        If clipping was observed in any chunk and the writer was created
        with ``clip="warn"``, a single `ClippedDataWarning` is issued
        here.  Calling `close` more than once has no effect.
        """
        if self._closed:
            return
        self._closed = True
        if self._wav is not None:
            self._wav.close()
        if self._clipped and self.clip == "warn":
            if self._kind == 'f' and self._clip_message is not None:
                _warnings.warn(ClippedDataWarning(self._clip_message))
            else:
                _warnings.warn(ClippedDataWarning())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def write_chunks(chunks,
                 file: BinaryIO | str,
                 rate: int,
                 scale: float | Literal["auto"] | None = None,
                 sampwidth: _sampwidth_t | None = None,
                 clip: Literal["ignore", "warn", "raise"] = "warn") -> None:
    """
    Write an iterable of array chunks to a WAV file.

    This is a convenience wrapper around `WavWriter` that mirrors the
    signature of `write`; it is handy for piping a generator such as
    `read_chunks` straight into a new file::

        chunks = wavio.read_chunks("source.wav", chunk_size=10000)
        wavio.write_chunks(chunks, "copy.wav", rate=44100)

    The scaling, rounding and clipping semantics are identical to those
    of `write`, and the resulting file is byte-for-byte identical to the
    file produced by `write` from the concatenated chunks.

    Parameters
    ----------
    chunks : iterable of numpy arrays
        Each chunk must be a 1- or 2-dimensional integer or floating
        point array.  All chunks must have the same data type kind and
        the same number of channels.  The iterable must yield at least
        one chunk.
    file : string or file object open for writing in binary mode
        Either the name of a file or an open file pointer.
    rate : int
        The sampling frequency (i.e. frame rate) of the data.
    sampwidth : int, optional
        The sample width, in bytes, of the output file.  If not given,
        it is inferred from the data type of the first chunk, following
        the same rules as in `write`.
    scale : float, optional
        Controls the output range for floating point chunks, with the
        same meaning as in `write`.  It must not be set for integer
        chunks, and ``"auto"`` is not supported by the streaming
        interface.
    clip : str, optional
        ``"ignore"``, ``"warn"`` (the default) or ``"raise"``, with the
        same meaning as in `write`.
    """
    if clip not in ["ignore", "warn", "raise"]:
        raise ValueError('clip must be one of "ignore", "warn" or "raise".')
    if scale == "auto":
        raise ValueError('scale="auto" is not supported by write_chunks, '
                         'because it requires the extrema of the complete '
                         'data set.  Compute a numeric scale up front and '
                         'pass that instead.')

    iterator = iter(chunks)
    try:
        first = next(iterator)
    except StopIteration:
        raise ValueError('write_chunks requires at least one chunk.')

    first_arr = _np.asarray(first)
    if sampwidth is None:
        if (not _np.issubdtype(first_arr.dtype, _np.integer)
                or first_arr.itemsize > 4):
            raise ValueError('when data.dtype is not an 8-, 16-, or 32-bit '
                             'integer type, sampwidth must be specified.')
        sampwidth = cast(_sampwidth_t, first_arr.itemsize)
    elif sampwidth not in [1, 2, 3, 4]:
        raise ValueError('sampwidth must be 1, 2, 3 or 4.')

    nchannels = 1 if first_arr.ndim == 1 else first_arr.shape[1]
    if _np.issubdtype(first_arr.dtype, _np.integer) and scale is not None:
        raise ValueError('The scale parameter must not be set when the '
                         'input is an integer array.  No shifting or '
                         'scaling is done to integer input values.')

    with WavWriter(file, rate=rate, sampwidth=sampwidth,
                   nchannels=nchannels, scale=scale, clip=clip) as writer:
        writer.write_chunk(first)
        for chunk in iterator:
            writer.write_chunk(chunk)
