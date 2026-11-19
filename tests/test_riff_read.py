"""
Acceptance tests for the self-contained RIFF parser used by ``wavio.read``.

Two groups of tests are included:

1. ``TestMalformedMatrix`` -- a matrix of malformed WAV samples.  Every
   sample must raise ``wavio.WavReadError`` whose message names the
   offending RIFF chunk (or the outer RIFF structure) and reports the
   absolute byte offset where the problem was detected.

2. ``TestRiffEquivalence`` -- differential tests against the standard
   library ``wave`` reader.  For every well-formed sample, the data,
   sample rate and sample width returned by ``wavio.read`` must match the
   values obtained through the old ``wave``-based read path byte for byte.
"""

import io
import os
import struct
import tempfile
import wave

import numpy as np
import pytest

import wavio


# --------------------------------------------------------------------------
# Builders for (mostly) well-formed WAV bytes
# --------------------------------------------------------------------------

def make_fmt(fmt_tag=0x0001, nchannels=1, rate=8000, sampwidth=2,
             bits=None, ext_body=None, fmt_size=None):
    if bits is None:
        bits = sampwidth * 8
    body = struct.pack('<HHLLH', fmt_tag, nchannels, rate,
                       rate * nchannels * sampwidth,
                       nchannels * sampwidth)
    body += struct.pack('<H', bits)
    if ext_body is not None:
        body += ext_body
    if fmt_size is None:
        fmt_size = len(body)
    if len(body) < fmt_size:
        # Pad a truncated-looking-but-declared-larger chunk with zeros;
        # tests that need real truncation just cut the file bytes.
        body = body + b'\x00' * (fmt_size - len(body))
    return b'fmt ' + struct.pack('<L', fmt_size) + body[:fmt_size]


def make_data(payload):
    return b'data' + struct.pack('<L', len(payload)) + payload


def make_riff(chunks, riff_size=None, form=b'WAVE'):
    body = form + b''.join(chunks)
    if riff_size is None:
        riff_size = len(body)
    return b'RIFF' + struct.pack('<L', riff_size) + body


def make_valid_wav(payload=b'\x01\x00\x02\x00\x03\x00\x04\x00',
                   chunks=None, **fmt_kwargs):
    if chunks is None:
        chunks = [make_fmt(**fmt_kwargs), make_data(payload)]
    return make_riff(chunks)


class NonSeekableReader:
    """Binary stream wrapper that disables seek/tell, like a pipe."""

    def __init__(self, data):
        self._buf = io.BytesIO(data)

    def read(self, size=-1):
        if size is None or size < 0:
            return self._buf.read()
        return self._buf.read(size)

    def seek(self, *args):
        raise OSError("stream is not seekable")

    def tell(self):
        raise OSError("stream is not seekable")


# --------------------------------------------------------------------------
# Malformed sample matrix
# --------------------------------------------------------------------------

class TestMalformedMatrix:

    @pytest.mark.parametrize('sample', [
        # (label, bytes, expected chunk id or None, expected offset)
        ("empty file", b"", None, 0),
        ("truncated RIFF id", b"RIF", None, 0),
        ("truncated RIFF size field", b"RIFF\x00", None, 4),
        ("truncated form type", b"RIFF" + struct.pack('<L', 4) + b"WAV",
         None, 8),
        ("bad RIFF id", b"RIFX" + struct.pack('<L', 12) + b"WAVE"
            + make_fmt() + make_data(b""), None, 0),
        ("bad form type",
         make_riff([make_fmt(), make_data(b"")], form=b"AVI "), None, 8),
        ("truncated subchunk header",
         b"RIFF" + struct.pack('<L', 16) + b"WAVE" + b"fmt " + b"\x10\x00",
         None, 12),
        ("missing fmt chunk",
         make_riff([b'JUNK' + struct.pack('<L', 4) + b"\x00\x00\x00\x00"]),
         None, 0),
        ("missing data chunk",
         make_riff([make_fmt(),
                    b'JUNK' + struct.pack('<L', 4) + b"\x00\x00\x00\x00"]),
         None, 0),
        ("data before fmt",
         make_riff([make_data(b"\x01\x00"), make_fmt()]), b'data', 12),
        ("fmt chunk too small",
         make_riff([b'fmt ' + struct.pack('<L', 10)
                    + struct.pack('<HHLLH', 1, 1, 8000, 16000, 2)[:10],
                    make_data(b"")]),
         b'fmt ', 12),
        ("truncated fmt body",
         (b"RIFF" + struct.pack('<L', 30) + b"WAVE"
          + b'fmt ' + struct.pack('<L', 16)
          + struct.pack('<HHLLH', 1, 1, 8000, 16000, 2)),
         b'fmt ', 34),
        ("unsupported format tag",
         make_riff([make_fmt(fmt_tag=0x0055), make_data(b"")]),
         b'fmt ', 20),
        ("zero bits per sample",
         make_riff([make_fmt(bits=0), make_data(b"")]), b'fmt ', 34),
        ("unsupported 40-bit width",
         make_riff([make_fmt(bits=40), make_data(b"")]), b'fmt ', 34),
        ("zero channels",
         make_riff([make_fmt(nchannels=0), make_data(b"")]), b'fmt ', 22),
        ("extensible truncated extension",
         make_riff([make_fmt(fmt_tag=0xFFFE, fmt_size=24),
                    make_data(b"")]), b'fmt ', 12),
        ("extensible non-PCM subformat",
         make_riff([make_fmt(
             fmt_tag=0xFFFE,
             ext_body=struct.pack('<HHL', 22, 16, 0) + b'\x02' * 16),
             make_data(b"")]), b'fmt ', 44),
        ("data chunk extends past RIFF container",
         make_riff([make_fmt(), make_data(b"\x00" * 8)], riff_size=30),
         b'data', 36),
        ("data chunk truncated by physical EOF",
         (make_riff([make_fmt(), make_data(b"\x00" * 8)])[:-4]),
         b'data', 44),
        ("unknown chunk extends past RIFF container",
         make_riff([b'JUNK' + struct.pack('<L', 8) + b'\x00' * 8],
                   riff_size=14),
         b'JUNK', 12),
        ("unknown chunk truncated at EOF (non-seekable stream)",
         (b"RIFF" + struct.pack('<L', 22) + b"WAVE"
          + b'JUNK' + struct.pack('<L', 8) + b'\x00\x00'),
         b'JUNK', 22),
        ("truncated fmt inside non-seekable stream",
         (b"RIFF" + struct.pack('<L', 30) + b"WAVE"
          + b'fmt ' + struct.pack('<L', 16)
          + struct.pack('<HHLLH', 1, 1, 8000, 16000, 2)),
         b'fmt ', 34),
    ])
    def test_malformed_raises_named_offset_error(self, sample):
        label, raw, expected_chunk, expected_offset = sample
        stream = NonSeekableReader(raw) if "non-seekable" in label else raw
        with pytest.raises(wavio.WavReadError) as exc_info:
            wavio.read(stream)
        err = exc_info.value
        # The error must carry structured chunk/offset information.
        assert err.chunk == expected_chunk, (label, err.chunk)
        assert err.offset == expected_offset, (label, err.offset)
        # ...and the message text must contain both identifiers.
        message = str(err)
        if expected_chunk is not None:
            assert repr(expected_chunk) in message, (label, message)
            assert "offset" in message, (label, message)
        else:
            assert "RIFF structure" in message, (label, message)
            assert "offset" in message, (label, message)

    def test_malformed_from_real_file(self, tmp_path):
        # The same diagnostics must be produced when reading a path name.
        raw = make_riff([make_fmt(bits=0), make_data(b"")])
        path = tmp_path / "bad.wav"
        path.write_bytes(raw)
        with pytest.raises(wavio.WavReadError) as exc_info:
            wavio.read(str(path))
        err = exc_info.value
        assert err.chunk == b'fmt '
        assert err.offset == 34
        assert repr(b'fmt ') in str(err)

    def test_malformed_is_value_error(self):
        # Backwards compatibility: callers catching ValueError still work.
        with pytest.raises(ValueError):
            wavio.read(b"not a wav file at all")

    def test_truncation_is_physical_not_declared(self, tmp_path):
        # A physically truncated data chunk must point at the data body.
        raw = make_riff([make_fmt(), make_data(b"\x01\x00\x02\x00")])[:-2]
        path = tmp_path / "trunc.wav"
        path.write_bytes(raw)
        with pytest.raises(wavio.WavReadError) as exc_info:
            wavio.read(path)
        err = exc_info.value
        assert err.chunk == b'data'
        # Body starts right after the 8-byte data chunk header.
        assert err.offset == 44
        assert "truncated" in str(err)

    def test_batch_quality_check_can_locate_every_bad_file(self, tmp_path):
        # Simulate a batch scan: good files load, bad files are reported with
        # a file name, chunk id and byte offset that pinpoint the damage.
        good = tmp_path / "good.wav"
        truncated = tmp_path / "truncated.wav"
        wide = tmp_path / "wide.wav"
        no_fmt = tmp_path / "no_fmt.wav"
        good.write_bytes(make_valid_wav())
        truncated.write_bytes(
            make_riff([make_fmt(), make_data(b"\x00" * 8)])[:-4])
        wide.write_bytes(make_riff([make_fmt(bits=40), make_data(b"")]))
        no_fmt.write_bytes(make_riff([make_data(b"\x01\x00")]))

        report = []
        for path in sorted(tmp_path.glob("*.wav")):
            try:
                wavio.read(path)
            except wavio.WavReadError as err:
                report.append((path.name, err.chunk, err.offset, str(err)))

        assert [row[0] for row in report] == ["no_fmt.wav",
                                              "truncated.wav", "wide.wav"]
        locations = {row[0]: (row[1], row[2]) for row in report}
        assert locations["truncated.wav"] == (b'data', 44)
        assert locations["wide.wav"] == (b'fmt ', 34)
        assert locations["no_fmt.wav"][0] == b'data'
        # Every report line is self-contained: name, chunk id and offset.
        for _name, chunk, offset, message in report:
            assert repr(chunk) in message
            assert str(offset) in message


# --------------------------------------------------------------------------
# Differential equivalence with the old wave-module-based read path
# --------------------------------------------------------------------------

def reference_read(raw):
    """Reproduce the old wavio.read implementation using stdlib wave."""
    with wave.open(io.BytesIO(raw)) as wav:
        rate = wav.getframerate()
        nchannels = wav.getnchannels()
        sampwidth = wav.getsampwidth()
        nframes = wav.getnframes()
        data = wav.readframes(nframes)
    array = wavio._wav2array(nchannels, sampwidth, data)
    return rate, sampwidth, array


class TestRiffEquivalence:

    @pytest.mark.parametrize('sampwidth', [1, 2, 3, 4])
    @pytest.mark.parametrize('nchannels', [1, 2])
    def test_all_sample_widths_and_channels(self, sampwidth, nchannels):
        rng = np.random.RandomState(sampwidth * 7 + nchannels)
        nframes = 13
        if sampwidth == 1:
            values = rng.randint(0, 256,
                                 size=(nframes, nchannels)).astype(np.uint8)
            payload = values.tobytes()
        else:
            bits = sampwidth * 8
            values = rng.randint(-2**(bits - 1), 2**(bits - 1),
                                 size=(nframes, nchannels)).astype('<i4')
            if sampwidth == 3:
                # Pack only the low three little-endian bytes per sample.
                payload = b''.join(
                    struct.pack('<I', int(v) & 0xFFFFFF)
                    for v in values.reshape(-1))
            else:
                payload = values.astype(f'<i{sampwidth}').tobytes()
        raw = make_valid_wav(payload, nchannels=nchannels,
                             sampwidth=sampwidth)
        self._assert_equal(raw)

    def test_24bit_sign_extension_bytes(self):
        values = np.array([[0x7FFFFF], [0x800000], [0x000001], [0xFFFFFF]],
                          dtype='<i4')
        # Pack only the low three bytes, little endian.
        payload = b''.join(struct.pack('<I', int(v) & 0xFFFFFF)
                           for v in values[:, 0])
        raw = make_valid_wav(payload, sampwidth=3)
        self._assert_equal(raw)

    def test_odd_sized_data_chunk_no_pad_byte(self):
        # stdlib wave writes an odd-sized final data chunk without a pad.
        payload = b'\x00\x01\x02'  # 3 bytes, 16-bit mono -> one frame + tail
        raw = make_riff([make_fmt(), make_data(payload)])
        self._assert_equal(raw)

    def test_odd_sized_data_chunk_with_pad_byte(self):
        # A spec-conformant file includes the word-alignment pad.
        payload = b'\x00\x01\x02'
        raw = (make_riff([make_fmt(), make_data(payload)]) + b'\x00')
        self._assert_equal(raw)

    def test_extra_unknown_chunks_are_skipped(self):
        junk = b'JUNK' + struct.pack('<L', 5) + b'hello' + b'\x00'
        # Another odd-length chunk after data to exercise trailing pad.
        tail = b'LIST' + struct.pack('<L', 3) + b'abc' + b'\x00'
        raw = make_riff([junk, make_fmt(),
                         make_data(b'\x01\x00\x02\x00'), tail])
        self._assert_equal(raw)

    def test_extensible_pcm_chunk(self):
        ext = struct.pack('<HHL', 22, 16, 0) + (
            b'\x01\x00\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x008\x9bq')
        raw = make_riff([make_fmt(fmt_tag=0xFFFE, ext_body=ext),
                         make_data(b'\x01\x00\x02\x00')])
        self._assert_equal(raw)

    def test_riff_size_larger_than_actual_payload(self):
        # Writers commonly leave a conservative RIFF size; wave tolerates it
        # and so must the new parser.
        raw = make_riff([make_fmt(), make_data(b'\x01\x00')],
                        riff_size=200)
        self._assert_equal(raw)

    def test_non_seekable_stream(self):
        raw = make_valid_wav()
        ref_rate, ref_width, ref_array = reference_read(raw)
        w = wavio.read(NonSeekableReader(raw))
        assert w.rate == ref_rate
        assert w.sampwidth == ref_width
        np.testing.assert_array_equal(w.data, ref_array)

    def test_path_input_round_trip(self, tmp_path):
        raw = make_valid_wav()
        path = tmp_path / "x.wav"
        path.write_bytes(raw)
        w = wavio.read(path)  # pathlib.Path support, beyond the old wave API
        ref_rate, ref_width, ref_array = reference_read(raw)
        assert w.rate == ref_rate
        assert w.sampwidth == ref_width
        np.testing.assert_array_equal(w.data, ref_array)

    def test_bytes_object_file_position_is_absolute(self):
        raw = b'\x00\x00' + make_valid_wav()
        stream = io.BytesIO(raw)
        stream.seek(2)
        w = wavio.read(stream)
        ref_rate, ref_width, ref_array = reference_read(raw[2:])
        assert w.rate == ref_rate
        assert w.sampwidth == ref_width
        np.testing.assert_array_equal(w.data, ref_array)

    def test_many_random_files_byte_identical(self):
        rng = np.random.RandomState(1234)
        for _ in range(50):
            sampwidth = int(rng.choice([1, 2, 3, 4]))
            nchannels = int(rng.choice([1, 2, 3]))
            nframes = int(rng.randint(0, 40))
            if sampwidth == 1:
                values = rng.randint(0, 256,
                                     size=(nframes, nchannels)).astype(np.uint8)
                payload = values.tobytes()
            else:
                bits = sampwidth * 8
                values = rng.randint(-2**(bits - 1), 2**(bits - 1),
                                     size=(nframes, nchannels)).astype('<i4')
                if sampwidth == 3:
                    payload = b''.join(
                        struct.pack('<I', int(v) & 0xFFFFFF)
                        for v in values.reshape(-1))
                else:
                    payload = values.astype(f'<i{sampwidth}').tobytes()
            chunks = [make_fmt(nchannels=nchannels, sampwidth=sampwidth),
                      make_data(payload)]
            raw = make_riff(chunks)
            self._assert_equal(raw)

    @staticmethod
    def _assert_equal(raw):
        ref_rate, ref_width, ref_array = reference_read(raw)
        w = wavio.read(io.BytesIO(raw))
        assert w.rate == ref_rate
        assert w.sampwidth == ref_width
        assert w.data.dtype == ref_array.dtype
        assert w.data.shape == ref_array.shape
        # Byte-identical audio payload, not just numerically equal.
        assert w.data.tobytes() == ref_array.tobytes()
        np.testing.assert_array_equal(w.data, ref_array)


def test_wave_still_used_for_write_path():
    # The migration is read-path only: write() must keep emitting exactly
    # the bytes produced by the standard library wave writer.
    data = np.array([-1000, 0, 1000, 2**15 - 1], dtype='<i2')
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "w.wav")
        wavio.write(path, data, 8000)
        with open(path, 'rb') as f:
            new_bytes = f.read()
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(8000)
            w.writeframes(data.tobytes())
        assert new_bytes == buf.getvalue()
