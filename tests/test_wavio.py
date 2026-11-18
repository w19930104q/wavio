import pytest
import tempfile
import os
import io
import struct
import wave
import numpy as np
import wavio


int32le = np.dtype('<i4')
int16le = np.dtype('<i2')

data1 = np.array([1, -2,
                  3, -4,
                  2**16, -2**16,
                  -2**20, 2**20,
                  2**23-1, -2**23],
                 dtype=int32le)


def check_basic(filename, nchannels, sampwidth, framerate):
    with wave.open(filename, 'r') as f:
        assert f.getnchannels() == nchannels, "unexpected nchannels"
        assert f.getsampwidth() == sampwidth, "unexpected sampwidth"
        assert f.getframerate() == framerate, "unexpected framerate"


def check_wavio_read(filename, rate, sampwidth, dtype, shape, data):
    if data.ndim == 1:
        # wavio.read always returns a 2-d array.
        data = data[:, None]
    w = wavio.read(filename)
    assert w.rate == rate, "unexpected rate"
    assert w.sampwidth == sampwidth, "unexpected sampwidth"
    assert w.data.dtype == dtype, "unexpected dtype"
    assert w.data.shape == shape, "unexpected shape"
    np.testing.assert_equal(w.data, data, "data does not match")


class TestWavio:

    def test0(self):
        filename = 'test1data.wav'
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            wavio.write(filename, data1, 44100, sampwidth=3)

            check_basic(filename, nchannels=1, sampwidth=3,
                        framerate=44100)
            check_wavio_read(filename, rate=44100, sampwidth=3,
                             dtype=int32le, shape=(len(data1), 1),
                             data=data1)

    @pytest.mark.parametrize('data', [data1[:, None],
                                      data1.reshape(-1, 2)])
    def test1(self, data):
        filename = 'test1data.wav'
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            wavio.write(filename, data, 44100, sampwidth=3)

            check_basic(filename, nchannels=data.shape[1], sampwidth=3,
                        framerate=44100)
            check_wavio_read(filename, rate=44100, sampwidth=3,
                             dtype=int32le, shape=data.shape,
                             data=data)

    def test2(self):
        filename = 'test2data.wav'
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            data2 = data1.reshape(-1, 2)
            wavio.write(filename, data2, 44100, sampwidth=3)

            check_basic(filename, nchannels=2, sampwidth=3, framerate=44100)
            check_wavio_read(filename, rate=44100, sampwidth=3,
                             dtype=int32le, shape=data2.shape,
                             data=data2)

    def test3(self):
        filename = 'test3data.wav'
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            data = np.zeros(32, dtype=int16le)
            data[1::4] = 10000
            data[3::4] = -10000
            wavio.write(filename, data, 44100)

            check_basic(filename, nchannels=1, sampwidth=2, framerate=44100)
            check_wavio_read(filename, rate=44100, sampwidth=2,
                             dtype=int16le, shape=(32, 1),
                             data=data)

    def test5(self):
        filename = 'test5data.wav'
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            data = np.zeros(32, dtype=int16le)
            data[1::4] = 10000
            data[3::4] = -10000
            wavio.write(filename, data, 44100, sampwidth=2)

            check_basic(filename, nchannels=1, sampwidth=2, framerate=44100)
            check_wavio_read(filename, rate=44100, sampwidth=2,
                             dtype=int16le, shape=(32, 1),
                             data=data.reshape(-1, 1))

    def test6(self):
        filename = 'test6data.wav'
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            data = np.zeros(32, dtype=int16le)
            data[1::4] = 10000
            data[3::4] = -10000
            wavio.write(filename, data, 44100, sampwidth=1, clip='ignore')

            check_basic(filename, nchannels=1, sampwidth=1, framerate=44100)
            expected = np.zeros(data.shape, dtype=np.uint8).reshape(-1, 1)
            expected[1::4, 0] = 255
            check_wavio_read(filename, rate=44100, sampwidth=1,
                             dtype=np.uint8, shape=(32, 1),
                             data=expected)

    def test_clip(self):
        filename = 'testdata.wav'
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            data = np.array([-100, 0, 100, 200, 300, 325])
            wavio.write(filename, data, 44100, sampwidth=1, clip='ignore')

            check_basic(filename, nchannels=1, sampwidth=1, framerate=44100)
            expected = np.array([0, 0, 100, 200, 255, 255],
                                dtype=np.uint8)
            check_wavio_read(filename, rate=44100, sampwidth=1,
                             dtype=np.uint8, shape=(len(data), 1),
                             data=expected)

    def test_signed8bit_full_range_round_trip(self):
        filename = 'testdata.wav'
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            # Negative values will be clipped.
            data = np.array([-2**7, -2**3, -1, 0, 1, 2**3, 2**7-1],
                            dtype=np.int8)
            wavio.write(filename, data, 44100, clip='ignore')

            check_basic(filename, nchannels=1, sampwidth=1, framerate=44100)
            int8max = np.iinfo(np.int8).max  # A long way to spell 127.
            expected_data = data.clip(0, int8max).astype(np.uint8)
            check_wavio_read(filename, rate=44100, sampwidth=1,
                             dtype=np.uint8, shape=(len(data), 1),
                             data=expected_data)

    def test_unsigned8bit_full_range_round_trip(self):
        # Regression test for github issue 5.
        filename = 'testdata.wav'
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            data = np.array([[0, 1], [2**4-1, 2**4], [2**8-2, 2**8-1]],
                            dtype=np.uint8)
            wavio.write(filename, data, 44100)

            check_basic(filename, nchannels=2, sampwidth=1,
                        framerate=44100)
            check_wavio_read(filename, rate=44100, sampwidth=1,
                             dtype=np.uint8, shape=data.shape,
                             data=data)

    def test_16bit_full_range_round_trip(self):
        # Regression test for github issue 5.
        filename = 'testdata.wav'
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            data = np.array([-2**15, -2**8, -1, 0, 1, 2**8, 2**15-1],
                            dtype=int16le)
            wavio.write(filename, data, 44100)

            check_basic(filename, nchannels=1, sampwidth=2, framerate=44100)
            check_wavio_read(filename, rate=44100, sampwidth=2,
                             dtype=int16le, shape=(len(data), 1),
                             data=data.reshape(-1, 1))

    def test_24bit_full_range_round_trip(self):
        # Regression test for github issue 5.
        filename = 'testdata.wav'
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            data = np.array([-2**23, -2**16, -1, 0, 1, 2**16, 2**23-1],
                            dtype=int32le)
            wavio.write(filename, data, 44100, sampwidth=3)

            check_basic(filename, nchannels=1, sampwidth=3, framerate=44100)
            check_wavio_read(filename, rate=44100, sampwidth=3,
                             dtype=int32le, shape=(len(data), 1),
                             data=data.reshape(-1, 1))

    def test_32bit_full_range_round_trip(self):
        # Regression test for github issue 5.
        filename = 'testdata.wav'
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            data = np.array([-2**31, -2**16, -1, 0, 1, 2**16, 2**31-1],
                            dtype=int32le)
            wavio.write(filename, data, 44100)

            check_basic(filename, nchannels=1, sampwidth=4, framerate=44100)
            check_wavio_read(filename, rate=44100, sampwidth=4,
                             dtype=int32le, shape=(len(data), 1),
                             data=data.reshape(-1, 1))

    def test_clipping_exception_float(self):
        data = np.array([0.5, 0.25, 0, -0.9, -1.5, 0])
        with pytest.raises(wavio.ClippedDataError):
            # The error will be detected before the function attempts to
            # create the output file, so we can pass in the filename knowing
            # that it won't actually try to create that file.
            wavio.write("foo.wav", data, rate=8000, sampwidth=2, scale=1.0,
                        clip='raise')

    def test_clipping_exception_int(self):
        data = np.array([0, 100, 200, 300, 400])
        with pytest.raises(wavio.ClippedDataError):
            # The error will be detected before the function attempts to
            # create the output file, so we can pass in the filename knowing
            # that it won't actually try to create that file.
            wavio.write("foo.wav", data, rate=100, sampwidth=1, clip='raise')

    @pytest.mark.parametrize('sampwidth, low, mid, high',
                             [(1, 1, 128, 255),
                              (2, -2**15+1, 0, 2**15-1),
                              (3, -2**23+1, 0, 2**23-1),
                              (4, -2**31+1, 0, 2**31-1)])
    def test_scale_limits(self, sampwidth, low, mid, high):
        data = np.array([-1.0, 0.0, 1.0])
        filename = 'testdata.wav'
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            wavio.write(filename, data, rate=22050, scale=1.0,
                        sampwidth=sampwidth, clip='raise')
            w = wavio.read(filename)
            np.testing.assert_equal(w.data, np.array([[low], [mid], [high]]))

    @pytest.mark.parametrize('scale', [None, 1, "auto"])
    def test_float_extra_negative(self, scale):
        # `scale=1.0` is used; the two values here that are less than 1
        # will be mapped to 0, while -1 is mapped to 1.
        data = np.array([-1.0075, -1.005, -1.0, 0, 1.0])
        filename = 'testdata.wav'
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            wavio.write(filename, data, rate=22050, scale=scale,
                        sampwidth=1, clip='raise')
            w = wavio.read(filename)
            np.testing.assert_equal(w.data, np.array([[0, 0, 1, 128, 255]]).T)

    def test_float_scale_auto(self):
        d = 1/(2**23 - 0.5)
        data = 4*np.array([-1.0 - 0.96*d, -1.0 - 0.01*d,
                           -1.0, -1.0 + 0.75*d, -1.0 + 1.25*d,
                           -2*d, -d, -0.25*d, 0, 0.25*d, d, 2*d,
                           1.0 - 1.25*d, 1.0 - 0.75*d, 1.0])
        filename = 'testdata.wav'
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            wavio.write(filename, data, rate=10, scale="auto", sampwidth=3)
            w = wavio.read(filename)
            expected = np.array([-2**23, -2**23,
                                 -2**23 + 1, -2**23 + 1, -2**23 + 2,
                                 -2, -1, 0, 0, 0, 1, 2,
                                 2**23 - 2, 2**23 - 1, 2**23 - 1])
            actual = w.data[:, 0]
            np.testing.assert_array_equal(actual, expected)


# Fixture generated by tests/make_cue_points_wav.py.  Chunk layout:
#   fmt (PCM, mono, 8000 Hz, 16 bit), Xtra (unknown, odd size),
#   cue (two cue points at sample offsets 0 and 1234), data, LIST (INFO).
CUE_FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'cue_points.wav')


def parse_cue_offsets(cue_data):
    """Extract the dwSampleOffset field of each cue point."""
    npoints = struct.unpack('<I', cue_data[:4])[0]
    return [struct.unpack('<I', cue_data[4 + 24*i + 20:4 + 24*i + 24])[0]
            for i in range(npoints)]


class TestChunks:

    def test_read_exposes_chunks_in_order(self):
        w = wavio.read(CUE_FIXTURE)
        assert [c.id for c in w.chunks] == [b'fmt ', b'Xtra', b'cue ',
                                            b'data', b'LIST']
        # The basic attributes are unaffected by the extra chunks.
        assert w.rate == 8000
        assert w.sampwidth == 2
        assert w.data.shape == (4000, 1)
        assert w.data.dtype == int16le

    def test_read_from_file_object(self):
        with open(CUE_FIXTURE, 'rb') as f:
            blob = f.read()
        w = wavio.read(io.BytesIO(blob))
        assert [c.id for c in w.chunks] == [b'fmt ', b'Xtra', b'cue ',
                                            b'data', b'LIST']

    def test_cue_points_and_info_exposed(self):
        w = wavio.read(CUE_FIXTURE)
        cue = next(c for c in w.chunks if c.id == b'cue ')
        assert parse_cue_offsets(cue.data) == [0, 1234]
        list_chunk = next(c for c in w.chunks if c.id == b'LIST')
        assert list_chunk.data.startswith(b'INFO')
        assert b'INAM' in list_chunk.data
        assert b'Cue Test\x00' in list_chunk.data

    def test_round_trip_byte_identical(self):
        # Reading the fixture and writing it back with its chunks must
        # reproduce the file byte for byte.
        with open(CUE_FIXTURE, 'rb') as f:
            original = f.read()
        w = wavio.read(io.BytesIO(original))
        with tempfile.TemporaryDirectory() as d:
            outname = os.path.join(d, 'out.wav')
            wavio.write(outname, w.data, w.rate, sampwidth=w.sampwidth,
                        chunks=w.chunks)
            with open(outname, 'rb') as f:
                result = f.read()
        assert result == original

    def test_round_trip_preserves_cue_offsets(self):
        # The cue sample offsets are relative to the start of the data,
        # so they must survive the recomputation of the file layout.
        w = wavio.read(CUE_FIXTURE)
        with tempfile.TemporaryDirectory() as d:
            outname = os.path.join(d, 'out.wav')
            wavio.write(outname, w.data, w.rate, sampwidth=w.sampwidth,
                        chunks=w.chunks)
            w2 = wavio.read(outname)
        cue = next(c for c in w2.chunks if c.id == b'cue ')
        assert parse_cue_offsets(cue.data) == [0, 1234]

    def test_unknown_chunk_pass_through(self):
        w = wavio.read(CUE_FIXTURE)
        xtra = next(c for c in w.chunks if c.id == b'Xtra')
        assert xtra.data == b'\xaa\xbb\xcc'
        with tempfile.TemporaryDirectory() as d:
            outname = os.path.join(d, 'out.wav')
            wavio.write(outname, w.data, w.rate, sampwidth=w.sampwidth,
                        chunks=w.chunks)
            w2 = wavio.read(outname)
        xtra2 = next(c for c in w2.chunks if c.id == b'Xtra')
        assert xtra2.data == b'\xaa\xbb\xcc'

    def test_write_merges_chunks_and_recomputes_layout(self):
        # Chunks given without 'fmt '/'data' markers: 'fmt ' goes first,
        # 'data' goes last, and all sizes/offsets are recomputed.
        cue_data = struct.pack('<I', 1) + struct.pack('<II4sIII', 7, 100,
                                                      b'data', 0, 0, 100)
        chunks = [wavio.Chunk(id=b'cue ', data=cue_data),
                  wavio.Chunk(id=b'Xtra', data=b'\x01\x02\x03')]
        data = np.zeros(100, dtype=int16le)
        with tempfile.TemporaryDirectory() as d:
            outname = os.path.join(d, 'out.wav')
            wavio.write(outname, data, 8000, chunks=chunks)
            check_basic(outname, nchannels=1, sampwidth=2, framerate=8000)
            with open(outname, 'rb') as f:
                blob = f.read()
            w = wavio.read(outname)
        assert [c.id for c in w.chunks] == [b'fmt ', b'cue ', b'Xtra',
                                            b'data']
        # The RIFF size field must agree with the actual file size.
        riff_size = struct.unpack('<I', blob[4:8])[0]
        assert riff_size == len(blob) - 8
        # The odd-sized unknown chunk got a pad byte.
        assert len(blob) % 2 == 0
        cue = next(c for c in w.chunks if c.id == b'cue ')
        assert parse_cue_offsets(cue.data) == [100]

    def test_default_write_has_no_extra_chunks(self):
        # The default output (chunks not given) is unchanged: only the
        # 'fmt ' and 'data' chunks are written.
        data = np.zeros(64, dtype=int16le)
        with tempfile.TemporaryDirectory() as d:
            outname = os.path.join(d, 'out.wav')
            wavio.write(outname, data, 44100)
            w = wavio.read(outname)
        assert [c.id for c in w.chunks] == [b'fmt ', b'data']

    def test_plain_file_round_trip_byte_identical(self):
        # A plain file written by wavio round-trips byte-for-byte when
        # its chunks are passed back to write.
        data = np.arange(100, dtype=int16le) - 50
        with tempfile.TemporaryDirectory() as d:
            name1 = os.path.join(d, 'a.wav')
            name2 = os.path.join(d, 'b.wav')
            wavio.write(name1, data, 44100)
            w = wavio.read(name1)
            wavio.write(name2, w.data, w.rate, sampwidth=w.sampwidth,
                        chunks=w.chunks)
            with open(name1, 'rb') as f:
                blob1 = f.read()
            with open(name2, 'rb') as f:
                blob2 = f.read()
        assert blob1 == blob2

    def test_wav_chunks_default_is_empty(self):
        w = wavio.Wav(data=np.zeros((4, 1), dtype=int16le), rate=8000,
                      sampwidth=2)
        assert w.chunks == ()

    def test_duplicate_chunk_read_error(self):
        with open(CUE_FIXTURE, 'rb') as f:
            blob = f.read()
        # Append a second 'fmt ' chunk.
        fmt = struct.pack('<HHIIHH', 1, 1, 8000, 16000, 2, 16)
        bad = blob + b'fmt ' + struct.pack('<I', len(fmt)) + fmt
        with pytest.raises(wavio.WavChunkError, match='duplicate'):
            wavio.read(io.BytesIO(bad))
        # WavChunkError is a subclass of wave.Error.
        with pytest.raises(wave.Error):
            wavio.read(io.BytesIO(bad))

    def test_duplicate_chunk_write_error(self):
        chunks = [wavio.Chunk(id=b'cue ', data=b'\x00' * 4),
                  wavio.Chunk(id=b'cue ', data=b'\x00' * 4)]
        data = np.zeros(16, dtype=int16le)
        with tempfile.TemporaryDirectory() as d:
            outname = os.path.join(d, 'out.wav')
            with pytest.raises(wavio.WavChunkError, match='duplicate'):
                wavio.write(outname, data, 8000, chunks=chunks)

    def test_truncated_chunk_error(self):
        with open(CUE_FIXTURE, 'rb') as f:
            blob = f.read()
        # Cut the file in the middle of the final LIST chunk.
        with pytest.raises(wavio.WavChunkError, match='corrupt'):
            wavio.read(io.BytesIO(blob[:-10]))
        # Truncated chunk header.
        with pytest.raises(wavio.WavChunkError, match='corrupt'):
            wavio.read(io.BytesIO(blob[:20]))

    def test_not_a_riff_file_error(self):
        with pytest.raises(wavio.WavChunkError, match='not a RIFF file'):
            wavio.read(io.BytesIO(b'RIFX' + b'\x00' * 40))
        with pytest.raises(wavio.WavChunkError, match='not a WAVE file'):
            wavio.read(io.BytesIO(b'RIFF' + b'\x24\x00\x00\x00'
                                  + b'AVI ' + b'\x00' * 32))
        with pytest.raises(wavio.WavChunkError, match='too short'):
            wavio.read(io.BytesIO(b'RIFF'))

    def test_chunk_id_must_be_four_bytes(self):
        with pytest.raises(ValueError, match='4 bytes'):
            wavio.Chunk(id=b'cue', data=b'')
