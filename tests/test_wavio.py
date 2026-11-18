import pytest
import tempfile
import os
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


class TestChunked:

    @pytest.mark.parametrize('sampwidth,dtype',
                             [(1, np.uint8),
                              (2, int16le),
                              (3, int32le),
                              (4, int32le)])
    @pytest.mark.parametrize('nchannels', [1, 2])
    def test_read_chunks_matches_read(self, sampwidth, dtype, nchannels):
        nframes = 37
        rng = np.random.default_rng(0)
        info = np.iinfo(dtype)
        data = rng.integers(info.min, info.max,
                            size=(nframes, nchannels), dtype=dtype)
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            wavio.write('full.wav', data, rate=8000, sampwidth=sampwidth)

            w = wavio.read('full.wav')
            # Chunk sizes chosen so that the data does not divide evenly:
            # the last chunk must be the short one.
            for chunk_size in [1, 7, 100]:
                chunks = list(wavio.read_chunks('full.wav', chunk_size))
                assert all(c.ndim == 2 for c in chunks)
                assert all(c.shape[1] == nchannels for c in chunks)
                assert all(c.dtype == w.data.dtype for c in chunks)
                assert sum(c.shape[0] for c in chunks) == nframes
                np.testing.assert_array_equal(np.concatenate(chunks),
                                              w.data)

    def test_read_chunks_24bit_sign_extension(self):
        # Values spanning the sign boundary must be sign-extended in
        # every chunk, including chunks that begin on an arbitrary frame.
        data = np.array([-2**23, -1, 0, 1, 2**23 - 1,
                         -12345678, 12345678, -2**23 + 1],
                        dtype=int32le)
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            wavio.write('signext.wav', data, rate=8000, sampwidth=3)
            w = wavio.read('signext.wav')
            for chunk_size in [1, 2, 3, 5]:
                chunks = list(wavio.read_chunks('signext.wav', chunk_size))
                np.testing.assert_array_equal(np.concatenate(chunks),
                                              w.data)

    def test_read_chunks_bad_chunk_size(self):
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            wavio.write('x.wav', data1, rate=8000, sampwidth=3)
            with pytest.raises(ValueError):
                list(wavio.read_chunks('x.wav', chunk_size=0))
            with pytest.raises(ValueError):
                list(wavio.read_chunks('x.wav', chunk_size=-3))

    @pytest.mark.parametrize('sampwidth', [1, 2, 3, 4])
    @pytest.mark.parametrize('nchannels', [1, 2])
    def test_write_chunks_integer_matches_write(self, sampwidth, nchannels):
        nframes = 41
        rng = np.random.default_rng(1)
        dtype = {1: np.uint8, 2: np.int16, 3: np.int32,
                 4: np.int32}[sampwidth]
        info = np.iinfo(dtype)
        data = rng.integers(info.min, info.max,
                            size=(nframes, nchannels), dtype=dtype)
        chunk_sizes = [1, 3, 8, 1000]
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            wavio.write('whole.wav', data, rate=11025, sampwidth=sampwidth)
            whole_bytes = open('whole.wav', 'rb').read()
            for chunk_size in chunk_sizes:
                name = f'chunked_{sampwidth}_{nchannels}_{chunk_size}.wav'
                chunks = (data[i:i + chunk_size]
                          for i in range(0, nframes, chunk_size))
                wavio.write_chunks(chunks, name, rate=11025,
                                   sampwidth=sampwidth)
                assert open(name, 'rb').read() == whole_bytes

    @pytest.mark.parametrize('sampwidth', [1, 2, 3, 4])
    def test_write_chunks_float_matches_write(self, sampwidth):
        nframes = 53
        rng = np.random.default_rng(2)
        data = rng.normal(scale=0.7, size=(nframes, 2)).astype(np.float64)
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            wavio.write('whole.wav', data, rate=22050, sampwidth=sampwidth,
                        scale=2.5, clip='ignore')
            whole_bytes = open('whole.wav', 'rb').read()
            for chunk_size in [1, 4, 9, 1000]:
                name = f'chunked_f_{sampwidth}_{chunk_size}.wav'
                chunks = [data[i:i + chunk_size]
                          for i in range(0, nframes, chunk_size)]
                wavio.write_chunks(chunks, name, rate=22050,
                                   sampwidth=sampwidth, scale=2.5,
                                   clip='ignore')
                assert open(name, 'rb').read() == whole_bytes

    def test_write_chunks_one_dimensional_input(self):
        data = data1.copy()
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            wavio.write('whole.wav', data, rate=8000, sampwidth=3)
            whole_bytes = open('whole.wav', 'rb').read()
            chunks = [data[i:i + 4] for i in range(0, len(data), 4)]
            wavio.write_chunks(chunks, 'chunked.wav', rate=8000,
                               sampwidth=3)
            assert open('chunked.wav', 'rb').read() == whole_bytes

    def test_wavwriter_round_trip_via_read_chunks(self):
        rng = np.random.default_rng(3)
        data = rng.integers(-2**23, 2**23, size=(100, 2), dtype=int32le)
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            with wavio.WavWriter('stream.wav', rate=48000, sampwidth=3,
                                 nchannels=2) as writer:
                for i in range(0, len(data), 17):
                    writer.write_chunk(data[i:i + 17])
            w = wavio.read('stream.wav')
            np.testing.assert_array_equal(w.data, data)
            assert w.rate == 48000
            assert w.sampwidth == 3

    def test_read_write_chunks_pipe_all_widths(self):
        # A full read_chunks -> write_chunks pipe reproduces the source
        # file byte for byte, including 24 bit data (which is decoded to
        # int32 and therefore needs sampwidth passed explicitly).
        rng = np.random.default_rng(4)
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            for sampwidth, dtype in [(1, np.uint8), (2, np.int16),
                                     (3, np.int32), (4, np.int32)]:
                info = np.iinfo(dtype)
                data = rng.integers(info.min, info.max, size=(77, 2),
                                    dtype=dtype)
                wavio.write('src.wav', data, rate=12000,
                            sampwidth=sampwidth)
                original = open('src.wav', 'rb').read()
                wavio.write_chunks(
                    wavio.read_chunks('src.wav', chunk_size=13),
                    'copy.wav', rate=12000, sampwidth=sampwidth)
                assert open('copy.wav', 'rb').read() == original

    def test_write_still_supports_scale_auto(self):
        # The streaming interface rejects "auto", but the whole-array
        # write() must keep supporting it unchanged.
        data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            wavio.write('auto.wav', data, rate=100, sampwidth=2,
                        scale="auto")
            w = wavio.read('auto.wav')
            # The auto scale maps the extrema to the reachable limits;
            # the negative minimum is -2**15 + 1 by wavio's convention.
            assert w.data.min() == -2**15 + 1
            assert w.data.max() == 2**15 - 1

    def test_wavwriter_empty_file_is_valid(self):
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            with wavio.WavWriter('empty.wav', rate=8000, sampwidth=2,
                                 nchannels=1):
                pass
            w = wavio.read('empty.wav')
            assert w.data.shape == (0, 1)
            assert w.rate == 8000

    def test_wavwriter_channel_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            with wavio.WavWriter('x.wav', rate=8000, sampwidth=2,
                                 nchannels=2) as writer:
                with pytest.raises(ValueError):
                    writer.write_chunk(np.zeros((3, 1), dtype=np.int16))

    def test_wavwriter_mixed_kinds_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            with wavio.WavWriter('x.wav', rate=8000, sampwidth=2,
                                 nchannels=1) as writer:
                writer.write_chunk(np.zeros(3, dtype=np.int16))
                with pytest.raises(ValueError):
                    writer.write_chunk(np.zeros(3, dtype=np.float32))

    def test_wavwriter_integer_scale_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            with wavio.WavWriter('x.wav', rate=8000, sampwidth=2,
                                 nchannels=1, scale=2.0) as writer:
                with pytest.raises(ValueError):
                    writer.write_chunk(np.zeros(3, dtype=np.int16))

    def test_wavwriter_auto_scale_rejected(self):
        with pytest.raises(ValueError):
            wavio.WavWriter('x.wav', rate=8000, sampwidth=2, nchannels=1,
                            scale="auto")
        with pytest.raises(ValueError):
            wavio.write_chunks(iter([np.zeros(3)]), 'x.wav', rate=8000,
                               sampwidth=2, scale="auto")

    def test_write_chunks_empty_iterable(self):
        with pytest.raises(ValueError):
            wavio.write_chunks(iter([]), 'x.wav', rate=8000, sampwidth=2)

    def test_wavwriter_clip_warn_aggregated(self):
        data = np.array([0, 100, 200, 300, 400], dtype=np.int32)
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            with pytest.warns(wavio.ClippedDataWarning) as record:
                with wavio.WavWriter('x.wav', rate=100, sampwidth=1,
                                     nchannels=1) as writer:
                    for i in range(0, len(data), 2):
                        writer.write_chunk(data[i:i + 2])
            # Exactly one aggregated warning, even though several
            # chunks contain clipped values.
            assert len(record) == 1

    def test_wavwriter_clip_raise(self):
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            with pytest.raises(wavio.ClippedDataError):
                with wavio.WavWriter('x.wav', rate=100, sampwidth=1,
                                     nchannels=1, clip='raise') as writer:
                    writer.write_chunk(np.array([0, 300], dtype=np.int32))

    def test_wavwriter_clip_ignore(self):
        data = np.array([0, 100, 200, 300], dtype=np.int32)
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                with wavio.WavWriter('x.wav', rate=100, sampwidth=1,
                                     nchannels=1, clip='ignore') as writer:
                    for i in range(0, len(data), 2):
                        writer.write_chunk(data[i:i + 2])
            w = wavio.read('x.wav')
            np.testing.assert_array_equal(
                w.data[:, 0], np.array([0, 100, 200, 255], dtype=np.uint8))

    def test_float_clip_semantics_match_in_chunks(self):
        # Floating point clipping detection and the detailed message
        # must be the same as on the whole-array path.
        data = np.array([0.5, 0.25, 0.0, -0.9, -1.5, 0.0])
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            wavio.write('whole.wav', data, rate=8000, sampwidth=2,
                        scale=1.0, clip='ignore')
            whole_bytes = open('whole.wav', 'rb').read()

            chunks = [data[i:i + 2] for i in range(0, len(data), 2)]
            with pytest.warns(wavio.ClippedDataWarning):
                wavio.write_chunks(chunks, 'chunked.wav', rate=8000,
                                   sampwidth=2, scale=1.0)
            assert open('chunked.wav', 'rb').read() == whole_bytes
