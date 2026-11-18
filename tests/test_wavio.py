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


# Fixed waveform matrix used by the resample/remix acceptance tests.
# 12 frames, 2 channels, int16.
waveform2ch = np.array([[0, 1000],
                        [100, -900],
                        [-200, 800],
                        [300, -700],
                        [-400, 600],
                        [500, -500],
                        [-600, 400],
                        [700, -300],
                        [-800, 200],
                        [900, -100],
                        [-1000, 0],
                        [1100, 100]], dtype=int16le)


def ref_resample_linear(data, rate_in, rate_out):
    # Independent reference implementation of linear-interpolation
    # resampling, written with an explicit Python loop.
    data = np.asarray(data, dtype=np.float64)
    n_in = data.shape[0]
    n_out = int(round(n_in * rate_out / rate_in))
    out = np.empty((n_out, data.shape[1]))
    for j in range(n_out):
        t = j * rate_in / rate_out  # in units of input samples
        i = min(int(t), n_in - 1)
        frac = t - i
        if i + 1 < n_in:
            out[j] = data[i] * (1 - frac) + data[i + 1] * frac
        else:
            out[j] = data[i]
    return out


class TestReadConvert:

    def test_no_params_byte_identical(self):
        # With no conversion parameters, read() must behave exactly as
        # before: the data must match the bytes in the file exactly.
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            wavio.write('w.wav', waveform2ch, 8000)
            w = wavio.read('w.wav')
            assert w.rate == 8000
            assert w.sampwidth == 2
            assert w.data.dtype == int16le
            np.testing.assert_array_equal(w.data, waveform2ch)
            with wave.open('w.wav', 'r') as f:
                raw = f.readframes(f.getnframes())
            assert w.data.tobytes() == raw

    def test_resample_upsample(self):
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            wavio.write('w.wav', waveform2ch, 8000)
            w = wavio.read('w.wav', rate=16000)
            assert w.rate == 16000
            assert w.sampwidth == 2
            assert w.data.dtype == np.float64
            assert w.data.shape == (24, 2)
            expected = ref_resample_linear(waveform2ch, 8000, 16000)
            np.testing.assert_allclose(w.data, expected, rtol=1e-12,
                                       atol=1e-9)

    def test_resample_downsample(self):
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            wavio.write('w.wav', waveform2ch, 8000)
            w = wavio.read('w.wav', rate=3000)
            assert w.rate == 3000
            # round(12 * 3000/8000) = round(4.5) = 4 (banker's rounding)
            assert w.data.shape == (4, 2)
            expected = ref_resample_linear(waveform2ch, 8000, 3000)
            np.testing.assert_allclose(w.data, expected, rtol=1e-12,
                                       atol=1e-9)

    def test_resample_same_rate_is_noop_values(self):
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            wavio.write('w.wav', waveform2ch, 8000)
            w = wavio.read('w.wav', rate=8000)
            assert w.rate == 8000
            assert w.data.dtype == np.float64
            np.testing.assert_array_equal(w.data,
                                          waveform2ch.astype(np.float64))

    def test_remix_mono_to_stereo(self):
        mono = waveform2ch[:, :1]
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            wavio.write('w.wav', mono, 8000)
            w = wavio.read('w.wav', nchannels=2)
            assert w.data.shape == (len(mono), 2)
            expected = np.repeat(mono.astype(np.float64), 2, axis=1)
            np.testing.assert_array_equal(w.data, expected)

    def test_remix_stereo_to_mono(self):
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            wavio.write('w.wav', waveform2ch, 8000)
            w = wavio.read('w.wav', nchannels=1)
            assert w.data.shape == (len(waveform2ch), 1)
            expected = waveform2ch.astype(np.float64).mean(axis=1,
                                                           keepdims=True)
            np.testing.assert_allclose(w.data, expected, rtol=1e-12,
                                       atol=1e-9)

    def test_remix_quad_to_stereo(self):
        quad = np.column_stack([waveform2ch[:, 0], waveform2ch[:, 1],
                                2 * waveform2ch[:, 0], -waveform2ch[:, 1]])
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            wavio.write('w.wav', quad.astype(int16le), 8000)
            w = wavio.read('w.wav', nchannels=2)
            assert w.data.shape == (len(quad), 2)
            # Fixed strategy: channels (0, 1) averaged, channels (2, 3)
            # averaged.
            expected = np.column_stack([quad[:, :2].mean(axis=1),
                                        quad[:, 2:].mean(axis=1)])
            np.testing.assert_allclose(w.data, expected, rtol=1e-12,
                                       atol=1e-9)

    def test_remix_stereo_to_triple(self):
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            wavio.write('w.wav', waveform2ch, 8000)
            w = wavio.read('w.wav', nchannels=3)
            assert w.data.shape == (len(waveform2ch), 3)
            # Fixed strategy: cyclic repetition of channels.
            expected = np.column_stack([waveform2ch[:, 0],
                                        waveform2ch[:, 1],
                                        waveform2ch[:, 0]]).astype(np.float64)
            np.testing.assert_array_equal(w.data, expected)

    def test_resample_and_remix(self):
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            wavio.write('w.wav', waveform2ch, 8000)
            w = wavio.read('w.wav', rate=16000, nchannels=1)
            assert w.rate == 16000
            assert w.data.shape == (24, 1)
            expected = ref_resample_linear(waveform2ch, 8000, 16000)
            expected = expected.mean(axis=1, keepdims=True)
            np.testing.assert_allclose(w.data, expected, rtol=1e-12,
                                       atol=1e-9)

    @pytest.mark.parametrize('rate', [0, -1, 2.5])
    def test_invalid_rate(self, rate):
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            wavio.write('w.wav', waveform2ch, 8000)
            with pytest.raises(ValueError):
                wavio.read('w.wav', rate=rate)

    @pytest.mark.parametrize('nchannels', [0, -2, 1.5])
    def test_invalid_nchannels(self, nchannels):
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            wavio.write('w.wav', waveform2ch, 8000)
            with pytest.raises(ValueError):
                wavio.read('w.wav', nchannels=nchannels)
