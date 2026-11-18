"""
Generate the test fixture ``cue_points.wav`` (in this directory).

The file is a standard PCM WAV file with extended RIFF chunks, laid out
as follows (chunks in file order)::

    RIFF/WAVE
    fmt   : 16 byte PCM format chunk, mono, 8000 Hz, 16 bit
    Xtra  : unknown chunk, 3 byte payload b'\\xaa\\xbb\\xcc'
            (odd size, so a pad byte follows)
    cue   : two cue points: id=1 at sample offset 0,
            id=2 at sample offset 1234
    data  : 4000 frames of 16 bit mono samples (8000 bytes)
    LIST  : 'INFO' list with INAM 'Cue Test' (odd-sized subchunk, padded
            inside the LIST payload) and ISFT 'wavio'

Run ``python make_cue_points_wav.py`` to regenerate the fixture.
"""

import os
import struct


def chunk(cid, payload):
    out = cid + struct.pack('<I', len(payload)) + payload
    if len(payload) % 2:
        out += b'\x00'
    return out


rate = 8000
nframes = 4000
samples = [((i * 137) % 4001) - 2000 for i in range(nframes)]
data = struct.pack(f'<{nframes}h', *samples)

fmt = struct.pack('<HHIIHH', 1, 1, rate, rate * 2, 2, 16)

xtra = b'\xaa\xbb\xcc'

cue = struct.pack('<I', 2)  # two cue points
cue += struct.pack('<II4sIII', 1, 0, b'data', 0, 0, 0)
cue += struct.pack('<II4sIII', 2, 1234, b'data', 0, 0, 1234)

inam = b'Cue Test\x00'  # 9 bytes (odd), so a pad byte follows inside LIST
isft = b'wavio\x00'
list_payload = b'INFO' + chunk(b'INAM', inam) + chunk(b'ISFT', isft)

body = (b'WAVE'
        + chunk(b'fmt ', fmt)
        + chunk(b'Xtra', xtra)
        + chunk(b'cue ', cue)
        + chunk(b'data', data)
        + chunk(b'LIST', list_payload))
riff = b'RIFF' + struct.pack('<I', len(body)) + body

here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, 'cue_points.wav'), 'wb') as f:
    f.write(riff)
