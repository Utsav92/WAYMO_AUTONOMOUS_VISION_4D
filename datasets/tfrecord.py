"""Minimal TFRecord reader (no TensorFlow).  Record = u64 length, u32 masked-crc(length), data, u32 masked-crc(data).
CRCs are not verified (files come from a checksummed download)."""
import struct


def iter_records(path, limit=None, skip=0):
    n = 0
    with open(path, "rb") as f:
        while True:
            head = f.read(12)
            if len(head) < 12:
                return
            (length,) = struct.unpack("<Q", head[:8])
            if n < skip:
                f.seek(length + 4, 1)
            else:
                data = f.read(length)
                f.read(4)
                yield data
            n += 1
            if limit is not None and n >= skip + limit:
                return
