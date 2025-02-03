import time
from collections import defaultdict
from collections.abc import Buffer
from typing import Optional

from numcodecs.abc import Codec

from .abc import CodecObserver


class WalltimeObserver(CodecObserver):
    last_encode: Optional[tuple[int, float]]
    last_decode: Optional[tuple[int, float]]

    codecs: dict[int, Codec]
    encode_times: defaultdict[int, list[float]]
    decode_times: defaultdict[int, list[float]]

    def __init__(self):
        self.last_encode = None
        self.last_decode = None

        self.codecs = dict()
        self.encode_times = defaultdict(list)
        self.decode_times = defaultdict(list)

    def pre_encode(self, codec: Codec, buf: Buffer) -> None:
        assert self.last_encode is None
        assert self.last_decode is None

        self.last_encode = (id(codec), time.perf_counter())

    def post_encode(self, codec: Codec, buf: Buffer, encoded: Buffer) -> None:
        assert self.last_encode is not None
        last_encode_codec, last_encode_start = self.last_encode
        assert last_encode_codec == id(codec)
        assert self.last_decode is None

        self.encode_times[id(codec)].append(time.perf_counter() - last_encode_start)
        self.codecs[id(codec)] = codec

        self.last_encode = None

    def pre_decode(
        self, codec: Codec, buf: Buffer, out: Optional[Buffer] = None
    ) -> None:
        assert self.last_encode is None
        assert self.last_decode is None

        self.last_decode = (id(codec), time.perf_counter())

    def post_decode(self, codec: Codec, buf: Buffer, decoded: Buffer) -> None:
        assert self.last_encode is None
        assert self.last_decode is not None
        last_decode_codec, last_decode_start = self.last_decode
        assert last_decode_codec == id(codec)

        self.decode_times[id(codec)].append(time.perf_counter() - last_decode_start)
        self.codecs[id(codec)] = codec

        self.last_decode = None

    def results(self) -> dict:
        return dict(
            encode_times={
                repr(self.codecs[c]): ts for c, ts in self.encode_times.items()
            },
            decode_times={
                repr(self.codecs[c]): ts for c, ts in self.decode_times.items()
            },
        )
