from collections import defaultdict
from collections.abc import Buffer
from dataclasses import dataclass

import numpy as np
from numcodecs.abc import Codec

from .abc import CodecObserver


@dataclass
class Bytesize:
    pre: int
    post: int


class ByesizeObserver(CodecObserver):
    codecs: dict[int, Codec]
    encode_sizes: defaultdict[int, list[Bytesize]]
    decode_sizes: defaultdict[int, list[Bytesize]]

    def __init__(self):
        self.codecs = dict()
        self.encode_sizes = defaultdict(list)
        self.decode_sizes = defaultdict(list)

    def post_encode(self, codec: Codec, buf: Buffer, encoded: Buffer) -> None:
        buf, encoded = np.asarray(buf), np.asarray(encoded)

        self.encode_sizes[id(codec)].append(
            Bytesize(pre=buf.nbytes, post=encoded.nbytes)
        )
        self.codecs[id(codec)] = codec

    def post_decode(self, codec: Codec, buf: Buffer, decoded: Buffer) -> None:
        buf, decoded = np.asarray(buf), np.asarray(decoded)

        self.decode_sizes[id(codec)].append(
            Bytesize(pre=buf.nbytes, post=decoded.nbytes)
        )
        self.codecs[id(codec)] = codec

    def results(self) -> dict:
        return dict(
            encode_sizes={
                repr(self.codecs[c]): ts for c, ts in self.encode_sizes.items()
            },
            decode_sizes={
                repr(self.codecs[c]): ts for c, ts in self.decode_sizes.items()
            },
        )
