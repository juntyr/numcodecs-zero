__all__ = ["CodecStack"]

import numcodecs
import numcodecs.abc
import numcodecs.registry
import numpy as np


class CodecStack(numcodecs.abc.Codec, tuple[numcodecs.abc.Codec]):
    """
    A stack of codecs, which makes up a combined codec.

    On encoding, the codecs are applied to encode from left to right, i.e.
    `CodecStack(a, b, c).encode(buf)` computes
    `c.encode(b.encode(a.encode(buf)))`.

    On decoding, the codecs are applied to decode from right to left, i.e.
    `CodecStack(a, b, c).decode(buf)` computes
    `a.decode(b.decode(c.decode(buf)))`.

    The `CodecStack` provides the additional `encode_decode(buf)` method that
    computes `stack.decode(stack.encode(buf))` but makes use of knowing the
    shapes and dtypes of all intermediary encoding stages.
    """

    __slots__ = ()

    codec_id = "combinators.stack"

    def __new__(cls, *args: tuple[dict | numcodecs.abc.Codec]):
        return super(CodecStack, cls).__new__(
            cls,
            tuple(
                codec
                if isinstance(codec, numcodecs.abc.Codec)
                else numcodecs.registry.get_codec(codec)
                for codec in args
            ),
        )

    def encode(self, buf):
        encoded = buf
        for codec in self:
            encoded = codec.encode(np.copy(encoded))
        return encoded

    def decode(self, buf, out=None):
        decoded = buf
        for codec in reversed(self):
            decoded = codec.decode(np.copy(decoded), out=None)
        return numcodecs.compat.ndarray_copy(decoded, out)

    def encode_decode(self, buf):
        """
        Encode, then decode the data in `buf`.

        Parameters
        ----------
        buf : buffer-like
            Data to be encoded. May be any object supporting the new-style
            buffer protocol.

        Returns
        -------
        dec : buffer-like
            Decoded data. May be any object supporting the new-style
            buffer protocol.
        """

        encoded = np.ascontiguousarray(buf)
        silhouettes = []

        for codec in self:
            silhouettes.append((encoded.shape, encoded.dtype))
            encoded = numcodecs.compat.ensure_contiguous_ndarray_like(
                codec.encode(encoded)
            )

        decoded = encoded

        for codec in reversed(self):
            shape, dtype = silhouettes.pop()
            out = np.empty(shape=shape, dtype=dtype)
            decoded = codec.decode(decoded, out).reshape(shape)

        return decoded

    def get_config(self):
        return dict(
            id=type(self).codec_id,
            codecs=tuple(codec.get_config() for codec in self),
        )

    @classmethod
    def from_config(cls, config):
        return cls(*config["codecs"])

    def __repr__(self):
        repr = ", ".join(f"{codec!r}" for codec in self)

        return f"{type(self).__name__}({repr})"

    def __add__(self, other):
        return CodecStack(*tuple.__add__(self, other))

    def __mul__(self, other):
        return CodecStack(*tuple.__mul__(self, other))

    def __rmul__(self, other):
        return CodecStack(*tuple.__rmul__(self, other))


numcodecs.registry.register_codec(CodecStack)
