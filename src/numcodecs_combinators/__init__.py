__all__ = ["CodecStack"]

from collections.abc import Buffer
from typing import Optional

import numcodecs
import numcodecs.registry
import numpy as np

from numcodecs.abc import Codec


class CodecStack(Codec, tuple[Codec]):
    """
    A stack of codecs, which makes up a combined codec.

    On encoding, the codecs are applied to encode from left to right, i.e.
    ```
    CodecStack(a, b, c).encode(buf)
    ```
    computes
    ```
    c.encode(b.encode(a.encode(buf)))
    ```

    On decoding, the codecs are applied to decode from right to left, i.e.
    ```
    CodecStack(a, b, c).decode(buf)
    ```
    computes
    ```
    a.decode(b.decode(c.decode(buf)))
    ```

    The [`CodecStack`][numcodecs_combinators.CodecStack] provides the additional
    [`encode_decode(buf)`][numcodecs_combinators.CodecStack.encode_decode]
    method that computes
    ```
    stack.decode(stack.encode(buf))
    ```
    but makes use of knowing the shapes and dtypes of all intermediary encoding
    stages.
    """

    __slots__ = ()

    codec_id = "combinators.stack"

    def __init__(self, *args: tuple[dict | Codec]):
        pass

    def __new__(cls, *args: tuple[dict | Codec]):
        return super(CodecStack, cls).__new__(
            cls,
            tuple(
                codec
                if isinstance(codec, Codec)
                else numcodecs.registry.get_codec(codec)
                for codec in args
            ),
        )

    def encode(self, buf: Buffer) -> Buffer:
        """Encode data in `buf`.

        Parameters
        ----------
        buf : Buffer
            Data to be encoded. May be any object supporting the new-style
            buffer protocol.

        Returns
        -------
        enc : Buffer
            Encoded data. May be any object supporting the new-style buffer
            protocol.
        """

        encoded = buf
        for codec in self:
            encoded = codec.encode(
                numcodecs.compat.ensure_contiguous_ndarray_like(encoded, flatten=False)
            )
        return encoded

    def decode(self, buf: Buffer, out: Optional[Buffer] = None):
        """Decode data in `buf`.

        Parameters
        ----------
        buf : Buffer
            Encoded data. May be any object supporting the new-style buffer
            protocol.
        out : Buffer, optional
            Writeable buffer to store decoded data. N.B. if provided, this buffer must
            be exactly the right size to store the decoded data.

        Returns
        -------
        dec : Buffer
            Decoded data. May be any object supporting the new-style
            buffer protocol.
        """

        decoded = buf
        for codec in reversed(self):
            decoded = codec.decode(
                numcodecs.compat.ensure_contiguous_ndarray_like(decoded, flatten=False),
                out=None,
            )
        return numcodecs.compat.ndarray_copy(decoded, out)

    def encode_decode(self, buf: Buffer) -> Buffer:
        """
        Encode, then decode the data in `buf`.

        Parameters
        ----------
        buf : Buffer
            Data to be encoded. May be any object supporting the new-style
            buffer protocol.

        Returns
        -------
        dec : Buffer
            Decoded data. May be any object supporting the new-style
            buffer protocol.
        """

        encoded = numcodecs.compat.ensure_contiguous_ndarray_like(buf, flatten=False)
        silhouettes = []

        for codec in self:
            silhouettes.append((encoded.shape, encoded.dtype))
            encoded = numcodecs.compat.ensure_contiguous_ndarray_like(
                codec.encode((encoded)), flatten=False
            )

        decoded = encoded

        for codec in reversed(self):
            shape, dtype = silhouettes.pop()
            out = np.empty(shape=shape, dtype=dtype)
            decoded = codec.decode(decoded, out).reshape(shape)

        return type(buf)(decoded)

    def get_config(self) -> dict:
        """
        Returns the configuration of the codec stack.

        [`numcodecs.registry.get_codec(config)`][numcodecs.registry.get_codec]
        can be used to reconstruct this stack from the returned config.

        Returns
        -------
        config : dict
            Configuration of the codec stack.
        """

        return dict(
            id=type(self).codec_id,
            codecs=tuple(codec.get_config() for codec in self),
        )

    @classmethod
    def from_config(cls, config: dict):
        """
        Instantiate the codec stack from a configuration [`dict`][dict].

        Parameters
        ----------
        config : dict
            Configuration of the codec stack.

        Returns
        -------
        stack : CodecStack
            Instantiated codec stack.
        """

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
