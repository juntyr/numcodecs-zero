__all__ = ["CodecStack"]

from collections.abc import Buffer
from typing import Optional, Self

import numcodecs
import numcodecs.compat
import numcodecs.registry
import numpy as np

from numcodecs.abc import Codec

from .observers.abc import (
    ObservableCodec,
    CodecObserver,
    encode_with_observers,
    decode_with_observers,
)


class CodecStack(ObservableCodec, tuple[Codec]):
    """
    A stack of codecs, which makes up a combined codec.

    On encoding, the codecs are applied to encode from left to right, i.e.
    ```python
    CodecStack(a, b, c).encode(buf)
    ```
    computes
    ```python
    c.encode(b.encode(a.encode(buf)))
    ```

    On decoding, the codecs are applied to decode from right to left, i.e.
    ```python
    CodecStack(a, b, c).decode(buf)
    ```
    computes
    ```python
    a.decode(b.decode(c.decode(buf)))
    ```

    The [`CodecStack`][numcodecs_combinators.CodecStack] provides the additional
    [`encode_decode(buf)`][numcodecs_combinators.CodecStack.encode_decode]
    method that computes
    ```python
    stack.decode(stack.encode(buf))
    ```
    but makes use of knowing the shapes and dtypes of all intermediary encoding
    stages.
    """

    __slots__ = ()

    codec_id = "combinators.stack"

    def __init__(self, *args: tuple[dict | Codec]):
        pass

    def __new__(cls, *args: tuple[dict | Codec]) -> Self:
        return super(CodecStack, cls).__new__(
            cls,
            tuple(
                codec
                if isinstance(codec, Codec)
                else numcodecs.registry.get_codec(codec)
                for codec in args
            ),
        )

    def encode(
        self, buf: Buffer, *, observers: Optional[list[CodecObserver]] = None
    ) -> Buffer:
        """Encode the data in `buf`.

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
            encoded = encode_with_observers(
                codec,
                numcodecs.compat.ensure_contiguous_ndarray_like(encoded, flatten=False),  # type: ignore
                observers=observers,
            )
        return encoded

    def decode(
        self,
        buf: Buffer,
        out: Optional[Buffer] = None,
        *,
        observers: Optional[list[CodecObserver]] = None,
    ) -> Buffer:
        """Decode the data in `buf`.

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
            decoded = decode_with_observers(
                codec,
                numcodecs.compat.ensure_contiguous_ndarray_like(decoded, flatten=False),  # type: ignore
                out=None,
                observers=observers,
            )
        return numcodecs.compat.ndarray_copy(decoded, out)  # type: ignore

    def encode_decode(
        self, buf: Buffer, *, observers: Optional[list[CodecObserver]] = None
    ) -> Buffer:
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
            silhouettes.append((encoded.shape, np.dtype(encoded.dtype.name)))
            encoded = numcodecs.compat.ensure_contiguous_ndarray_like(
                encode_with_observers(
                    codec,
                    encoded,  # type: ignore
                    observers=observers,
                ),
                flatten=False,
            )

        decoded = encoded

        for codec in reversed(self):
            shape, dtype = silhouettes.pop()
            out = np.empty(shape=shape, dtype=dtype)
            decoded = decode_with_observers(
                codec,
                decoded,  # type: ignore
                out=out,
                observers=observers,
            ).reshape(shape)  # type: ignore

        return decoded  # type: ignore

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
