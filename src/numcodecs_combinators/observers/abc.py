__all__ = [
    "CodecObserver",
    "ObservableCodec",
    "encode_with_observers",
    "decode_with_observers",
]

from abc import abstractmethod
from collections.abc import Buffer
from typing import Optional

from numcodecs.abc import Codec


class CodecObserver:
    def pre_encode(self, codec: Codec, buf: Buffer) -> None:
        pass

    def post_encode(self, codec: Codec, buf: Buffer, encoded: Buffer) -> None:
        pass

    def pre_decode(
        self, codec: Codec, buf: Buffer, out: Optional[Buffer] = None
    ) -> None:
        pass

    def post_decode(self, codec: Codec, buf: Buffer, decoded: Buffer) -> None:
        pass

    @abstractmethod
    def results(self) -> dict:
        pass


class ObservableCodec(Codec):
    @abstractmethod
    def encode(
        self, buf: Buffer, *, observers: Optional[list[CodecObserver]] = None
    ) -> Buffer:
        """Encode data in `buf`.

        Parameters
        ----------
        buf : buffer-like
            Data to be encoded. May be any object supporting the new-style
            buffer protocol.

        Returns
        -------
        enc : buffer-like
            Encoded data. May be any object supporting the new-style buffer
            protocol.
        """

    @abstractmethod
    def decode(
        self,
        buf: Buffer,
        out: Optional[Buffer] = None,
        *,
        observers: Optional[list[CodecObserver]] = None,
    ) -> Buffer:
        """Decode data in `buf`.

        Parameters
        ----------
        buf : buffer-like
            Encoded data. May be any object supporting the new-style buffer
            protocol.
        out : buffer-like, optional
            Writeable buffer to store decoded data. N.B. if provided, this buffer must
            be exactly the right size to store the decoded data.

        Returns
        -------
        dec : buffer-like
            Decoded data. May be any object supporting the new-style
            buffer protocol.
        """


def encode_with_observers(
    codec: Codec, buf: Buffer, *, observers: Optional[list[CodecObserver]] = None
) -> Buffer:
    if observers is not None:
        for observer in observers:
            observer.pre_encode(codec, buf)

    if isinstance(codec, ObservableCodec):
        encoded: Buffer = codec.encode(buf, observers=observers)
    else:
        encoded: Buffer = codec.encode(buf)  # type: ignore

    if observers is not None:
        for observer in observers:
            observer.post_encode(codec, buf, encoded)

    return encoded


def decode_with_observers(
    codec: Codec,
    buf: Buffer,
    out: Optional[Buffer] = None,
    *,
    observers: Optional[list[CodecObserver]] = None,
) -> Buffer:
    if observers is not None:
        for observer in observers:
            observer.pre_decode(codec, buf, out=out)

    if isinstance(codec, ObservableCodec):
        decoded: Buffer = codec.decode(buf, out=out, observers=observers)
    else:
        decoded: Buffer = codec.decode(buf, out=out)  # type: ignore

    if observers is not None:
        for observer in observers:
            observer.post_decode(codec, buf, decoded)

    return decoded
