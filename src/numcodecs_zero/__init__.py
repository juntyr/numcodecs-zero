"""
[`ZeroCodec`][numcodecs_zero.ZeroCodec] for the [`numcodecs`][numcodecs] buffer compression API.
"""

__all__ = ["ZeroCodec"]

from io import BytesIO

import leb128
import numcodecs.compat
import numcodecs.registry
import numpy as np
from numcodecs.abc import Codec
from typing_extensions import Buffer  # MSPV 3.12


class ZeroCodec(Codec):
    """
    Codec that decodes to an all-zero array of the same data type and shape as
    the original data.

    Encoding produces a bytestring containing information on the data type and
    shape.
    """

    __slots__ = ()

    codec_id: str = "zero"  # type: ignore

    def encode(self, buf: Buffer) -> bytes:
        """
        Encode the `buf`fer information.

        Parameters
        ----------
        buf : Buffer
            Data to be encoded. May be any object supporting the new-style
            buffer protocol.

        Returns
        -------
        enc : bytes
            Encoded `buf`fer information as a bytestring.
        """

        a = numcodecs.compat.ensure_ndarray(buf)
        dtype, shape = a.dtype, a.shape

        # message: dtype shape
        message = []

        message.append(leb128.u.encode(len(dtype.str)))
        message.append(dtype.str.encode("ascii"))

        message.append(leb128.u.encode(len(shape)))
        for s in shape:
            message.append(leb128.u.encode(s))

        return b"".join(message)

    def decode(self, buf: Buffer, out: None | Buffer = None) -> Buffer:
        """
        Decode the `buf`fer information.

        Parameters
        ----------
        buf : Buffer
            Encoded buffer information. Must be an object representing a
            bytestring, e.g. [`bytes`][bytes] or a 1D array of
            [`np.uint8`][numpy.uint8]s etc.
        out : Buffer, optional
            Writeable buffer to store decoded data. N.B. if provided, this
            buffer must be exactly the right size to store the decoded data.

        Returns
        -------
        dec : Buffer
            Decoded data. May be any object supporting the new-style buffer
            protocol. The decoded data will be all-zero after decoding.
        """

        b = numcodecs.compat.ensure_bytes(buf)

        b_io = BytesIO(b)

        dtype = np.dtype(b_io.read(leb128.u.decode_reader(b_io)[0]).decode("ascii"))
        shape = tuple(
            leb128.u.decode_reader(b_io)[0]
            for _ in range(leb128.u.decode_reader(b_io)[0])
        )

        decoded = np.zeros(shape, dtype)

        return numcodecs.compat.ndarray_copy(decoded, out)  # type: ignore


numcodecs.registry.register_codec(ZeroCodec)
