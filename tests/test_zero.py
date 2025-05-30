import numcodecs
import numcodecs.registry
import numpy as np


def test_from_config():
    codec = numcodecs.registry.get_codec(dict(id="zero"))
    assert codec.__class__.__name__ == "ZeroCodec"
    assert codec.__class__.__module__ == "numcodecs_zero"


def check_roundtrip(data: np.ndarray):
    codec = numcodecs.registry.get_codec(dict(id="zero"))

    encoded = codec.encode(data)
    decoded = codec.decode(encoded)

    assert decoded.dtype == data.dtype
    assert decoded.shape == data.shape
    assert np.all(decoded == 0)


def test_roundtrip():
    check_roundtrip(np.zeros(tuple()))
    check_roundtrip(np.zeros((0,)))
    check_roundtrip(np.arange(1000).reshape(10, 10, 10))
    check_roundtrip(np.array([np.inf, -np.inf, np.nan, -np.nan, 0.0, -0.0]))
