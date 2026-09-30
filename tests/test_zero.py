import numcodecs
import numcodecs.registry
import numpy as np


def test_from_config():
    codec = numcodecs.registry.get_codec(dict(id="zero"))
    assert codec.__class__.__name__ == "ZeroCodec"
    assert codec.__class__.__module__ == "numcodecs_zero"


def check_roundtrip(data: np.ndarray, value: int | float = 0):
    codec = numcodecs.registry.get_codec(dict(id="zero", value=value))
    assert codec.get_config() == dict(id="zero", value=value)

    encoded = codec.encode(data)
    decoded = codec.decode(encoded)

    assert decoded.dtype == data.dtype
    assert decoded.shape == data.shape
    if isinstance(value, float) and np.isnan(value):
        assert np.all(np.isnan(decoded))
    else:
        assert np.all(decoded == value)


def test_roundtrip():
    check_roundtrip(np.zeros(tuple()))
    check_roundtrip(np.zeros((0,)))
    check_roundtrip(np.arange(1000).reshape(10, 10, 10))
    check_roundtrip(np.array([np.inf, -np.inf, np.nan, -np.nan, 0.0, -0.0]))


def test_constant_value():
    check_roundtrip(np.zeros((3, 4)), value=2.5)
    check_roundtrip(np.zeros((3, 4), dtype=np.int32), value=-7)
    check_roundtrip(np.zeros((3, 4)), value=np.nan)
    check_roundtrip(np.zeros((0,)), value=1.0)

    # the default is still an all-zero reconstruction
    codec = numcodecs.registry.get_codec(dict(id="zero"))
    assert codec.get_config() == dict(id="zero", value=0)
    assert repr(codec) == "ZeroCodec(value=0)"
