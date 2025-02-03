import numcodecs_combinators
import numcodecs
import numpy as np

observers = [
    numcodecs_combinators.observers.ByesizeObserver(),
    numcodecs_combinators.observers.WalltimeObserver(),
]
stack = numcodecs_combinators.CodecStack(
    numcodecs.BitRound(keepbits=6), numcodecs.Zlib(level=7)
)

data = np.random.normal(size=(100, 100))

stack.encode_decode(data, observers=observers)

print([o.results() for o in observers])
