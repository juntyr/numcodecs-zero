# numcodecs-combinators

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
