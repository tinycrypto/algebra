import numpy as np


def big_int_to_chunks(n, chunk_bits=32, num_chunks=None):
  if n < 0:
    raise ValueError("n must be nonnegative")
  base = 1 << chunk_bits
  chunks = []
  while n:
    chunks.append(n % base)
    n //= base
  if not chunks:
    chunks.append(0)
  if num_chunks is not None:
    if len(chunks) < num_chunks:
      chunks.extend([0] * (num_chunks - len(chunks)))
    else:
      chunks = chunks[:num_chunks]
  return np.array(chunks, dtype=np.uint32)


def bigints_to_tensor(bigints, chunk_bits=32, num_chunks=None):
  if isinstance(bigints, int):
    bigints = [bigints]

  chunks_list = [big_int_to_chunks(n, chunk_bits, num_chunks) for n in bigints]
  arr = np.stack(chunks_list)
  return arr
