import numpy as np

def big_int_to_chunks(n, chunk_bits=32, num_chunks=None):
    """
    Convert a big integer n into a NumPy array of 32-bit (or chunk_bits-bit) pieces.
    The least-significant chunk is first (little-endian order).
    
    Parameters:
      n          : a Python integer (can be huge)
      chunk_bits : number of bits in each chunk (default 32)
      num_chunks : if given, pad or truncate the result to exactly this many chunks
      
    Returns:
      A NumPy array of dtype np.uint32 containing the chunks.
    """
    if n < 0:
        raise ValueError("n must be nonnegative")
    base = 1 << chunk_bits
    chunks = []
    while n:
        chunks.append(n % base)
        n //= base
    if not chunks:
        chunks.append(0)
    # If a fixed number of chunks is desired, pad with zeros (or truncate)
    if num_chunks is not None:
        if len(chunks) < num_chunks:
            chunks.extend([0] * (num_chunks - len(chunks)))
        else:
            chunks = chunks[:num_chunks]
    return np.array(chunks, dtype=np.uint32)

def bigints_to_tensor(bigints, chunk_bits=32, num_chunks=None):
    """
    Convert a big integer or a list of Python big integers into a tinygrad Tensor
    where each big integer is represented by an array of chunks.
    """
    # If bigints is a single integer, wrap it in a list.
    if isinstance(bigints, int):
        bigints = [bigints]
    
    chunks_list = [big_int_to_chunks(n, chunk_bits, num_chunks) for n in bigints]
    arr = np.stack(chunks_list)
    return arr