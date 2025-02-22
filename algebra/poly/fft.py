from tinygrad import Tensor, dtypes
import math
import time
import numpy as np

def bit_reverse_indices(n):
    indices = Tensor.arange(n, dtype=dtypes.uint32)
    nbits = int(math.log2(n))
    reversed_indices = indices.zeros_like()
    for i in range(nbits):
        bit = (indices >> i) & 1
        reversed_indices = reversed_indices + (bit << (nbits - 1 - i))
    return reversed_indices

def precompute_twiddles(N, isign):
    stages = int(math.log2(N))
    twiddles = {}
    for stage in range(stages):
        mmax = 2 ** (stage + 1)
        theta = isign * 2 * math.pi / mmax
        angles = Tensor.arange(mmax // 2) * theta
        twiddles[mmax] = (angles.cos(), angles.sin())
    return twiddles

def fft(x_real: Tensor, x_imag: Tensor, inverse: bool = False) -> tuple[Tensor, Tensor]:
    N = x_real.shape[0]
    assert N == x_imag.shape[0] and (N & (N - 1)) == 0, "N must be a power of 2"
    assert x_real.dtype == dtypes.int32, f"x_real must be i32, got {x_real.dtype}"
    assert x_imag.dtype == dtypes.int32, f"x_imag must be i32, got {x_imag.dtype}"

    x_real_float = x_real.float()
    x_imag_float = x_imag.float()

    rev_indices = bit_reverse_indices(N)
    x_real_float = x_real_float[rev_indices].contiguous()
    x_imag_float = x_imag_float[rev_indices].contiguous()

    isign = 1 if inverse else -1
    twiddles = precompute_twiddles(N, isign)
    mmax = 2
    while mmax <= N:
        wr, wi = twiddles[mmax]
        n_twiddles = mmax // 2
        for m in range(n_twiddles):
            i_start = m
            step = mmax
            i_indices = Tensor.arange(i_start, N, step, dtype=dtypes.int32)
            j_indices = i_indices + n_twiddles
            wr_m = wr[m]
            wi_m = wi[m]
            tempr = x_real_float[j_indices] * wr_m - x_imag_float[j_indices] * wi_m
            tempi = x_real_float[j_indices] * wi_m + x_imag_float[j_indices] * wr_m
            x_real_float[j_indices] = x_real_float[i_indices] - tempr
            x_imag_float[j_indices] = x_imag_float[i_indices] - tempi
            x_real_float[i_indices] = x_real_float[i_indices] + tempr
            x_imag_float[i_indices] = x_imag_float[i_indices] + tempi
        mmax *= 2

    if inverse:
        x_real_float = x_real_float / N
        x_imag_float = x_imag_float / N

    x_real_out = x_real_float.cast(dtypes.int32)
    x_imag_out = x_imag_float.cast(dtypes.int32)
    return x_real_out, x_imag_out

def test_fft():
    N = 8
    x_real = Tensor.arange(N, dtype=dtypes.int32)
    x_imag = Tensor.zeros(N, dtype=dtypes.int32)

    start = time.time()
    fft_real, fft_imag = fft(x_real, x_imag, inverse=False)
    fft_time = time.time() - start
    print(f"FFT took {fft_time} seconds")
    print("FFT Real:", fft_real.numpy())
    print("FFT Imag:", fft_imag.numpy())

    start = time.time()
    ifft_real, ifft_imag = fft(fft_real, fft_imag, inverse=True)
    ifft_time = time.time() - start
    print(f"IFFT took {ifft_time} seconds")
    print("IFFT Real (should match input):", ifft_real.numpy())
    print("IFFT Imag (should be ~0):", ifft_imag.numpy())

    x_np = np.arange(N, dtype=np.int32) + 0j
    start = time.time()
    fft_np = np.fft.fft(x_np)
    numpy_time = time.time() - start
    print(f"NumPy FFT took {numpy_time} seconds")
    print("NumPy FFT Real:", fft_np.real)
    print("NumPy FFT Imag:", fft_np.imag)

if __name__ == "__main__":
    test_fft()