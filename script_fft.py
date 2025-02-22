import time
import numpy as np
from tinygrad import Tensor, dtypes
from algebra.poly.fft import fft


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
