from tinygrad import Tensor
import time
import numpy as np

from algebra.ff.m31 import M31
from algebra.poly.univariate import Polynomial


def poly_mul_loop(a: list[int], b: list[int]) -> list[int]:
  na = len(a)
  nb = len(b)
  result_len = na + nb - 1

  a_list = a
  b_list = b

  result = [0] * result_len

  for i in range(na):
    temp = [a_list[i] * b_val for b_val in b_list]
    for j in range(nb):
      result[i + j] += temp[j]

  return result


def benchmark_poly_mul(sizes=[2, 10, 50, 100, 500, 1000, 5000, 10000]):
  opt_times = []

  for size in sizes:
    a = Tensor(np.random.randint(-10, 11, size=size, dtype=np.int32)).contiguous()
    b = Tensor(np.random.randint(-10, 11, size=size, dtype=np.int32)).contiguous()

    # Warm-up runs
    a_poly = Polynomial(a.numpy().tolist(), M31)
    b_poly = Polynomial(b.numpy().tolist(), M31)
    a_poly * b_poly

    # Benchmark loop version
    start_time = time.time()
    n_runs = 2

    # Benchmark optimal version
    start_time = time.time()
    for _ in range(n_runs):
      a_poly * b_poly
    opt_time = (time.time() - start_time) / n_runs
    opt_times.append(opt_time * 1000)  # Convert to ms

    print(f"Size {size}:")
    print(f"  Optimal version:     {opt_time * 1000:.4f} ms")
    print()


if __name__ == "__main__":
  benchmark_poly_mul()

# bench on m2 mac pro
# ❯ uv run script_poly_mul.py
# Size 2:
#   Optimal version:     1.5551 ms

# Size 10:
#   Optimal version:     1.6236 ms

# Size 50:
#   Optimal version:     1.5205 ms

# Size 100:
#   Optimal version:     1.7134 ms

# Size 500:
#   Optimal version:     1.8010 ms

# Size 1000:
#   Optimal version:     1.5864 ms

# Size 5000:
#   Optimal version:     1.5751 ms

# Size 10000:
#   Optimal version:     1.6276 ms
