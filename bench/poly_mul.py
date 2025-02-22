from tinygrad import Tensor, dtypes
import time
import numpy as np


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


def poly_mul_optimal(a: Tensor, b: Tensor) -> Tensor:
  na, nb = a.shape[0], b.shape[0]
  result_len = na + nb - 1

  idx = Tensor.arange(result_len, dtype=dtypes.int32).reshape(result_len, 1, 1)
  ai = Tensor.arange(na, dtype=dtypes.int32).reshape(1, na, 1)
  bi = Tensor.arange(nb, dtype=dtypes.int32).reshape(1, 1, nb)

  idx_expanded = idx.expand(result_len, na, nb)
  ai_expanded = ai.expand(result_len, na, nb)
  bi_expanded = bi.expand(result_len, na, nb)

  mask = ai_expanded + bi_expanded == idx_expanded

  a_exp = a.reshape(1, na, 1).expand(result_len, na, nb)
  b_exp = b.reshape(1, 1, nb).expand(result_len, na, nb)

  products = a_exp * b_exp * mask
  return products.sum(axis=(1, 2))


def benchmark_poly_mul(sizes=[2, 10, 50, 100, 500, 1000, 5000, 10000]):
  loop_times = []
  opt_times = []

  for size in sizes:
    a = Tensor(np.random.randint(-10, 11, size=size, dtype=np.int32)).contiguous()
    b = Tensor(np.random.randint(-10, 11, size=size, dtype=np.int32)).contiguous()

    # Warm-up runs
    a_list = a.numpy().tolist()
    b_list = b.numpy().tolist()
    poly_mul_loop(a_list, b_list)
    poly_mul_optimal(a, b).realize()

    # Benchmark loop version
    start_time = time.time()
    n_runs = 2
    for _ in range(n_runs):
      result_loop = poly_mul_loop(a_list, b_list)
    loop_time = (time.time() - start_time) / n_runs
    loop_times.append(loop_time * 1000)  # Convert to ms

    # Benchmark optimal version
    start_time = time.time()
    for _ in range(n_runs):
      result_opt = poly_mul_optimal(a, b).realize()
    opt_time = (time.time() - start_time) / n_runs
    opt_times.append(opt_time * 1000)  # Convert to ms

    # Verify results match
    np.testing.assert_equal(result_loop, result_opt.numpy())

    print(f"Size {size}:")
    print(f"  Loop version (lists): {loop_time * 1000:.4f} ms")
    print(f"  Optimal version:     {opt_time * 1000:.4f} ms")
    print(f"  Speedup:             {loop_time / opt_time:.2f}x")
    print()


if __name__ == "__main__":
  benchmark_poly_mul()
