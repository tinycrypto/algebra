from tinygrad import Tensor

a = Tensor([1, 2, 3])
b = Tensor([4, 5, 6])

print((a % 3).numpy())
