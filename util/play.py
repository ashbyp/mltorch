import torch
import numpy as np


def l():
    print('-' * 80)


def play_with_tensors():
    x = torch.rand(2, 3)
    print(x)
    x = torch.zeros(2, 3)
    print(x)
    x = torch.ones(2, 3)
    print(x)
    x = torch.tensor([[1, 2], [3, 4]])
    print(x)
    x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float16)
    print(x)

    x = torch.ones(2, 3)
    y = x * 2

    z = x.add(y)
    print(x)
    print(y)
    print(z)

    x.add_(y)
    print(x)
    print(y)

    l()
    x = torch.rand(5, 3)
    print(x)
    print(x[:,2])
    print(x[4, :])

    l()
    x = torch.rand(4, 4)
    print(x.view(16))
    print(x.view(-1, 8))
    print(x.view(8, -1))

    l()
    x = torch.ones(5)
    y = x.numpy()
    print(x, y)

    # share memory
    x.add_(3)
    print(x, y)

    a = np.ones(5)
    b = torch.from_numpy(a)
    print(a, b)
    a *= 6
    print(a, b)


def play_with_gradients():
    x = torch.randn(3, requires_grad=True)
    y = x + 2
    print(x, y)

    z = y * y * 2
    print(x, z)

    z = z.mean()
    print(z)
    z.backward()
    print(x.grad)


if __name__ == '__main__':
    #play_with_tensors()
    #play_with_gradients()
    pass
