"""Steps 1 to 3 packed into a small library, used by steps 3 to 8.

A network is a list of layers. Every layer has forward() and backward(). backward() takes the
blame for the layer's output and returns the blame for its input, and it stores a gradient for
each of the layer's own weights. Training is plain stochastic gradient descent.
"""
import numpy as np

np.seterr(divide='ignore', over='ignore', invalid='ignore')   # numpy 2.0 on Apple chips prints false matmul warnings


def patches(x, k):
    """Cut every k x k window out of a stack of pictures. x: (n, channels, height, width)."""
    v = np.lib.stride_tricks.sliding_window_view(x, (k, k), axis=(2, 3))    # n, c, oh, ow, k, k
    n, c, oh, ow = v.shape[:4]
    return v.transpose(0, 2, 3, 1, 4, 5).reshape(n, oh, ow, c * k * k)


class Conv:
    """c_out filters, each k x k, slid over every position. The same weights at every position."""
    def __init__(self, c_in, c_out, k, rng):
        self.k, self.c_in = k, c_in
        self.W = rng.normal(0, np.sqrt(1.0 / (c_in * k * k)), (c_in * k * k, c_out))
        self.b = np.zeros(c_out)

    def forward(self, x, train=False):
        self.shape, self.p = x.shape, patches(x, self.k)
        return (self.p @ self.W + self.b).transpose(0, 3, 1, 2)             # n, c_out, oh, ow

    def backward(self, d):
        d = d.transpose(0, 2, 3, 1)                                         # n, oh, ow, c_out
        n, oh, ow, _ = d.shape
        # one shared weight is used at every position, so its gradient is the sum over positions
        self.gW = np.tensordot(self.p, d, axes=([0, 1, 2], [0, 1, 2]))
        self.gb = d.sum(axis=(0, 1, 2))
        back = (d @ self.W.T).reshape(n, oh, ow, self.c_in, self.k, self.k)
        dx = np.zeros(self.shape)
        for i in range(self.k):
            for j in range(self.k):
                dx[:, :, i:i + oh, j:j + ow] += back[..., i, j].transpose(0, 3, 1, 2)
        return dx


class Pool:
    """Keep the largest value in every 2 x 2 block: the picture gets half as wide and half as tall."""
    W = b = None

    def forward(self, x, train=False):
        n, c, h, w = x.shape
        blocks = x.reshape(n, c, h // 2, 2, w // 2, 2).transpose(0, 1, 2, 4, 3, 5).reshape(n, c, h // 2, w // 2, 4)
        self.arg, self.shape = blocks.argmax(axis=-1), x.shape
        return blocks.max(axis=-1)

    def backward(self, d):
        n, c, h, w = self.shape
        blocks = np.zeros((n, c, h // 2, w // 2, 4))
        np.put_along_axis(blocks, self.arg[..., None], d[..., None], axis=-1)   # blame goes to the winner
        return blocks.reshape(n, c, h // 2, w // 2, 2, 2).transpose(0, 1, 2, 4, 3, 5).reshape(n, c, h, w)


class Relu:
    W = b = None

    def forward(self, x, train=False):
        self.on = x > 0
        return x * self.on

    def backward(self, d):
        return d * self.on


class Tanh:
    W = b = None

    def forward(self, x, train=False):
        self.y = np.tanh(x)
        return self.y

    def backward(self, d):
        return d * (1 - self.y ** 2)


class Flat:
    W = b = None

    def forward(self, x, train=False):
        self.shape = x.shape
        return x.reshape(len(x), -1)

    def backward(self, d):
        return d.reshape(self.shape)


class Dense:
    """Every input connected to every output. scale=1 is the starting size from step 4."""
    def __init__(self, n_in, n_out, rng, scale=1.0):
        self.W = rng.normal(0, scale * np.sqrt(1.0 / n_in), (n_in, n_out))
        self.b = np.zeros(n_out)

    def forward(self, x, train=False):
        self.x = x
        return x @ self.W + self.b

    def backward(self, d):
        self.gW, self.gb = self.x.T @ d, d.sum(axis=0)
        return d @ self.W.T


class Model:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x, train=False):
        for layer in self.layers:
            x = layer.forward(x, train)
        return x

    def backward(self, d):
        for layer in reversed(self.layers):
            d = layer.backward(d)
        return d

    def step(self, lr):
        for layer in self.layers:
            if layer.W is not None:
                layer.W -= lr * layer.gW
                layer.b -= lr * layer.gb

    def n_weights(self):
        return sum(layer.W.size + layer.b.size for layer in self.layers if layer.W is not None)


def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def onehot(y, n=10):
    out = np.zeros((len(y), n))
    out[np.arange(len(y)), y] = 1
    return out


def train(model, x, y, epochs=30, batch=16, lr=0.05, seed=0, watch=None):
    """Stochastic gradient descent on the cross-entropy between the answers and the labels."""
    rng, t = np.random.default_rng(seed), onehot(y, int(y.max()) + 1)
    for epoch in range(epochs):
        order = rng.permutation(len(x))
        for i in range(0, len(x), batch):
            idx = order[i:i + batch]
            p = softmax(model.forward(x[idx], train=True))
            model.backward((p - t[idx]) / len(idx))
            model.step(lr)
        if watch:
            watch(epoch, model)
    return model


def accuracy(model, x, y):
    return float((model.forward(x).argmax(axis=1) == y).mean())


def lenet(rng, size=12):
    """conv 3x3 -> pool -> conv 3x3 -> flatten -> 10 answers. A small copy of LeNet's layout."""
    s = ((size - 2) // 2 - 2)
    return Model([Conv(1, 6, 3, rng), Relu(), Pool(), Conv(6, 16, 3, rng), Relu(), Flat(), Dense(16 * s * s, 10, rng)])
