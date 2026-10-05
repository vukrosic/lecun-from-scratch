"""Step 3. LeNet: convolution, pooling, convolution, answer.

Now the filters learn from labels only. Nobody says what an edge is. The network gets a
picture, says which digit it is, and the blame for a wrong answer flows back to the filters.

The test that shows why this design matters: the digit can be anywhere on the page.
A network of filters looks for the same strokes everywhere. A network where every pixel
has its own weights has to learn each digit again at each position.
"""
import numpy as np

from conv import Conv, Dense, Flat, Model, Pool, Relu, accuracy, lenet, softmax, onehot, train
from data import load

SIZE = 12                      # the page is 12 x 12, the digit on it is 8 x 8


def scatter(x, rng, size=SIZE):
    """Put each 8 x 8 digit at a random place on an empty page."""
    x = x.reshape(-1, 8, 8)
    out = np.zeros((len(x), 1, size, size))
    for i, img in enumerate(x):
        top, left = rng.integers(0, size - 7, 2)
        out[i, 0, top:top + 8, left:left + 8] = img
    return out


def dense_net(hidden, rng, size=SIZE):
    """The rival: every pixel wired to every hidden unit. No filters, no sharing."""
    return Model([Flat(), Dense(size * size, hidden, rng), Relu(), Dense(hidden, 10, rng)])


def check_blame(model, x, y):
    """Nudge one filter weight, measure the error again, compare with what backward() says."""
    t = onehot(y)
    loss = lambda: -np.mean(np.sum(t * np.log(softmax(model.forward(x)) + 1e-12), axis=1))
    model.backward((softmax(model.forward(x)) - t) / len(x))
    said, W, nudge = model.layers[0].gW[4, 0], model.layers[0].W, 1e-5
    W[4, 0] += nudge
    up = loss()
    W[4, 0] -= 2 * nudge
    down = loss()
    W[4, 0] += nudge
    return said, (up - down) / (2 * nudge)


def run(epochs=30):
    xtr, ytr, xte, yte = load()
    rng = np.random.default_rng(1)
    Xtr, Xte = scatter(xtr, rng), scatter(xte, rng)
    print(f'{len(Xtr)} training pages, {len(Xte)} test pages, each {SIZE} x {SIZE}, the digit at a random place')

    net = lenet(np.random.default_rng(0), SIZE)
    said, measured = check_blame(net, Xtr[:8], ytr[:8])
    print(f'blame for one filter weight: backward() says {said:.6f}, nudging says {measured:.6f}')

    shapes, x = [], Xtr[:1]
    for layer in net.layers:
        x = layer.forward(x)
        shapes.append((type(layer).__name__, x.shape[1:]))
    print('one page going through the network:')
    for name, shape in shapes:
        print(f'   {name:6s} -> {" x ".join(str(s) for s in shape)}')

    curve = []
    watch = lambda epoch, m: curve.append(accuracy(m, Xte, yte))
    train(net, Xtr, ytr, epochs=epochs, watch=watch)
    small = train(dense_net(15, np.random.default_rng(0)), Xtr, ytr, epochs=epochs)
    big = train(dense_net(128, np.random.default_rng(0)), Xtr, ytr, epochs=epochs)
    rows = [('filters (LeNet layout)', net), ('every pixel wired, same size', small), ('every pixel wired, 8 times bigger', big)]
    print(f'\n{"network":34s} {"weights":>8s} {"train":>7s} {"test":>7s}')
    out = []
    for name, m in rows:
        out.append((name, m.n_weights(), accuracy(m, Xtr, ytr), accuracy(m, Xte, yte)))
        print(f'{name:34s} {out[-1][1]:8,d} {out[-1][2]:7.1%} {out[-1][3]:7.1%}')

    print('\nthe 6 filters it learned in the first layer (+ pulls up, - pulls down):')
    filters = net.layers[0].W.T.reshape(6, 3, 3)
    for row in range(3):
        print('   ' + '    '.join(' '.join('+' if v > 0.15 else '-' if v < -0.15 else '.' for v in f[row]) for f in filters))
    return dict(net=net, rows=out, curve=curve, filters=filters, shapes=shapes, Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte,
                said=said, measured=measured)


if __name__ == '__main__':
    run()
