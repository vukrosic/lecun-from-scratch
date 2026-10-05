"""Step 4. Efficient BackProp: two small habits that decide whether a network trains.

Same network, same data, same learning rate, plain gradient descent. Only two things change:

1. the inputs: raw pixel values 0..16, or each pixel shifted to average 0 and scaled to spread 1
2. the starting weights: random numbers of size 1, or random numbers of size 1 / sqrt(inputs)

The reason is the tanh neuron. Feed it a large number and it sits flat at +1 or -1.
A flat neuron passes back almost no blame, so the weights behind it stop learning.
"""
import numpy as np

from conv import Dense, Model, Tanh, accuracy, train
from data import load

HIDDEN, RATE, EPOCHS = 32, 0.1, 20


def network(start_size, rng):
    """64 pixels -> 32 tanh neurons -> 10 answers. Dense() already divides by sqrt(inputs);
    multiplying by sqrt(inputs) again gives plain size-1 weights."""
    scale1 = np.sqrt(64) if start_size == 'size 1' else 1.0
    scale2 = np.sqrt(HIDDEN) if start_size == 'size 1' else 1.0
    return Model([Dense(64, HIDDEN, rng, scale1), Tanh(), Dense(HIDDEN, 10, rng, scale2)])


def stuck(model, x):
    """Share of hidden neurons sitting flat (output beyond +0.99 or -0.99), over all pictures."""
    model.forward(x)
    return float((np.abs(model.layers[1].y) > 0.99).mean())


def run():
    xtr, ytr, xte, yte = load()
    raw_tr, raw_te = xtr * 16, xte * 16                       # the pixel values as stored: 0..16
    mean, spread = raw_tr.mean(axis=0), raw_tr.std(axis=0) + 1e-3
    inputs = {'raw 0..16': (raw_tr, raw_te), 'centred': ((raw_tr - mean) / spread, (raw_te - mean) / spread)}
    print(f'a raw picture: average pixel {raw_tr.mean():.2f}, largest {raw_tr.max():.0f}')
    print(f'the same picture centred: average {inputs["centred"][0].mean():+.2f}, spread {inputs["centred"][0].std():.2f}')

    print(f'\n{"inputs":10s} {"starting weights":18s} {"flat at start":>13s} {"after 1 epoch":>14s} {"after 20":>9s}')
    rows = []
    for name in ('raw 0..16', 'centred'):
        for start in ('size 1', '1 / sqrt(inputs)'):
            a, b = inputs[name]
            net = network(start, np.random.default_rng(0))
            flat, curve = stuck(net, a), []
            train(net, a, ytr, epochs=EPOCHS, lr=RATE, watch=lambda e, m: curve.append(accuracy(m, b, yte)))
            rows.append(dict(inputs=name, start=start, flat=flat, curve=curve))
            print(f'{name:10s} {start:18s} {flat:13.1%} {curve[0]:14.1%} {curve[-1]:9.1%}')

    w = np.random.default_rng(0)
    for n in (64, 1024):
        x = w.normal(0, 1, (2000, n))
        print(f'\n{n} inputs of spread 1, weights of size 1          : the sum has spread {np.std(x @ w.normal(0, 1, n)):.1f}')
        print(f'{n} inputs of spread 1, weights of size 1/sqrt({n}): the sum has spread {np.std(x @ w.normal(0, 1 / np.sqrt(n), n)):.1f}')
    return dict(rows=rows, raw_mean=float(raw_tr.mean()), raw_max=float(raw_tr.max()))


if __name__ == '__main__':
    run()
