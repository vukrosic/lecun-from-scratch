"""Step 8. Joint embedding: learn from pictures with no labels at all.

Take one digit and put it at two different places on a page. That is two views of the same
thing. A filter network (step 3's layout) turns each view into 16 numbers. Training rule:
the two views of one picture should get the same 16 numbers.

There is a trap. The easiest way to obey is to give EVERY picture the same 16 numbers.
That is called collapse, and then the numbers say nothing. LeCun's fix (VICReg, 2022) adds
two more rules, measured across a batch of pictures:

    variance   : each of the 16 numbers must keep a spread of at least 1 across pictures
    covariance : the 16 numbers must not copy each other

This is the idea behind his JEPA proposal: predict in the space of the network's own
numbers, not pixel by pixel.
"""
import numpy as np

from conv import Conv, Dense, Flat, Model, Pool, Relu, accuracy, lenet, onehot, softmax, train
from data import load
from step3_lenet import scatter

NUMBERS, STEPS, BATCH, RATE = 16, 6000, 64, 0.3
VAR_WEIGHT, COV_WEIGHT = 1.0, 0.04


def encoder(rng):
    """Step 3's network, but the last layer gives 16 numbers and not 10 answers."""
    return Model([Conv(1, 6, 3, rng), Relu(), Pool(), Conv(6, 16, 3, rng), Relu(), Flat(), Dense(144, NUMBERS, rng)])


def rules(za, zb, var_weight, cov_weight):
    """The three rules as one error, plus the blame for every number of both views."""
    n, d = za.shape
    same = ((za - zb) ** 2).mean()                             # rule 1: two views agree
    blame_a = 2 * (za - zb) / (n * d)
    blame_b = -blame_a
    spread_err = copy_err = 0.0
    blames = []
    for z, blame in ((za, blame_a), (zb, blame_b)):
        zc = z - z.mean(axis=0)
        spread = np.sqrt((zc ** 2).sum(axis=0) / (n - 1) + 1e-4)
        spread_err += np.maximum(0, 1 - spread).mean()         # rule 2: keep a spread of 1
        blame = blame - var_weight * (spread < 1) * zc / ((n - 1) * spread) / d
        cov = zc.T @ zc / (n - 1)
        off = cov - np.diag(np.diag(cov))
        copy_err += (off ** 2).sum() / d                       # rule 3: do not copy each other
        blame = blame + cov_weight * 4 * zc @ off / ((n - 1) * d)
        blames.append(blame)
    return same + var_weight * spread_err + cov_weight * copy_err, blames[0], blames[1]


def learn(x, var_weight, cov_weight, steps=STEPS, seed=0):
    net, rng, history = encoder(np.random.default_rng(seed)), np.random.default_rng(seed + 1), []
    for step in range(steps):
        i = rng.integers(0, len(x), BATCH)
        views = np.concatenate([scatter(x[i], rng), scatter(x[i], rng)])    # same digits, two places
        z = net.forward(views)
        _, blame_a, blame_b = rules(z[:BATCH], z[BATCH:], var_weight, cov_weight)
        net.backward(np.concatenate([blame_a, blame_b]))
        net.step(RATE)
        if step % 250 == 0 or step == steps - 1:
            history.append((step, float(z.std(axis=0).mean())))
    return net, history


def pick_labelled(y, per_digit, seed):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.permutation(np.flatnonzero(y == d))[:per_digit] for d in range(10)])


def read_out(f_train, y_train, f_test, y_test, labelled):
    """Freeze the features. Train only a single layer on the few labelled pictures."""
    mean, spread = f_train.mean(axis=0), f_train.std(axis=0) + 1e-6
    a, b, t = (f_train[labelled] - mean) / spread, (f_test - mean) / spread, onehot(y_train[labelled])
    W, bias = np.zeros((a.shape[1], 10)), np.zeros(10)
    for _ in range(300):
        p = softmax(a @ W + bias)
        W -= 0.1 * a.T @ (p - t) / len(a) + 1e-3 * W
        bias -= 0.1 * (p - t).mean(axis=0)
    return float(((b @ W + bias).argmax(axis=1) == y_test).mean())


def run():
    xtr, ytr, xte, yte = load()
    rng = np.random.default_rng(5)
    Xtr, Xte = scatter(xtr, rng), scatter(xte, rng)

    # check the blame formula by nudging one number
    r = np.random.default_rng(0)
    za, zb = r.normal(0, 0.7, (16, 5)), r.normal(0, 0.7, (16, 5))
    err, blame_a, _ = rules(za, zb, 25.0, 1.0)
    za[3, 2] += 1e-5
    measured = (rules(za, zb, 25.0, 1.0)[0] - err) / 1e-5
    print(f'blame for one number: formula {blame_a[3, 2]:.5f}, nudging {measured:.5f}')

    print('\nrule 1 only (two views must agree):')
    collapsed, h1 = learn(xtr, 0.0, 0.0, steps=2000)
    for step, spread in h1[::2]:
        print(f'   step {step:4d}: spread of the 16 numbers across pictures {spread:.4f}')
    print('all three rules:')
    net, h3 = learn(xtr, VAR_WEIGHT, COV_WEIGHT)
    for step, spread in h3[::4]:
        print(f'   step {step:4d}: spread of the 16 numbers across pictures {spread:.4f}')

    flat = lambda x: x.reshape(len(x), -1)
    feats = {'raw pixels': (flat(Xtr), flat(Xte)),
             'rule 1 only (collapsed)': (collapsed.forward(Xtr), collapsed.forward(Xte)),
             'all three rules': (net.forward(Xtr), net.forward(Xte))}
    print(f'\nread the digit from the features, test pictures, average of 5 draws of the labelled pictures')
    print(f'{"features":28s} {"10 labels (1 per digit)":>24s} {"100 labels (10 per digit)":>26s}')
    table = {}
    for name, (a, b) in feats.items():
        table[name] = [float(np.mean([read_out(a, ytr, b, yte, pick_labelled(ytr, k, s)) for s in range(5)])) for k in (1, 10)]
        print(f'{name:28s} {table[name][0]:24.1%} {table[name][1]:26.1%}')
    sup = []
    for k in (1, 10):
        accs = []
        for s in range(5):
            idx = pick_labelled(ytr, k, s)
            accs.append(accuracy(train(lenet(np.random.default_rng(s), 12), Xtr[idx], ytr[idx], epochs=100), Xte, yte))
        sup.append(float(np.mean(accs)))
    table['step 3 network, labels only'] = sup
    print(f'{"step 3 network, labels only":28s} {sup[0]:24.1%} {sup[1]:26.1%}')
    print(f'\nthe filter network saw {len(xtr):,} pictures with no label before the labels arrived')
    return dict(h1=h1, h3=h3, table=table, formula=float(blame_a[3, 2]), measured=float(measured),
                z=net.forward(Xte), zc=collapsed.forward(Xte), yte=yte)


if __name__ == '__main__':
    run()
