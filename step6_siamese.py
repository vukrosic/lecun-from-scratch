"""Step 6. Siamese network: learn to compare, not to name.

Two pictures go through the same network (the same weights, used twice). Each comes out as
a point: 2 numbers. The network is never told which digit a picture is. It is only told,
for each pair, "same" or "different":

    same pair      -> pull the two points together
    different pair -> push them apart, until they are at least MARGIN apart

LeCun's group built this in 1993 to check signatures, and gave the loss this form in 2005-2006.
"""
import numpy as np

from conv import Dense, Model, Tanh
from data import load

MARGIN, STEPS, PAIRS, RATE = 2.0, 4000, 32, 0.05


def twin(rng):
    """64 pixels -> 64 tanh neurons -> a point with 2 coordinates."""
    return Model([Dense(64, 64, rng), Tanh(), Dense(64, 2, rng)])


def train_pairs(net, x, y, rng):
    history = []
    for step in range(STEPS):
        i = rng.integers(0, len(x), PAIRS)
        same = rng.random(PAIRS) < 0.5
        j = np.array([rng.choice(np.flatnonzero((y == y[a]) if s else (y != y[a]))) for a, s in zip(i, same)])
        z = net.forward(np.concatenate([x[i], x[j]]))          # both pictures, one network
        gap = z[:PAIRS] - z[PAIRS:]
        dist = np.sqrt((gap ** 2).sum(axis=1) + 1e-9)
        pull = 2 * gap                                         # blame for a same pair: distance squared
        push = -2 * np.maximum(MARGIN - dist, 0)[:, None] * gap / dist[:, None]    # (MARGIN - distance) squared
        blame = np.where(same[:, None], pull, push) / PAIRS
        net.backward(np.concatenate([blame, -blame]))
        net.step(RATE)
        if step % 200 == 0 or step == STEPS - 1:
            history.append((step, float(dist[same].mean()), float(dist[~same].mean())))
    return history


def pair_test(points, y, rng, n=4000):
    """Distances for n random pairs, and whether each pair really is the same digit."""
    i, j = rng.integers(0, len(y), n), rng.integers(0, len(y), n)
    keep = i != j
    i, j = i[keep], j[keep]
    return np.sqrt(((points[i] - points[j]) ** 2).sum(axis=1)), y[i] == y[j]


def best_line(dist, same):
    """The distance that best separates same pairs from different pairs."""
    lines = np.linspace(0, dist.max(), 300)
    score = [((dist[same] < v).mean() + (dist[~same] >= v).mean()) / 2 for v in lines]
    return float(lines[int(np.argmax(score))])


def verify(tr_points, ytr, te_points, yte, rng):
    """Pick the dividing distance on training pictures, score it on test pictures.
    The score is the average of 'same pairs called same' and 'different pairs called different'."""
    line = best_line(*pair_test(tr_points, ytr, rng))
    dist, same = pair_test(te_points, yte, rng)
    return float(((dist[same] < line).mean() + (dist[~same] >= line).mean()) / 2), line


def one_example(points, y, rng, trials=100):
    """Name one example of each digit. Every other picture gets the name of the nearest example."""
    out = []
    for _ in range(trials):
        ref = [rng.choice(np.flatnonzero(y == d)) for d in range(10)]
        nearest = np.argmin(((points[:, None] - points[ref][None]) ** 2).sum(axis=2), axis=1)
        out.append((nearest == y).mean())
    return float(np.mean(out))


def run():
    xtr, ytr, xte, yte = load()
    rng = np.random.default_rng(0)
    net = twin(rng)
    start = net.forward(xte).copy()
    history = train_pairs(net, xtr, ytr, rng)
    for step, near, far in history[::4]:
        print(f'step {step:4d}: same pairs {near:.2f} apart, different pairs {far:.2f} apart')
    points = net.forward(xte)

    r = np.random.default_rng(1)
    v_net, line = verify(net.forward(xtr), ytr, points, yte, r)
    v_raw, _ = verify(xtr, ytr, xte, yte, r)
    print(f'\n"are these two the same digit?" on test pictures it never saw:')
    print(f'   distance between the raw pictures: {v_raw:.1%} right')
    print(f'   distance between the two points  : {v_net:.1%} right (same if closer than {line:.2f})')
    o_net, o_raw = one_example(points, yte, r), one_example(xte, yte, r)
    print('one named example per digit, every other test picture takes the nearest name:')
    print(f'   nearest raw picture: {o_raw:.1%} right')
    print(f'   nearest point      : {o_net:.1%} right')
    print('\nwhere each digit ended up (average point of its test pictures):')
    centres = [points[yte == d].mean(axis=0) for d in range(10)]
    for d, c in enumerate(centres):
        print(f'   {d}: ({c[0]:+.2f}, {c[1]:+.2f})')
    return dict(history=history, points=points, start=start, yte=yte, v_net=v_net, v_raw=v_raw, line=line,
                o_net=o_net, o_raw=o_raw, centres=centres)


if __name__ == '__main__':
    run()
