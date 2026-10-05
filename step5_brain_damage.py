"""Step 5. Optimal Brain Damage: delete most of a trained network's weights.

Which weights can go? LeCun's answer (1989): for each weight, estimate how much the error
would grow if that weight were set to zero, and delete the ones that cost the least.

    cost of deleting a weight  =  1/2 x curvature x weight x weight

Curvature says how sharply the error bends when that one weight moves. A big weight in a
flat direction is cheap to delete. A small weight in a sharp direction is not.
"""
import numpy as np

from conv import Dense, Model, Tanh, accuracy, onehot, softmax, train
from data import canvas, load

HIDDEN = 32


def make():
    rng = np.random.default_rng(0)
    return Model([Dense(144, HIDDEN, rng), Tanh(), Dense(HIDDEN, 10, rng)])


def error(model, x, t):
    return float(-np.mean(np.sum(t * np.log(softmax(model.forward(x)) + 1e-12), axis=1)))


def curvature(model, x):
    """LeCun's shortcut: pass second derivatives backwards, the same way blame is passed.
    One backward pass gives the curvature of every weight."""
    p = softmax(model.forward(x))
    d2 = p * (1 - p)                                          # curvature at the 10 answers
    h_out = (model.layers[2].x ** 2).T @ d2 / len(x)
    d2 = d2 @ (model.layers[2].W ** 2).T                      # back through the weights, squared
    d2 = d2 * (1 - model.layers[1].y ** 2) ** 2               # back through tanh, slope squared
    h_in = (x ** 2).T @ d2 / len(x)
    return [h_in, h_out]


def curvature_by_nudging(model, x, t, W, idx, nudge=0.01):
    """The slow way, one weight at a time: error a bit above, at, and a bit below the weight."""
    base, w0 = error(model, x, t), W[idx]
    W[idx] = w0 + nudge
    up = error(model, x, t)
    W[idx] = w0 - nudge
    down = error(model, x, t)
    W[idx] = w0
    return (up - 2 * base + down) / nudge ** 2


def deletion_cost(model, x, t, W, idx):
    """The slowest way: really delete the weight and measure how much the error grew."""
    base, w0 = error(model, x, t), W[idx]
    W[idx] = 0.0
    cost = error(model, x, t) - base
    W[idx] = w0
    return cost


def scores(model, x, rule, rng):
    Ws = [model.layers[0].W, model.layers[2].W]
    if rule == 'brain damage':
        return np.concatenate([(0.5 * h * W ** 2).ravel() for h, W in zip(curvature(model, x), Ws)])
    if rule == 'smallest first':
        return np.concatenate([np.abs(W).ravel() for W in Ws])
    return rng.random(sum(W.size for W in Ws))               # 'at random'


def masks_for(score, alive, share, shapes):
    """Mark the lowest-scoring weights for deletion until `share` of all weights are gone."""
    score = np.where(alive, score, -1.0)                      # already deleted: stay deleted
    keep = np.ones(len(score))
    keep[np.argsort(score)[:int(share * len(score))]] = 0
    n = shapes[0][0] * shapes[0][1]
    return [keep[:n].reshape(shapes[0]), keep[n:].reshape(shapes[1])]


def retrain(model, masks, x, t, epochs=10, lr=0.1, seed=0):
    """Ordinary training, but deleted weights are held at zero."""
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        order = rng.permutation(len(x))
        for i in range(0, len(x), 16):
            idx = order[i:i + 16]
            model.backward((softmax(model.forward(x[idx])) - t[idx]) / len(idx))
            model.step(lr)
            model.layers[0].W *= masks[0]
            model.layers[2].W *= masks[1]


def run():
    xtr, ytr, xte, yte = load()
    Xtr, Xte, t = canvas(xtr).reshape(-1, 144), canvas(xte).reshape(-1, 144), onehot(ytr)
    model = train(make(), Xtr, ytr, epochs=40, lr=0.1)
    Ws = [model.layers[0].W, model.layers[2].W]
    shapes, total = [W.shape for W in Ws], sum(W.size for W in Ws)
    full = accuracy(model, Xte, yte)
    print(f'trained network: {total:,} connections, {full:.1%} on test digits')

    # is the formula any good? try it on 300 weights against really deleting them
    rng, H = np.random.default_rng(0), curvature(model, Xtr)
    sample = []
    for _ in range(300):
        layer = int(rng.integers(0, 2))
        idx = tuple(int(rng.integers(0, s)) for s in shapes[layer])
        w = Ws[layer][idx]
        sample.append(dict(w=float(w), shortcut=float(H[layer][idx]),
                           nudged=curvature_by_nudging(model, Xtr, t, Ws[layer], idx),
                           predicted=float(0.5 * H[layer][idx] * w * w),
                           real=deletion_cost(model, Xtr, t, Ws[layer], idx)))
    col = lambda k: np.array([s[k] for s in sample])
    print(f'300 weights: curvature by shortcut vs by nudging, correlation {np.corrcoef(col("shortcut"), col("nudged"))[0, 1]:.2f}')
    print(f'300 weights: predicted cost vs real cost of deleting, correlation {np.corrcoef(col("predicted"), col("real"))[0, 1]:.2f}')
    print(f'300 weights: size of weight vs real cost of deleting, correlation {np.corrcoef(np.abs(col("w")), col("real"))[0, 1]:.2f}')

    # delete in one cut, no retraining
    print(f'\ndelete in one cut, no retraining   {"brain damage":>13s} {"smallest first":>15s} {"at random":>10s}')
    saved, one_cut = [W.copy() for W in Ws], []
    for share in (0.5, 0.7, 0.8, 0.9):
        row = []
        for rule in ('brain damage', 'smallest first', 'at random'):
            masks = masks_for(scores(model, Xtr, rule, rng), np.ones(total, bool), share, shapes)
            for W, m in zip(Ws, masks):
                W *= m
            row.append(accuracy(model, Xte, yte))
            for W, s in zip(Ws, saved):
                W[:] = s
        one_cut.append((share, row))
        print(f'   {share:4.0%} of connections deleted   {row[0]:13.1%} {row[1]:15.1%} {row[2]:10.1%}')

    # delete a bit, retrain, delete more
    print('\ndelete, retrain 10 epochs, delete more:')
    steps, result = [], {}
    for rule in ('brain damage', 'smallest first'):
        model = train(make(), Xtr, ytr, epochs=40, lr=0.1)
        Ws = [model.layers[0].W, model.layers[2].W]
        masks = [np.ones(s) for s in shapes]
        for share in (0.5, 0.7, 0.8, 0.9, 0.95):
            alive = np.concatenate([m.ravel() for m in masks]) > 0
            masks = masks_for(scores(model, Xtr, rule, rng), alive, share, shapes)
            for W, m in zip(Ws, masks):
                W *= m
            before = accuracy(model, Xte, yte)
            retrain(model, masks, Xtr, t)
            left = int(sum(m.sum() for m in masks))
            steps.append(dict(rule=rule, share=share, left=left, before=before, after=accuracy(model, Xte, yte)))
            print(f'   {rule:14s} {share:4.0%} deleted, {left:5,d} left: {before:6.1%} right after the cut, {steps[-1]["after"]:6.1%} after retraining')
            if share == 0.9:
                result[rule] = masks[0].sum(axis=1).reshape(12, 12)        # connections left per input pixel

    alive = result['brain damage']
    print('\nconnections left from each of the 144 input pixels (of 32) at 90 % deleted:')
    for row in alive:
        print('   ' + ' '.join(f'{int(v):2d}' if v else ' .' for v in row))
    print(f'pixels with no connection left: {int((alive == 0).sum())} of 144')
    return dict(total=total, full=full, sample=sample, one_cut=one_cut, steps=steps, alive=alive)


if __name__ == '__main__':
    run()
