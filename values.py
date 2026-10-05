"""Runs every step once more and saves the numbers and pictures the film draws: values.json."""
import contextlib
import io
import json

import numpy as np

import step1_convolution as s1
import step2_shared_weights as s2
import step3_lenet as s3
import step4_efficient_backprop as s4
import step5_brain_damage as s5
import step6_siamese as s6
import step7_energy as s7
import step8_joint_embedding as s8
from conv import Dense, Model, Tanh, softmax
from data import canvas, load

R = lambda a, k=3: np.round(np.asarray(a, dtype=float), k).tolist()


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


xtr, ytr, xte, yte = load()
V = {'data': {'n_train': len(xtr), 'n_test': len(xte), 'samples': R(xtr[:40], 2), 'labels': ytr[:40].tolist(),
              'test': R(xte[:40], 2), 'test_labels': yte[:40].tolist()}}

# ---------- step 1: sliding a filter
r1 = quiet(s1.run)
img = r1['img']
V['conv'] = {
    'img': R(img, 2), 'label': r1['label'], 'vertical': s1.VERTICAL, 'horizontal': s1.HORIZONTAL,
    'v': R(r1['v'], 2), 'h': R(r1['h'], 2), 'at': list(r1['at']), 'at_sum': round(r1['at_sum'], 2),
    'pa': R(s1.page(img, 1, 1), 2), 'pb': R(s1.page(img, 3, 4), 2), 'a': R(r1['a'], 2), 'b': R(r1['b'], 2),
    'sa': list(r1['sa']), 'sb': list(r1['sb']), 'same': bool(r1['same']),
    'vmax': round(max(max(row) for row in r1['v']), 2), 'vbest': list(s1.strongest(r1['v'])),
}
print('conv', V['conv']['at_sum'], V['conv']['sa'], V['conv']['sb'], V['conv']['vbest'], V['conv']['vmax'])

# ---------- step 2: learning the filter
r2 = quiet(s2.run)
pics = [x.reshape(8, 8).tolist() for x in xtr[:s2.N_PICTURES]]
targets = [s1.convolve(p, s2.SECRET) for p in pics]
g0 = s2.blame(r2['start'], pics, targets)
out0 = s1.convolve(pics[0], r2['start'])
miss0 = [[2 * (out0[r][c] - targets[0][r][c]) for c in range(6)] for r in range(6)]
V['shared'] = {
    'start': R(r2['start'], 2), 'snaps': {str(k): R(v, 3) for k, v in r2['snaps'].items()}, 'final': R(r2['filt'], 3),
    'history': R(r2['history'], 4), 'base': round(r2['base'], 1), 'formula': round(r2['formula'], 2), 'measured': round(r2['measured'], 2),
    'blame0': R(g0, 1), 'out0': R(out0, 2), 'target0': R(targets[0], 2), 'miss0': R(miss0, 2),
    'contrib0': R([[miss0[r][c] * pics[0][r][c + 2] for c in range(6)] for r in range(6)], 2),
    'outs': {str(k): R(s1.convolve(pics[0], v), 2) for k, v in r2['snaps'].items()},
    'rate': s2.RATE, 'n': s2.N_PICTURES,
}
print('shared', V['shared']['base'], V['shared']['formula'], V['shared']['measured'], [round(r2['history'][k], 3) for k in (1, 10, 200)])

# ---------- step 3: a network that reads digits
r3 = quiet(s3.run)
net = r3['net']


def trace(x):
    acts, a = [], x
    for layer in net.layers:
        a = layer.forward(a)
        acts.append(a)
    return acts


ex = []
order = [int(np.flatnonzero(r3['yte'] == d)[k]) for k in (0, 1) for d in (3, 7, 4, 0, 8, 2, 5, 9, 1, 6)]
for i in order[:8]:
    a = trace(r3['Xte'][i:i + 1])
    ex.append({'page': R(r3['Xte'][i, 0], 2), 'label': int(r3['yte'][i]), 'c1': R(a[1][0], 2), 'pool': R(a[2][0], 2),
               'c2': R(a[4][0], 2), 'p': R(softmax(a[6])[0], 3), 'said': int(a[6][0].argmax())})
pred = net.forward(r3['Xte']).argmax(axis=1)
V['lenet'] = {
    'rows': [{'name': n, 'w': int(w), 'train': round(a, 4), 'test': round(b, 4)} for n, w, a, b in r3['rows']],
    'curve': R(r3['curve'], 4), 'filters': R(r3['filters'], 2), 'shapes': [[n, list(map(int, s))] for n, s in r3['shapes']],
    'said': round(float(r3['said']), 4), 'measured': round(float(r3['measured']), 4), 'ex': ex,
    'pages': R(r3['Xtr'][:24, 0], 2), 'page_labels': r3['ytr'][:24].tolist(),
    'grid': {'pages': R(r3['Xte'][:60, 0], 2), 'labels': r3['yte'][:60].tolist(), 'said': pred[:60].tolist()},
    'w_parts': [int(l.W.size + l.b.size) for l in net.layers if l.W is not None],
}
print('lenet', V['lenet']['rows'], V['lenet']['said'], V['lenet']['measured'], V['lenet']['w_parts'], [(e['label'], e['said']) for e in ex])

# ---------- step 4: training tricks
r4 = quiet(s4.run)
raw_tr = xtr * 16
mean, spread = raw_tr.mean(axis=0), raw_tr.std(axis=0) + 1e-3
inputs = {'raw 0..16': raw_tr, 'centred': (raw_tr - mean) / spread}
rows = []
for row in r4['rows']:
    m = s4.network(row['start'], np.random.default_rng(0))
    a = inputs[row['inputs']]
    pre = m.layers[0].forward(a)
    y = np.tanh(pre)
    rows.append({'inputs': row['inputs'], 'start': row['start'], 'flat': round(row['flat'], 4), 'curve': R(row['curve'], 4),
                 'pre_spread': round(float(pre.std()), 2), 'pre_mean': round(float(pre.mean()), 2),
                 'dots': R(pre[:3].ravel(), 2), 'hist': (np.histogram(y, bins=20, range=(-1, 1))[0] / y.size).round(4).tolist()})
w = np.random.default_rng(0)
sums = []
for n in (64, 1024):
    x = w.normal(0, 1, (2000, n))
    sums.append([n, round(float(np.std(x @ w.normal(0, 1, n))), 1), round(float(np.std(x @ w.normal(0, 1 / np.sqrt(n), n))), 1)])
V['effbp'] = {'rows': rows, 'raw_mean': round(r4['raw_mean'], 2), 'raw_max': r4['raw_max'], 'sums': sums,
              'pic_raw': R(raw_tr[s1.PICK], 0), 'pic_centred': R(inputs['centred'][s1.PICK], 2), 'mean_img': R(mean, 2),
              'hidden': s4.HIDDEN, 'epochs': s4.EPOCHS}
print('effbp', [(r['inputs'], r['start'], r['flat'], r['curve'][0], r['curve'][-1], r['pre_spread']) for r in rows], sums, V['effbp']['raw_mean'])

# ---------- step 5: brain damage
r5 = quiet(s5.run)
col = lambda k: np.array([s[k] for s in r5['sample']])
alive = np.asarray(r5['alive'])
inside = np.zeros((12, 12), bool)
inside[2:10, 2:10] = True
V['damage'] = {
    'total': int(r5['total']), 'full': round(r5['full'], 4), 'hidden': s5.HIDDEN,
    'sample': [[round(s['w'], 3), float(f"{s['predicted']:.3g}"), float(f"{s['real']:.3g}"), float(f"{s['shortcut']:.3g}"), float(f"{s['nudged']:.3g}")] for s in r5['sample']],
    'corr': [round(float(np.corrcoef(col('shortcut'), col('nudged'))[0, 1]), 2), round(float(np.corrcoef(col('predicted'), col('real'))[0, 1]), 2),
             round(float(np.corrcoef(np.abs(col('w')), col('real'))[0, 1]), 2)],
    'one_cut': [[share, R(row, 4)] for share, row in r5['one_cut']],
    'steps': [{**s, 'before': round(s['before'], 4), 'after': round(s['after'], 4)} for s in r5['steps']],
    'alive': alive.astype(int).tolist(), 'dead': int((alive == 0).sum()), 'dead_border': int(((alive == 0) & ~inside).sum()),
    'dead_inside': int(((alive == 0) & inside).sum()), 'mean_page': R(canvas(xtr)[:, 0].mean(axis=0), 2),
    'page': R(canvas(xtr[s1.PICK])[0, 0], 2),
}
print('damage', V['damage']['total'], V['damage']['full'], V['damage']['corr'], V['damage']['one_cut'], V['damage']['dead'], V['damage']['dead_border'], V['damage']['dead_inside'])
print('  steps', [(s['rule'][:5], s['share'], s['left'], s['before'], s['after']) for s in V['damage']['steps']])

# ---------- step 6: twin networks
r6 = quiet(s6.run)
STEPS6 = s6.STEPS
frames = {}
for k in (60, 150, 300, 600, 1200, 2400):
    rng = np.random.default_rng(0)
    twin = s6.twin(rng)
    s6.STEPS = k
    s6.train_pairs(twin, xtr, ytr, rng)
    frames[str(k)] = R(twin.forward(xte), 2)
s6.STEPS = STEPS6
frames['0'], frames[str(STEPS6)] = R(r6['start'], 2), R(r6['points'], 2)
dist, same = s6.pair_test(r6['points'], yte, np.random.default_rng(1))
rawd, raws = s6.pair_test(xte, yte, np.random.default_rng(1))
hist = lambda d, m, top: (np.histogram(d[m], bins=30, range=(0, top))[0] / m.sum()).round(4).tolist()
V['siamese'] = {
    'history': [[int(a), round(b, 2), round(c, 2)] for a, b, c in r6['history']], 'frames': frames, 'yte': yte.tolist(),
    'v_net': round(r6['v_net'], 4), 'v_raw': round(r6['v_raw'], 4), 'line': round(r6['line'], 2), 'o_net': round(r6['o_net'], 4),
    'o_raw': round(r6['o_raw'], 4), 'centres': R(r6['centres'], 2), 'steps': STEPS6, 'pairs': s6.PAIRS, 'margin': s6.MARGIN,
    'hist_same': hist(dist, same, 6), 'hist_diff': hist(dist, ~same, 6), 'raw_same': hist(rawd, raws, float(rawd.max())),
    'raw_diff': hist(rawd, ~raws, float(rawd.max())), 'raw_top': round(float(rawd.max()), 2),
}
print('siamese', V['siamese']['history'][::5], V['siamese']['v_net'], V['siamese']['v_raw'], V['siamese']['o_net'], V['siamese']['o_raw'], V['siamese']['line'], V['siamese']['margin'])

# ---------- step 7: energy
r7 = quiet(s7.run)
STEPS7 = s7.STEPS
x7, y7 = s7.data(np.random.default_rng(0))
curves = {}
for k in (0, 300, 1000, 3000, 6000):
    s7.STEPS = k
    e = s7.train_energy(x7, y7, np.random.default_rng(0))
    curves[str(k)] = R(s7.landscape(e, 0.5)[::2], 3)
s7.STEPS = STEPS7
curves[str(STEPS7)] = R(r7['curve'][::2], 3)
V['energy'] = {
    'pts': R(np.stack([r7['x'][:400], r7['y'][:400]], axis=1), 3), 'questions': R(r7['questions'], 2), 'said': R(r7['said'], 3),
    'found': [R(v, 2) for v in r7['found']], 'miss_ord': round(r7['miss_ord'], 3), 'miss_en': round(r7['miss_en'], 3),
    'curves': curves, 'surface': R(r7['surface'][:, ::4], 2), 'line': R(r7['line'], 3), 'grid': [-1.2, 1.2, 121],
    'steps': STEPS7, 'margin': s7.MARGIN, 'true': R(np.sqrt(r7['questions']), 2),
}
print('energy', V['energy']['said'], V['energy']['found'], V['energy']['miss_ord'], V['energy']['miss_en'], [round(min(c), 2) for c in curves.values()], [round(max(c), 2) for c in curves.values()])

# ---------- step 8: learning without labels
r8 = quiet(s8.run)
rng = np.random.default_rng(3)
pick = [int(np.flatnonzero(ytr == d)[0]) for d in (4, 7, 3, 0, 8, 5)]
views = [[R(s3.scatter(xtr[i:i + 1], rng)[0, 0], 2), R(s3.scatter(xtr[i:i + 1], rng)[0, 0], 2)] for i in pick]
z, zc = r8['z'], r8['zc']
zm = z - z.mean(axis=0)
_, _, vt = np.linalg.svd(zm, full_matrices=False)
V['jepa'] = {
    'h1': [[int(a), round(b, 4)] for a, b in r8['h1']], 'h3': [[int(a), round(b, 4)] for a, b in r8['h3']],
    'table': {k: R(v, 4) for k, v in r8['table'].items()}, 'formula': round(r8['formula'], 5), 'measured': round(r8['measured'], 5),
    'views': views, 'view_labels': [int(ytr[i]) for i in pick], 'z2': R(zm @ vt[:2].T, 2), 'zc2': R((zc - zc.mean(axis=0)) @ vt[:2].T, 4),
    'spread': R(z.std(axis=0), 2), 'spread_c': R(zc.std(axis=0), 4), 'yte': r8['yte'].tolist(), 'numbers': s8.NUMBERS,
    'steps': s8.STEPS, 'zrow': R(z[:6], 2), 'zcrow': R(zc[:6], 3),
    'corr': R(np.corrcoef(z.T), 2),
}
print('jepa', V['jepa']['h1'][::2], V['jepa']['h3'][::4], V['jepa']['table'], V['jepa']['formula'], V['jepa']['measured'])

json.dump(V, open('values.json', 'w'))
print('values.json', len(json.dumps(V)))
