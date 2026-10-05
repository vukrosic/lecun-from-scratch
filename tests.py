"""Quick checks for every step. Run: python3 tests.py  (about 3 seconds)"""
import numpy as np

from conv import Conv, Dense, Flat, Model, Pool, Relu, lenet, onehot, softmax
from step1_convolution import VERTICAL, convolve, page, window_sum
from step2_shared_weights import blame, error
from step5_brain_damage import masks_for
from step7_energy import GRID, valleys
from step8_joint_embedding import rules

rng = np.random.default_rng(0)

# step 1: a flat patch gives 0, an edge gives 3, and moving the picture moves the feature map
assert window_sum([[1, 1, 1]] * 3, VERTICAL, 0, 0) == 0
assert window_sum([[0, 0, 1]] * 3, VERTICAL, 0, 0) == 3
img = rng.random((8, 8)).tolist()
a, b = convolve(page(img, 0, 0), VERTICAL), convolve(page(img, 2, 3), VERTICAL)
assert len(a) == 10 and len(a[0]) == 10
assert all(abs(a[r][c] - b[r + 2][c + 3]) < 1e-12 for r in range(8) for c in range(7))

# step 2: the blame formula agrees with nudging a weight
pictures = [rng.random((8, 8)).tolist() for _ in range(3)]
targets = [convolve(p, VERTICAL) for p in pictures]
filt = [[0.1, -0.2, 0.3], [0.0, 0.4, -0.1], [0.2, 0.1, -0.3]]
g, base = blame(filt, pictures, targets), error(filt, pictures, targets)
filt[1][2] += 1e-6
assert abs((error(filt, pictures, targets) - base) / 1e-6 - g[1][2]) < 1e-3

# conv.py: backward() agrees with nudging, for a conv weight, a later conv weight and a dense weight
net, x, t = lenet(np.random.default_rng(1)), rng.random((4, 1, 12, 12)), onehot(np.array([1, 3, 5, 7]))
loss = lambda: -np.mean(np.sum(t * np.log(softmax(net.forward(x))), axis=1))
net.backward((softmax(net.forward(x)) - t) / len(x))
for layer, idx in ((0, (4, 2)), (3, (10, 5)), (6, (20, 3))):
    said, W = net.layers[layer].gW[idx], net.layers[layer].W
    W[idx] += 1e-5
    up = loss()
    W[idx] -= 2e-5
    down = loss()
    W[idx] += 1e-5
    assert abs(said - (up - down) / 2e-5) < 1e-6, (layer, said)
assert net.n_weights() == 2390

# pooling keeps the largest of each 2 x 2 block and sends blame only to it
pool = Pool()
out = pool.forward(np.array([[[[1., 2.], [5., 3.]]]]))
assert out.tolist() == [[[[5.0]]]]
assert pool.backward(np.array([[[[7.]]]])).tolist() == [[[[0.0, 0.0], [7.0, 0.0]]]]

# step 5: the lowest scores are deleted, and deleted weights stay deleted
masks = masks_for(np.array([5., 1., 4., 2., 3., 6.]), np.array([1, 1, 0, 1, 1, 1], bool), 0.5, [(2, 2), (2, 1)])
assert masks[0].tolist() == [[1, 0], [0, 0]] and masks[1].tolist() == [[1], [1]]

# step 7: a W-shaped curve has two valleys
assert len(valleys((GRID ** 2 - 0.5) ** 2)) == 2

# step 8: the three-rule blame agrees with nudging, and identical outputs break rule 2
za, zb = rng.normal(0, 0.7, (16, 5)), rng.normal(0, 0.7, (16, 5))
err, ba, bb = rules(za, zb, 25.0, 1.0)
zb[7, 1] += 1e-5
assert abs((rules(za, zb, 25.0, 1.0)[0] - err) / 1e-5 - bb[7, 1]) < 1e-3
same = np.ones((16, 5))
assert rules(same, same, 1.0, 0.0)[0] > 1.9          # collapsed: agreement is perfect, spread is 0

print('all tests pass')
