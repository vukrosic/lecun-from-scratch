"""Step 7. Energy-based model: when a question has more than one right answer.

The question: "which number, multiplied by itself, gives x?"  For x = 0.49 there are two
right answers: +0.7 and -0.7.

An ordinary network gives one output. Trained on both answers, it settles between them, near
0, which is wrong. An energy network takes the question AND a candidate answer and gives one
number, the energy: low means "these two fit together". To answer, try every candidate and
keep the ones with the lowest energy. Two valleys mean two answers.
"""
import numpy as np

from conv import Dense, Model, Tanh

MARGIN, STEPS, RATE, BATCH = 0.5, 12000, 0.03, 32
GRID = np.linspace(-1.2, 1.2, 241)                 # the candidate answers we try


def data(rng, n=2000):
    """Questions x between 0.05 and 1, each with one of its two answers, plus a little noise."""
    x = rng.uniform(0.05, 1, n)
    y = np.sqrt(x) * rng.choice([-1, 1], n) + rng.normal(0, 0.02, n)
    return x, y


def train_ordinary(x, y, rng):
    """One number in, one number out, squared error."""
    net = Model([Dense(1, 32, rng), Tanh(), Dense(32, 32, rng), Tanh(), Dense(32, 1, rng)])
    for _ in range(4000):
        i = rng.integers(0, len(x), BATCH)
        out = net.forward(x[i, None])
        net.backward(2 * (out - y[i, None]) / BATCH)
        net.step(0.05)
    return net


def train_energy(x, y, rng):
    """Two numbers in (question, candidate), one number out (energy).
    Push the energy of a right answer down and of a wrong answer up, until the right one is
    at least MARGIN lower. Then stop pushing that pair."""
    net = Model([Dense(2, 32, rng), Tanh(), Dense(32, 32, rng), Tanh(), Dense(32, 1, rng)])
    for _ in range(STEPS):
        i = rng.integers(0, len(x), BATCH)
        wrong = rng.uniform(-1.2, 1.2, BATCH)                               # a random candidate
        really_wrong = np.abs(np.abs(wrong) - np.sqrt(x[i])) > 0.1          # skip the lucky guesses
        pairs = np.concatenate([np.stack([x[i], y[i]], axis=1), np.stack([x[i], wrong], axis=1)])
        e = net.forward(pairs)[:, 0]
        too_close = ((MARGIN + e[:BATCH] - e[BATCH:]) > 0) & really_wrong
        net.backward(np.concatenate([too_close, -1.0 * too_close])[:, None] / BATCH)
        net.step(RATE)
    return net


def landscape(net, question):
    """Energy of every candidate answer for one question."""
    return net.forward(np.stack([np.full(len(GRID), question), GRID], axis=1))[:, 0]


def valleys(energy):
    """Candidates whose energy is lower than both neighbours."""
    return [float(GRID[k]) for k in range(1, len(GRID) - 1) if energy[k] < energy[k - 1] and energy[k] <= energy[k + 1]]


def run():
    rng = np.random.default_rng(0)
    x, y = data(rng)
    ordinary = train_ordinary(x, y, rng)
    energy = train_energy(x, y, np.random.default_rng(0))

    questions = np.linspace(0.1, 1, 10)
    said = ordinary.forward(questions[:, None])[:, 0]
    found = [valleys(landscape(energy, q)) for q in questions]
    print(f'{"x":>5s} {"right answers":>16s} {"ordinary network":>17s}   energy network: valleys at')
    for q, s, v in zip(questions, said, found):
        print(f'{q:5.1f} {"+-%.2f" % np.sqrt(q):>16s} {s:+17.2f}   {"  ".join("%+.2f" % a for a in v)}')
    miss_ord = float(np.mean(np.abs(np.abs(said) - np.sqrt(questions))))
    miss_en = float(np.mean([abs(abs(a) - np.sqrt(q)) for q, v in zip(questions, found) for a in v]))
    print(f'\naverage distance from a right answer: ordinary {miss_ord:.2f}, energy {miss_en:.2f}')
    print(f'questions with exactly two valleys: {sum(len(v) == 2 for v in found)} of {len(questions)}')
    surface = np.array([landscape(energy, q) for q in np.linspace(0.05, 1, 20)])
    return dict(x=x, y=y, questions=questions, said=said, found=found, miss_ord=miss_ord, miss_en=miss_en,
                curve=landscape(energy, 0.5), curve_x=0.5, surface=surface,
                line=ordinary.forward(np.linspace(0.05, 1, 20)[:, None])[:, 0])


if __name__ == '__main__':
    run()
