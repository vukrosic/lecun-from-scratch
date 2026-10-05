"""Step 2. Shared weights: let the network find the filter itself.

In step 1 we wrote the filter by hand. Here the filter starts as 9 random numbers and learns.
A second filter, kept secret, marks 40 pictures. The learner only sees the pictures and the
marked feature maps, and has to end up with the same 9 numbers.

The one new idea: the same weight is used at all 36 positions, so its blame is the sum of the
blame from all 36 positions.
"""
import random

from data import load
from step1_convolution import convolve, VERTICAL

SECRET = VERTICAL
N_PICTURES = 40
RATE = 0.0003                   # how far to move each weight against its blame
STEPS = 200
SHOW = (1, 2, 5, 10, 20, 50, 100, STEPS)


def error(filt, pictures, targets):
    """Sum of squared differences between the filter's feature maps and the target maps."""
    total = 0.0
    for img, target in zip(pictures, targets):
        out = convolve(img, filt)
        total += sum((out[r][c] - target[r][c]) ** 2 for r in range(6) for c in range(6))
    return total


def blame(filt, pictures, targets):
    """For each of the 9 weights: how much the error grows if that weight grows a little."""
    g = [[0.0] * 3 for _ in range(3)]
    for img, target in zip(pictures, targets):
        out = convolve(img, filt)
        for r in range(6):                      # every position where the filter was laid
            for c in range(6):
                miss = 2 * (out[r][c] - target[r][c])
                for i in range(3):
                    for j in range(3):
                        g[i][j] += miss * img[r + i][c + j]     # the pixel this weight touched here
    return g


def run():
    xtr, _, _, _ = load()
    pictures = [x.reshape(8, 8).tolist() for x in xtr[:N_PICTURES]]
    targets = [convolve(img, SECRET) for img in pictures]

    rng = random.Random(0)
    filt = [[rng.uniform(-0.5, 0.5) for _ in range(3)] for _ in range(3)]
    start = [row[:] for row in filt]
    print('the filter starts as 9 random numbers:')
    for row in filt:
        print('   ' + '  '.join(f'{v:+.2f}' for v in row))

    # check the blame formula the slow way: nudge one weight and measure the error again
    g, base, nudge = blame(filt, pictures, targets), error(filt, pictures, targets), 1e-6
    filt[0][2] += nudge
    measured = (error(filt, pictures, targets) - base) / nudge
    filt[0][2] -= nudge
    print(f'\nblame for the top-right weight, by the formula: {g[0][2]:.3f}')
    print(f'blame for the top-right weight, by nudging it : {measured:.3f}')

    history, snaps = [base], {0: [row[:] for row in filt]}
    for step in range(1, STEPS + 1):
        g = blame(filt, pictures, targets)
        for i in range(3):
            for j in range(3):
                filt[i][j] -= RATE * g[i][j]
        history.append(error(filt, pictures, targets))
        if step in SHOW:
            snaps[step] = [row[:] for row in filt]
            print(f'step {step:3d}: error {history[-1]:10.4f}')
    print('\nthe filter it found:')
    for row in filt:
        print('   ' + '  '.join(f'{v:+.2f}' for v in row))
    print('the secret filter:')
    for row in SECRET:
        print('   ' + '  '.join(f'{v:+.2f}' for v in row))

    print('\nweights needed to turn 64 pixels into a 36-number feature map:')
    print(f'   every pixel wired to every output: 64 x 36 = {64 * 36:,}')
    print('   one shared filter               : 9')
    return dict(start=start, filt=filt, history=history, snaps=snaps, formula=blame(start, pictures, targets)[0][2],
                measured=measured, base=base)


if __name__ == '__main__':
    run()
