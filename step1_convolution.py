"""Step 1. Convolution: slide one small filter over a picture.

A filter is 9 numbers in a 3 x 3 square. Lay it on the picture, multiply each filter number by
the pixel under it, add the 9 products. That sum says how well this patch matches the filter.
Do it at every position and you get a new, smaller picture: a feature map.
"""
from data import load, show

VERTICAL = [[-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]]         # dark on the left, bright on the right: the left side of a stroke

HORIZONTAL = [[-1, -1, -1],
              [0, 0, 0],
              [1, 1, 1]]        # dark above, bright below: the top side of a stroke

PICK = 16                       # which training picture to look at


def window_sum(img, filt, row, col):
    """Filter's top-left corner at (row, col): multiply the 9 pairs and add them up."""
    total = 0.0
    for i in range(3):
        for j in range(3):
            total += filt[i][j] * img[row + i][col + j]
    return total


def convolve(img, filt):
    """Every position where the filter fits. An h x w picture gives (h - 2) x (w - 2) sums."""
    h, w = len(img), len(img[0])
    return [[window_sum(img, filt, r, c) for c in range(w - 2)] for r in range(h - 2)]


def page(img, top, left, size=12):
    """An empty size x size page with the 8 x 8 picture's corner at (top, left)."""
    out = [[0.0] * size for _ in range(size)]
    for r in range(8):
        for c in range(8):
            out[top + r][left + c] = img[r][c]
    return out


def strongest(fmap):
    """Row and column of the largest value in a feature map."""
    return max(((r, c) for r in range(len(fmap)) for c in range(len(fmap[0]))), key=lambda rc: fmap[rc[0]][rc[1]])


def positive(fmap):
    """For printing: keep the matches, drop the negative sums, scale to 0..1."""
    top = max(max(row) for row in fmap) or 1.0
    return [[max(v, 0) / top for v in row] for row in fmap]


def run():
    xtr, ytr, _, _ = load()
    img = xtr[PICK].reshape(8, 8).tolist()
    print(f'the picture: a {ytr[PICK]}, 8 x 8 pixels')
    show(img)

    r, c = 3, 0
    print(f'\nthe vertical filter laid at row {r}, column {c}:')
    for i in range(3):
        print('   ' + '   '.join(f'{VERTICAL[i][j]:+d} x {img[r + i][c + j]:.2f}' for j in range(3)))
    print(f'   sum = {window_sum(img, VERTICAL, r, c):+.2f}')

    v, h = convolve(img, VERTICAL), convolve(img, HORIZONTAL)
    print(f'\nat all {len(v)} x {len(v[0])} = {len(v) * len(v[0])} positions (vertical filter, matches only):')
    show(positive(v))
    print('horizontal filter:')
    show(positive(h))

    # move the digit on a bigger page: the feature map moves with it, nothing else changes
    a, b = convolve(page(img, 1, 1), VERTICAL), convolve(page(img, 3, 4), VERTICAL)
    same = all(abs(b[r + 2][c + 3] - a[r][c]) < 1e-12 for r in range(8) for c in range(7))
    print(f'\ndigit moved 3 right and 2 down on a 12 x 12 page:')
    print(f'   strongest match was at {strongest(a)}, is now at {strongest(b)}')
    print(f'   the whole feature map moved by the same 3 and 2: {same}')
    print(f'\none filter: 9 weights, used at {len(v) * len(v[0])} positions')
    return dict(img=img, label=int(ytr[PICK]), v=v, h=h, at=(r, c), at_sum=window_sum(img, VERTICAL, r, c),
                a=a, b=b, sa=strongest(a), sb=strongest(b), same=same)


if __name__ == '__main__':
    run()
