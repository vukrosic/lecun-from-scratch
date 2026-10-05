# LeCun from scratch

Code for the video **Become Yann LeCun in 27 Minutes — Learn Everything He Knew, From Scratch**.
Eight of Yann LeCun's ideas, each rebuilt as a small program you can read in one sitting.
Python 3.9+ and numpy. No deep-learning library: the convolution layer and its backward pass are in `conv.py`.

```bash
python3 tests.py      # checks every step, about 3 seconds
./run_all.sh          # runs all eight, about 30 seconds; expected output is in demo-output.txt
```

| File | Idea | Year | What the run shows |
| --- | --- | --- | --- |
| `step1_convolution.py` | Convolution | 1989 | One 3 × 3 filter slid over a digit by hand: 9 weights used at 36 positions. Move the digit and the feature map moves with it. |
| `step2_shared_weights.py` | Shared weights learn | 1989 | A filter that starts as 9 random numbers recovers a secret filter from 40 examples. Error 4,274 → 0.0019. The blame for a shared weight is the sum over every position. |
| `step3_lenet.py` | LeNet | 1989 / 1998 | Digits at random places on a page. Filters: 84.4 % with 2,390 weights. Every pixel wired separately: 46.0 % at the same size, 60.6 % with 8 times more weights. |
| `step4_efficient_backprop.py` | Efficient BackProp | 1998 | Same network, same learning rate. Raw inputs and size-1 weights: 96.2 % of neurons flat, 31.2 % after one epoch. Centred inputs and 1/sqrt(inputs) weights: 1.4 % flat, 89.6 % after one epoch. |
| `step5_brain_damage.py` | Optimal Brain Damage | 1989 | 4,928 connections cut to 247 (95 % deleted), accuracy 96.4 % → 93.6 %. Deleting at random: 12.8 % at 90 %. |
| `step6_siamese.py` | Siamese network | 1993 / 2006 | Trained only on "same" or "different". Same-digit check on unseen pictures: 94.1 % (raw pixels: 80.1 %). One named example per digit: 86.7 % (raw pixels: 64.9 %). |
| `step7_energy.py` | Energy-based model | 2006 | A question with two right answers. An ordinary network answers near 0 (off by 0.67). The energy network finds both answers for 10 of 10 questions (off by 0.02). |
| `step8_joint_embedding.py` | Joint embedding (VICReg) | 2022 | 1,297 pictures with no labels. Agreement alone collapses (spread 0.13 → 0.0009). With the variance and covariance rules, 10 labels are enough for 64.0 % and 100 labels for 82.3 % (the same network trained on labels only: 15.1 % and 35.8 %). |

`conv.py` holds the layers (convolution, pooling, ReLU, tanh, dense) with forward and backward passes; steps 3 to 8 use it.
`data.py` loads the digits. `values.py` saves the numbers and pictures the video draws.

## Exercise

`step3_lenet.py` makes the page 12 × 12. Change `SIZE = 12` to `SIZE = 10` (less room to move), run it,
and see how much of the gap between the filter network and the fully wired one is left.

## What is simplified

- **Data.** 1,797 digits of 8 × 8 pixels, so every step runs in seconds on a laptop. LeCun's 1989 network read 16 × 16 zip-code digits; LeNet-5 read 28 × 28 MNIST digits.
- **Step 2.** The filter copies a known filter. In the 1989 network the filters learned from the digit labels, which is what step 3 does.
- **Step 3.** Two convolution layers with 6 and 16 filters of 3 × 3, max pooling and ReLU. LeNet-5 used 5 × 5 filters, average pooling and tanh-like units, and had about 60,000 weights.
- **Step 4.** Two of the paper's recommendations (centred inputs, weight scale). The paper has more: shuffling, a scaled tanh, per-weight learning rates.
- **Step 5.** One hidden layer. "Delete the smallest weights first" does as well as the curvature formula on this network; the output prints both.
- **Step 6.** The twin network is two dense layers and its output is 2 numbers so it can be drawn. The 1993 system compared signatures over time.
- **Step 7.** A one-dimensional toy problem. Wrong answers are drawn at random; the 2006 tutorial covers many other ways to push energy up.
- **Step 8.** The only change between the two views is the position on the page. VICReg used crops, colour changes and blur on ImageNet photos, and weights 25 / 25 / 1. A full JEPA also hides part of the input and predicts its features; this step shows only the collapse problem and its fix.

## Data

`digits.csv.gz` is the test part of the UCI "Optical Recognition of Handwritten Digits" data set
(E. Alpaydin and C. Kaynak, 1998, CC BY 4.0), as shipped with scikit-learn.

## Licence

MIT for the code.
