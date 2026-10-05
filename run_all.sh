#!/bin/sh
# Run every step in video order and print what each one finds (about 30 s).
cd "$(dirname "$0")" || exit 1
for f in step1_convolution step2_shared_weights step3_lenet step4_efficient_backprop step5_brain_damage step6_siamese step7_energy step8_joint_embedding; do
  echo "=== $f.py"
  python3 "$f.py" || exit 1
  echo
done
