#!/bin/bash
# Train a single CDAN model (conjugate-SCF branch, channels 2,3).
# The released checkpoints were produced by a sweep over
#   seeds in {123, 2024, 32, 777, 2025, 11, 22, 44, 55, ...}
#   domain weights in {1.0, 1.5, 2.0, 2.5, 3.0}
# with --method cdan --ch 2,3 --cyc_tag cyclic --da_ramp --save_ckpt.
# Below is one representative command (data1 / 6 ft).
set -e
export RFFI_H5=/path/to/data/h5
python cdan_model.py --data data1 --method cdan --seed 123 --dom_weight 1.5 \
    --ch 2,3 --cyc_tag cyclic --da_ramp --save_ckpt --gpus 4 --epochs 200
