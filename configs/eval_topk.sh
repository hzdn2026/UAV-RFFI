#!/bin/bash
# Reproduce the best cross-burst accuracy via source-validation top-k model
# selection + label-free test-time adaptation (TENT). Requires the released
# checkpoints under RFFI_CKPT and the cyclic-SCF h5 under RFFI_H5.
set -e
export RFFI_H5=/path/to/data/h5
export RFFI_CKPT=/path/to/checkpoints
python eval_topk.py --data data1 --topk 18   # 6 ft   -> ~94.36% (SOTA 93.05)
python eval_topk.py --data data2 --topk 9    # 9 ft   -> ~99.2%  (SOTA 98.90)
python eval_topk.py --data data3 --topk 10   # 12 ft  -> ~87%    (far-field UDA degrades)
