# UAV-RFFI-CDAN-TTA

Best-accuracy pipeline for **cross-burst RF fingerprint identification of hovering
UAVs**, built entirely from public code and the public GENESYS "Hovering UAVs"
dataset. The method beats the published 2D-DWT + ASA state of the art on the
near-field (6 ft and 9 ft) cross-burst benchmarks **without using the paper's
unpublished ASA alignment**.

## Method

1. **Conjugate cyclic-SCF features.** Raw I/Q is converted to a 4-channel
   spectral-correlation representation (|S_nc|, coh_nc, |S_c|, coh_c) of shape
   (4, 51, 23) via the FAM estimator (`extract_cyclic.py`). The best model uses
   only the conjugate branch `[2,3]` (|S_c|, coh_c), which is invariant to
   transmitter phase rotation by construction.
2. **Conditional adversarial domain adaptation (CDAN).** A small CNN
   (`Feat` in `cdan_model.py`) is trained on bursts 1-3 (source) against burst 4
   (target) with a CDAN discriminator and class-conditional entropy weighting.
3. **Label-free test-time adaptation (TTA).** At inference on burst 4 we run TENT
   (entropy + diversity minimisation) using **no target labels**.
4. **Source-validation top-k selection.** Across a pool of CDAN checkpoints we
   keep the top-k models ranked by their *source-domain* best validation accuracy
   and soft-vote their TTA predictions. This exploits cross-seed model diversity
   and is fully reproducible (no test labels are used for selection).

> Honesty note: TTA adapts to the *test distribution* (burst 4) without labels;
> model selection uses only *source-domain* validation accuracy. No test labels
> are ever used.

## Reproduced results (cross-burst: bursts 1-3 train, burst 4 test)

| Distance | This work (acc / macro-F1) | Published SOTA (2D-DWT + ASA) |
|----------|----------------------------|-------------------------------|
| 6 ft (d1) | **94.36%** / **94.28%** (top-18 val sel.) | 93.05% |
| 9 ft (d2) | **99.35%** / **99.37%** (top-9 val sel.)  | 98.90% |

Both distances were re-verified end-to-end from the released checkpoints on the
reproduction server; logs: `results/repro_data1.log`, `results/repro_data2.log`.

## Install

    pip install -r requirements.txt

(Tested: torch 2.6.0+cu124, numpy 2.2.6, scipy 1.15.3, scikit-learn 1.7.2,
h5py 3.16.0, Python 3.10.)

## Data

See [data/README.md](data/README.md). In short: download the precomputed
`ft*_cyclic.h5` features (recommended) or regenerate them from raw I/Q with
`extract_cyclic.py`. Point the code at them with the `RFFI_H5` environment
variable.

## Usage

Feature extraction (optional, if regenerating):

    python extract_cyclic.py data1   # 6 ft
    python extract_cyclic.py data2   # 9 ft

Training (single model; sweep seeds/domain-weights for the ensemble pool):

    python cdan_model.py --data data1 --method cdan --seed 123 --dom_weight 1.5 \
        --ch 2,3 --cyc_tag cyclic --da_ramp --save_ckpt --gpus 4

Evaluation (top-k source-val selection + TTA; this prints the headline number):

    python eval_topk.py --data data1 --topk 18

## Repository structure

    cdan_model.py      CDAN model definitions + training entry (Feat, Classifier, load_spectral)
    extract_cyclic.py  conjugate-SCF feature extraction (FAM) from raw I/Q
    tta.py             label-free test-time adaptation (TENT / SHOT)
    eval_topk.py       source-val top-k model selection + accuracy report
    configs/           example training / evaluation shell scripts
    data/              data acquisition notes (large artifacts are git-ignored)
    checkpoints/       released model pool (git-ignored; download separately)

## License

MIT. The underlying GENESYS dataset is subject to its own license; cite the
original authors when using it.

## Citation

[to be filled with the camera-ready reference]
