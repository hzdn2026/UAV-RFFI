# Data

This repository uses the GENESYS "Hovering UAVs" RF fingerprinting dataset
(Cai et al., IEEE TIFS/TVT 2025/2026; Soltani et al., IEEE TVT 2020), collected in
an anechoic chamber with 7 DJI M100 UAVs at distances 6, 9, 12 ft. Each distance
has 4 bursts captured by pausing ~10 s between I/Q captures. This defines the
cross-burst evaluation protocol: train on bursts 1-3, test on burst 4.

## Option A - precomputed cyclic-SCF features (recommended)
Download the files ft6/ft9/ft12 _burst123_cyclic.h5 and ft*_burst4_cyclic.h5 from
the GitHub Release / Zenodo link noted in the main README, and place them under
data/h5/. Each file holds a dataset named "data_spectral" of shape (N, 4, 51, 23)
float32 (4 channels = [|S_nc|, coh_nc, |S_c|, coh_c]) plus a "label" array.

## Option B - regenerate from raw I/Q
1. Obtain the raw float16 I/Q recordings (one file per UAV per burst) and put them
   in data/raw/ (or set the RFFI_RAW environment variable).
2. Provide the filename-to-label map as data/uav_hos_all.npz (or set RFFI_NPZ); this
   mirrors preprocess_fusion.py so segments align 1:1 with the published protocol.
3. Run:
   python extract_cyclic.py data1   # -> data/h5/ft6_burst123_cyclic.h5 + ft6_burst4_cyclic.h5
   python extract_cyclic.py data2   # -> ft9_*
   python extract_cyclic.py data3   # -> ft12_*

The cyclic-SCF representation is computed with the FAM (FFT Accumulation Method)
estimator; the best model uses only the conjugate branch channels [2,3]
(|S_c|, coh_c), which is phase-rotation invariant by construction.
