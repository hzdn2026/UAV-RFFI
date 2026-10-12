"""循环谱 (Spectral Correlation Function, SCF) 特征提取，作为 RFFI 谱分支的新表示。

背景: 原谱分支是 log-幅度 2D-DWT（相位/高阶全丢），循环谱则刻画信号的**周期平稳性**
——由发射机硬件损伤(CFO、IQ失衡、PA非线性、符号速率)引入的、在循环频率 alpha 上
的重复结构。这些是设备相关、且对信道/传播不敏感的指纹，正是 log-幅度谱缺的那块。

实现: FFT Accumulation / Strip 法估计 SCF。
  X[p,k] = FFT_L(w * strip_p)            # 通道化短时谱 (P 条, L 频点)
  S_nc[m,k] = mean_p X[k+m] * conj(X[k]) # 非共轭循环谱 (符号速率等)
  S_c [m,k] = mean_p X[k+m] * X[k]       # 共轭循环谱 (CFO/IQ失衡等)
  alpha_m = m * fs/L ;  m=-M..M

输出 h5 含:
  data_spectral (M,4,51,23) float32   # 4 通道 = [|S_nc|, coh_nc, |S_c|, coh_c]
  label        (M,)        int64
与 preprocess_fusion.py 完全一致的切分(SEG=5888, 非重叠, uav1..7->burst->sorted) +
复刻 npz 跳过逻辑，保证与 *_fusion.h5 的 data_phys(HOS) 逐段对齐，可 1:1 融合。

通道说明:
  |S_nc|  非共轭循环谱幅度 (含 alpha=0 的 PSD)
  coh_nc  SCF 相干系数 = |S_nc|^2 / |S(0)|^4 (抑制 PSD 主导, 凸显循环结构)
  |S_c|   共轭循环谱幅度 (CFO/IQ失衡指纹)
  coh_c   共轭相干系数
4 通道正好匹配 FusionNet 的 Conv2d(4,...)。

用法:
  python3 extract_cyclic.py --selftest          # 合成信号自检(验证 FAM 正确)
  python3 extract_cyclic.py data1               # 生成 ft6 burst123/4 循环谱 h5
  python3 extract_cyclic.py data2               # 生成 ft9 burst123/4 循环谱 h5
"""
import os, glob, h5py, sys, argparse
import numpy as np
from scipy import signal

SRC = os.environ.get('RFFI_RAW', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'raw'))
OUT = os.environ.get('RFFI_H5', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'h5'))
NPZ = os.environ.get('RFFI_NPZ', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'uav_hos_all.npz'))
SEG = 5888
FS = 10e6
L = 256          # FAM strip length (freq resolution = fs/L ≈ 39 kHz)
M = 26           # cyclic lag half-range -> alpha in [-M, M]*fs/L ≈ ±1.0 MHz, 53 rows


def read_iq(path):
    raw = np.fromfile(path, dtype=np.float16)
    if raw.shape[0] % 2 != 0:
        raw = raw[:raw.shape[0] - 1]
    return raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)


def norm_rms(x):
    rms = np.sqrt(np.mean(np.abs(x) ** 2))
    return x if rms <= 0 else x / rms


def cyclic_scf(seg, L=L, M=M):
    """返回 (S_nc, S_c)，各形状 (2M+1, L) complex。"""
    N = len(seg)
    P = N // L
    x = seg[:P * L]
    w = np.hamming(L)
    X = np.fft.fft((x.reshape(P, L)) * w[None, :], axis=1)          # (P, L)
    Xsh = np.stack([np.roll(X, -m, axis=1) for m in range(-M, M + 1)], axis=0)  # (53,P,L)
    Snc = np.mean(Xsh * np.conj(X[None]), axis=1)                   # 非共轭
    Sc = np.mean(Xsh * X[None], axis=1)                             # 共轭
    return Snc, Sc


def to_channels(Snc, Sc):
    """(53,L) complex -> (4,51,23) float32。"""
    Snc = Snc[1:-1]                                                # 丢 alpha=±M 边行 -> 51
    Sc = Sc[1:-1]
    fidx = np.linspace(13, 242, 23).astype(int)                    # 跳带边避免 wrap 伪影
    Snc = Snc[:, fidx]                                             # (51,23)
    Sc = Sc[:, fidx]
    m0 = 25                                                        # 原 index M=26, 丢首行后在新位置 25
    S0 = np.abs(Snc[m0]) ** 2 + 1e-12
    coh_nc = np.clip(np.abs(Snc) ** 2 / (S0[None, :] ** 2 + 1e-12), 0, 1)
    coh_c = np.clip(np.abs(Sc) ** 2 / (S0[None, :] ** 2 + 1e-12), 0, 1)
    mag_nc = np.abs(Snc)
    mag_c = np.abs(Sc)
    out = np.stack([mag_nc, coh_nc, mag_c, coh_c], axis=0).astype(np.float32)
    return out


def extract_file(fp, phys_keys):
    bn = os.path.basename(fp)
    if bn not in phys_keys:          # 复刻 preprocess_fusion 的 npz 跳过, 保证对齐
        return None
    iq = read_iq(fp)
    n = len(iq)
    if n < SEG:
        return None
    feats = []
    for s in range(0, n - SEG + 1, SEG):
        seg = norm_rms(iq[s:s + SEG])
        snc, sc = cyclic_scf(seg)
        feats.append(to_channels(snc, sc))
    return np.array(feats, dtype=np.float32)    # (nseg,4,51,23)


def selftest():
    """合成方波门控复信号: x = e^{j2πfc t} * g(t), g 为 fm 周期方波(50%占空)。
    周期门控在 alpha=±fm, ±2fm... 产生强循环平稳峰(非共轭)。
    取 fm = 3*fs/L (精确落在整数 FFT bin 3), 避免半 bin 分裂, 期望峰在 m=±3。"""
    fs = 10e6
    N = 5888
    df = fs / L
    fc = 20 * df                                  # 精确落在整数 bin 20, 避免半 bin 泄漏
    fm = 3 * df                                   # 精确 bin 3, 期望峰在 m=±3
    t = np.arange(N) / fs
    q = (np.sin(2 * np.pi * fm * t) > 0).astype(float) * 2 - 1   # 方波 ±1
    x = np.exp(2j * np.pi * fc * t) * (0.5 + 0.5 * q)            # 周期门控
    snc, sc = cyclic_scf(x)
    alpha = np.arange(-M, M + 1) * df
    mag = np.abs(snc)
    mag0 = mag.copy(); mag0[M] = 0.0              # 屏蔽 alpha=0 (PSD)
    flat = mag0.ravel().argsort()[::-1][:6]
    print('[selftest] top non-conj off-alpha peaks (m, alpha_Hz, |S|, freqbin):')
    for idx in flat:
        mi, ki = np.unravel_index(idx, mag0.shape)
        print(f'   m={mi - M: d}  alpha={alpha[mi]:.4e}  |S|={mag0[mi, ki]:.3e}  fbin={ki}')
    peak_m = flat[0] // mag0.shape[1]
    peak_alpha = alpha[peak_m]
    ok = abs(abs(peak_alpha) - fm) < 0.6 * df
    print(f'[selftest] peak alpha={peak_alpha:.4e} Hz, expected ~{fm:.4e} ({3}*df)')
    print('[selftest]', 'PASS' if ok else 'FAIL')
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('which', nargs='?', choices=['data1', 'data2', 'data3'])
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if not args.which:
        print('usage: extract_cyclic.py [--selftest] {data1,data2,data3}')
        return

    npz = np.load(NPZ, allow_pickle=True)
    phys_keys = set(str(x) for x in npz['filenames'])
    print(f'[phys] loaded {len(phys_keys)} filenames from npz (skip-set for alignment)', flush=True)

    dist = {'data1': '6ft', 'data2': '9ft', 'data3': '12ft'}[args.which]
    for bursts, tag in [([1, 2, 3], '123'), ([4], '4')]:
        sp, ll = [], []
        cnt_files = 0
        for uav in range(1, 8):
            for b in bursts:
                for fp in sorted(glob.glob(f'{SRC}/uav{uav}_{dist}_burst{b}_*.bin')):
                    feats = extract_file(fp, phys_keys)
                    if feats is None:
                        continue
                    cnt_files += 1
                    for fv in feats:
                        sp.append(fv)
                        ll.append(uav - 1)
        sp = np.array(sp, dtype=np.float32)
        ll = np.array(ll, dtype=np.int64)
        os.makedirs(OUT, exist_ok=True)
        out = f'{OUT}/ft{dist.replace("ft", "")}_burst{tag}_cyclic'
        with h5py.File(out + '.h5', 'w') as hf:
            hf.create_dataset('data_spectral', data=sp)
            hf.create_dataset('label', data=ll)
        # 对齐校验: 与 fusion h5 的 data_spectral 段数对得上?
        align = ''
        fpath = f'{OUT}/ft{dist.replace("ft", "")}_burst{tag}_fusion.h5'
        if os.path.exists(fpath):
            with h5py.File(fpath, 'r') as hf:
                n_fus = hf['data_spectral'].shape[0]
            align = f'  ALIGN fusion.data_spectral={n_fus} -> {"OK" if n_fus == sp.shape[0] else "MISMATCH!"}'
        print(f'[DONE] {out}: spectral={sp.shape} label={ll.shape} files={cnt_files}{align}', flush=True)


if __name__ == '__main__':
    main()
