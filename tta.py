import os, sys, glob, argparse, copy
import numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cdan_model import Feat, Classifier, load_spectral, H5

SEL = [2, 3]; CYCTAG = 'cyclic'
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import random as _random
_SEED = 20240811
torch.manual_seed(_SEED); np.random.seed(_SEED); _random.seed(_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
def _wk_init(w):
    torch.manual_seed(_SEED + w); np.random.seed(_SEED + w)
_gen = torch.Generator(); _gen.manual_seed(_SEED)
FT_MAP = {'data1': '6', 'data2': '9', 'data3': '12'}
FT_LABEL = {'data1': '6ft', 'data2': '9ft', 'data3': '12ft'}


def build_prep(seed, data):
    ft = FT_MAP[data]
    c_tr, ll_tr = load_spectral(f'{H5}/ft{ft}_burst123_{CYCTAG}.h5')
    c_te, ll_te = load_spectral(f'{H5}/ft{ft}_burst4_{CYCTAG}.h5')
    X_all = c_tr[:, SEL]; Xt = c_te[:, SEL]
    itr, _ = train_test_split(np.arange(len(ll_tr)), test_size=0.2,
                              random_state=seed, stratify=ll_tr)
    Xs_tr = X_all[itr]
    mu, sd = Xs_tr.mean(), Xs_tr.std() + 1e-8
    return (Xt - mu) / sd, ll_te


def load_model(ck):
    feat = Feat(in_ch=ck['in_dim'], hw=tuple(ck['hw'])).to(DEV)
    cls = Classifier(emb=256 if ck['fusion'] else 128).to(DEV)
    feat.load_state_dict(ck['feat_c']); cls.load_state_dict(ck['cls'])
    return feat, cls


def entropy_loss(logits):
    p = F.softmax(logits, 1)
    return -(p * torch.log(p + 1e-8)).sum(1).mean()


def diversity_loss(logits):
    p = F.softmax(logits, 1).mean(0)
    return (p * torch.log(p + 1e-8)).sum()


def tta_tent(feat, cls, Xt_t, steps=40, lr=1e-3):
    feat.train(); cls.train()
    opt = torch.optim.Adam(list(feat.parameters()) + list(cls.parameters()), lr=lr)
    dl = DataLoader(TensorDataset(Xt_t), batch_size=256, shuffle=True, generator=_gen, worker_init_fn=_wk_init)
    it = iter(dl)
    for _ in range(steps):
        try: x = next(it)[0]
        except StopIteration: it = iter(dl); x = next(it)[0]
        x = x.to(DEV)
        opt.zero_grad()
        loss = entropy_loss(cls(feat(x))) + diversity_loss(cls(feat(x)))
        loss.backward(); opt.step()
    feat.eval(); cls.eval(); return feat, cls


def tta_shot(feat, cls, Xt_t, steps=40, lr=1e-3, tau=0.9, lam_pl=1.0):
    # SHOT: entropy + diversity + confident pseudo-label CE
    feat.train(); cls.train()
    opt = torch.optim.Adam(list(feat.parameters()) + list(cls.parameters()), lr=lr)
    dl = DataLoader(TensorDataset(Xt_t), batch_size=256, shuffle=True, generator=_gen, worker_init_fn=_wk_init)
    it = iter(dl)
    for _ in range(steps):
        try: x = next(it)[0]
        except StopIteration: it = iter(dl); x = next(it)[0]
        x = x.to(DEV)
        e = feat(x); logits = cls(e); p = F.softmax(logits, 1)
        loss = entropy_loss(logits) + diversity_loss(logits)
        conf, pred = p.max(1)
        mask = conf > tau
        if mask.sum() > 0:
            loss = loss + lam_pl * F.cross_entropy(logits[mask], pred[mask])
        opt.zero_grad(); loss.backward(); opt.step()
    feat.eval(); cls.eval(); return feat, cls



def tta_cotta(feat, cls, Xt_t, steps=40, lr=1e-3, ema=0.999, tau=0.95,
              restore_prob=0.1, restore_blend=0.1, aug_k=8, strength=0.04,
              lam_pl=1.0, lam_ent=1.0):
    # CoTTA-style TTA: EMA teacher + confident pseudo-label CE + stochastic
    # restoration. Entropy minimization is kept always-on as a stability
    # fallback (the cross-burst source model is low-confidence at init, so a
    # hard confident-mask alone would produce zero-gradient steps).
    feat.train(); cls.train()
    teacher_feat = copy.deepcopy(feat).eval()
    teacher_cls = copy.deepcopy(cls).eval()
    sd0_feat = {n: p.clone() for n, p in feat.state_dict().items()}
    sd0_cls = {n: p.clone() for n, p in cls.state_dict().items()}
    opt = torch.optim.Adam(list(feat.parameters()) + list(cls.parameters()), lr=lr)
    dl = DataLoader(TensorDataset(Xt_t), batch_size=256, shuffle=True,
                    generator=_gen, worker_init_fn=_wk_init)
    it = iter(dl)
    for _ in range(steps):
        if torch.rand(1).item() < restore_prob:
            for n in sd0_feat:
                feat.state_dict()[n].data.copy_(
                    sd0_feat[n] * restore_blend + feat.state_dict()[n].data * (1 - restore_blend))
            for n in sd0_cls:
                cls.state_dict()[n].data.copy_(
                    sd0_cls[n] * restore_blend + cls.state_dict()[n].data * (1 - restore_blend))
        try:
            x = next(it)[0]
        except StopIteration:
            it = iter(dl); x = next(it)[0]
        x = x.to(DEV)
        augs = aug_copies(x, aug_k, strength)
        with torch.no_grad():
            t_soft = torch.stack([F.softmax(teacher_cls(teacher_feat(a)), 1) for a in augs]).mean(0)
            conf, pseudo = t_soft.max(1)
            mask = conf > tau
        s_logits = torch.cat([cls(feat(a)) for a in augs], 0)
        # entropy minimization: always-on, prevents collapse when pseudo-labels unreliable
        loss = lam_ent * (entropy_loss(s_logits) + diversity_loss(s_logits))
        if mask.sum() > 0:
            pseudo_r = pseudo.repeat(aug_k)
            conf_r = conf.repeat(aug_k)
            mask_r = mask.repeat(aug_k)
            loss = loss + lam_pl * (F.cross_entropy(s_logits[mask_r], pseudo_r[mask_r],
                                                     reduction='none') * conf_r[mask_r]).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            for pt, ps in zip(teacher_feat.parameters(), feat.parameters()):
                pt.data.mul_(ema).add_(ps.data, alpha=1 - ema)
            for pt, ps in zip(teacher_cls.parameters(), cls.parameters()):
                pt.data.mul_(ema).add_(ps.data, alpha=1 - ema)
    feat.eval(); cls.eval(); return feat, cls


def soft_of(feat, cls, Xt_t):
    dl = DataLoader(TensorDataset(Xt_t), batch_size=256, shuffle=False)
    outs = []
    with torch.no_grad():
        for (x,) in dl:
            outs.append(torch.softmax(cls(feat(x.to(DEV))), 1).cpu())
    return torch.cat(outs)


def aug_copies(Xt_t, k, strength):
    # 域随机化: 每通道独立 mild affine jitter (模拟 burst 内抖动)
    outs = []
    for _ in range(k):
        scale = 1.0 + strength * torch.randn(Xt_t.shape[1], 1, 1, device=DEV)
        shift = strength * torch.randn(Xt_t.shape[1], 1, 1, device=DEV)
        outs.append(Xt_t * scale + shift)
    return outs


def report(preds, name, ll_te):
    acc = 100 * (preds == ll_te).float().mean().item()
    f1 = 100 * f1_score(ll_te.numpy(), preds.numpy(), average='macro', zero_division=0)
    print(f'  [{name}] acc={acc:.2f}% macroF1={f1:.2f}%')
    return acc, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data1', choices=['data1', 'data2', 'data3'])
    ap.add_argument('--steps', type=int, default=40)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--k', type=int, default=8, help='aug copies for aug-invariant TTA')
    ap.add_argument('--strength', type=float, default=0.12)
    args = ap.parse_args()

    CKPT = os.environ.get('RFFI_CKPT', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints'))
    files = sorted(set(glob.glob(f'{CKPT}/ckpt_x_{args.data}_*.pth'))
                   | set(glob.glob(f'{CKPT}/ckpt_{args.data}_*.pth')))
    files = [f for f in files if '_resnet' not in os.path.basename(f)
             and '_wide' not in os.path.basename(f)]
    keep = []
    for f in files:
        bn = os.path.basename(f)
        if bn.startswith('ckpt_') and not bn.startswith('ckpt_x_'):
            keep.append(f)
        else:
            m = torch.load(f, map_location='cpu', weights_only=False).get('mtag')
            if m == 'cdan':
                keep.append(f)
    files = keep
    print(f'# {len(files)} cnn cdan checkpoints for {args.data}')

    # per-model tent soft votes + bestval
    tent_softs, bestvals = [], []
    shot_softs = []
    aug_softs = []  # per-model: mean over K augs of tent-adapted prediction
    ll_te = None
    for f in files:
        ck = torch.load(f, map_location='cpu', weights_only=False)
        Xt_n, ll_te = build_prep(int(ck['seed']), args.data)
        Xt_t = torch.tensor(Xt_n).to(DEV)
        feat, cls = load_model(ck)
        feat_t, cls_t = tta_tent(feat, cls, Xt_t, args.steps, args.lr)
        tent_softs.append(soft_of(feat_t, cls_t, Xt_t).cpu())
        bestvals.append(ck['bestval'])
        # shot variant
        feat2, cls2 = load_model(ck)
        feat_s, cls_s = tta_shot(feat2, cls2, Xt_t, args.steps, args.lr)
        shot_softs.append(soft_of(feat_s, cls_s, Xt_t).cpu())
        # aug-invariant: adapt once on original, average pred over K augs
        augs = aug_copies(Xt_t, args.k, args.strength)
        with torch.no_grad():
            pa = [soft_of(feat_t, cls_t, a) for a in augs]
        aug_softs.append(torch.stack(pa).mean(0).cpu())

    ll_te = torch.tensor(ll_te)
    tent_softs = torch.stack(tent_softs)        # (M,N,C)
    shot_softs = torch.stack(shot_softs)
    aug_softs = torch.stack(aug_softs)
    bestvals = torch.tensor(bestvals)

    lines = [f'# TTA boost (label-free) — {args.data}({FT_LABEL[args.data]}) CDAN CNN',
             '', f'steps={args.steps} lr={args.lr} k={args.k} strength={args.strength} M={len(files)}', '']
    lines.append('| 变体 | acc | macroF1 |')
    lines.append('|------|-----|---------|')
    summary = {}

    # baseline tent equal-weight
    a, fi = report(tent_softs.mean(0).argmax(1), 'tent_eq', ll_te)
    summary['tent_eq'] = (a, fi)
    lines.append(f'| tent 等权(基线) | {a:.2f}% | {fi:.2f}% |')

    # tent val-weighted
    w = F.softmax(bestvals * 30.0, 0)  # 放大差异
    tw = (tent_softs * w.view(-1, 1, 1)).sum(0)
    a, fi = report(tw.argmax(1), 'tent_valw', ll_te)
    summary['tent_valw'] = (a, fi)
    lines.append(f'| tent 按val加权 | {a:.2f}% | {fi:.2f}% |')

    # shot equal-weight
    a, fi = report(shot_softs.mean(0).argmax(1), 'shot_eq', ll_te)
    summary['shot_eq'] = (a, fi)
    lines.append(f'| shot 等权 | {a:.2f}% | {fi:.2f}% |')

    # aug-invariant tent
    a, fi = report(aug_softs.mean(0).argmax(1), 'tent_aug', ll_te)
    summary['tent_aug'] = (a, fi)
    lines.append(f'| tent+aug-invariant(k={args.k}) | {a:.2f}% | {fi:.2f}% |')

    # combined: val-weighted of aug-invariant
    twa = (aug_softs * w.view(-1, 1, 1)).sum(0)
    a, fi = report(twa.argmax(1), 'tent_aug_valw', ll_te)
    summary['tent_aug_valw'] = (a, fi)
    lines.append(f'| tent+aug 按val加权 | {a:.2f}% | {fi:.2f}% |')
    # combined: tent + shot average (both ~93.1, complementary errors)
    comb = (tent_softs + shot_softs) / 2.0
    a, fi = report(comb.mean(0).argmax(1), 'tent+shot_eq', ll_te)
    summary['tent+shot_eq'] = (a, fi)
    lines.append(f'| tent+shot 平均 | {a:.2f}% | {fi:.2f}% |')
    # tent+shot val-weighted
    twc = (comb * w.view(-1, 1, 1)).sum(0)
    a, fi = report(twc.argmax(1), 'tent+shot_valw', ll_te)
    summary['tent+shot_valw'] = (a, fi)
    lines.append(f'| tent+shot 按val加权 | {a:.2f}% | {fi:.2f}% |')

    lines.append('')
    lines.append('## 对照: SOTA(2D-DWT+ASA)=93.05 (d1)')
    out = f'{args.data}_tta_report.md'
    with open(out, 'w') as fo:
        fo.write('\n'.join(lines) + '\n')
    print(f'WROTE {out}')
    print('SUMMARY', {m: (round(x, 2), round(y, 2)) for m, (x, y) in summary.items()})


if __name__ == '__main__':
    main()

def tta_bnm(feat, cls, Xt_t, steps=40, lr=1e-3, bnm_weight=0.1):
    # BNM (Batch Nuclear Norm Maximization) TTA: entropy+diversity minimization
    # plus maximization of the nuclear norm of the target feature covariance,
    # which encourages diverse, well-structured predictions. More robust than
    # pure entropy minimization when the target set is small / class-imbalanced.
    feat.train(); cls.train()
    opt = torch.optim.Adam(list(feat.parameters()) + list(cls.parameters()), lr=lr)
    dl = DataLoader(TensorDataset(Xt_t), batch_size=256, shuffle=True, generator=_gen, worker_init_fn=_wk_init)
    it = iter(dl)
    for _ in range(steps):
        try:
            x = next(it)[0]
        except StopIteration:
            it = iter(dl); x = next(it)[0]
        x = x.to(DEV)
        e = feat(x); logits = cls(e)
        loss = entropy_loss(logits) + diversity_loss(logits)
        fe = e - e.mean(0, keepdim=True)
        cov = (fe.T @ fe) / (fe.size(0) - 1)
        bnm = -torch.norm(cov, "nuc")
        loss = loss + bnm_weight * bnm
        opt.zero_grad(); loss.backward(); opt.step()
    feat.eval(); cls.eval(); return feat, cls
