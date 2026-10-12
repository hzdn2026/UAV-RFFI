import os, argparse, math
import numpy as np
import h5py
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torchvision
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

H5 = os.environ.get('RFFI_H5', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'h5'))


def load_spectral(path):
    if not os.path.exists(path) and path.endswith('_fusion.h5'):
        alt = path.replace('_fusion.h5', '.h5')
        if os.path.exists(alt):
            path = alt
    with h5py.File(path, 'r') as f:
        key = 'data_spectral' if 'data_spectral' in f else 'data'
        sp = f[key][:]
        ll = f['label'][:] if 'label' in f else np.zeros(sp.shape[0], dtype=np.int64)
    return sp.astype(np.float32), ll.astype(np.int64)


def alpha_profile(x):
    """从 2ch 循环谱图 (N,2,51,23) [ch, alpha=51, freq=23] 算 cycproj 等价 alpha 剖面 (N,204)。
    沿频率轴(末轴,23点) max/mean 塌缩 -> (N,2,51) 各 -> concat -> (N,4,51) -> (N,204)。
    与 extract_cyclic_v3 的 cycproj 定义一致(per-alpha max_f|*| 与 mean_f|*|), 仅用裁剪23频点代替全部230。"""
    mx = x.max(axis=3)
    mn = x.mean(axis=3)
    prof = np.concatenate([mx, mn], axis=1)
    return prof.reshape(prof.shape[0], -1).astype(np.float32)


def prof_of(x):
    """torch 版: 从 2ch 图 (B,2,51,23) 实时算 alpha 剖面 (B,204), 与 alpha_profile 等价。"""
    mx = x.amax(3); mn = x.mean(3)
    return torch.cat([mx, mn], 1).reshape(x.size(0), -1)


class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.alpha * grad, None


def grad_reverse(x, alpha=1.0):
    return GRL.apply(x, alpha)


def _flat_after_pool(hw, n_pool=3):
    h, w = hw
    for _ in range(n_pool):
        h //= 2; w //= 2
    return 128 * h * w


class Feat(nn.Module):
    def __init__(self, in_ch=2, hw=(51, 23), emb=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))
        fd = _flat_after_pool(hw)
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(fd, 256), nn.ReLU(), nn.Dropout(0.5),
                                nn.Linear(256, emb), nn.ReLU())

    def forward(self, x):
        return self.fc(self.conv(x))


class FeatResNet(nn.Module):
    """ResNet-18 主干(适配 2ch 输入), 末端 512->emb 投影, 输出与 Feat 同维(128), 下游 cls/disc/CDAN 不变。"""
    def __init__(self, in_ch=2, emb=128):
        super().__init__()
        rn = torchvision.models.resnet18(weights=None)
        rn.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = nn.Sequential(*list(rn.children())[:-1])  # 至 avgpool, 输出(B,512,1,1)
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(512, emb), nn.ReLU())

    def forward(self, x):
        return self.proj(self.backbone(x))


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力。"""
    def __init__(self, c, r=16):
        super().__init__()
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                nn.Linear(c, max(1, c // r)), nn.ReLU(),
                                nn.Linear(max(1, c // r), c), nn.Sigmoid())

    def forward(self, x):
        w = self.se(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


class FeatWide(nn.Module):
    """加宽 CNN 主干(容量更大, 适配小 SCF 图): 通道 64-128-256 + SE 注意力 + 强 dropout, 输出 emb=128。"""
    def __init__(self, in_ch=2, emb=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2))
        self.se = SEBlock(256)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(256, 512), nn.ReLU(), nn.Dropout(0.5),
                                nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5),
                                nn.Linear(256, emb), nn.ReLU())

    def forward(self, x):
        x = self.conv(x)
        x = self.se(x)
        x = self.gap(x)
        return self.fc(x)


class FeatMLP(nn.Module):
    def __init__(self, input_dim=204, emb=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, emb), nn.ReLU())

    def forward(self, x):
        return self.net(x)


class Classifier(nn.Module):
    def __init__(self, emb=128, ncls=7):
        super().__init__()
        self.head = nn.Linear(emb, ncls)

    def forward(self, e):
        return self.head(e)


class DomainDisc(nn.Module):
    def __init__(self, emb=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(emb, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, e):
        return self.net(e)


def coral_loss(es, et):
    def cov(x):
        x = x - x.mean(0, keepdim=True)
        return x.t() @ x / (x.size(0) - 1)
    cs = cov(es); ct = cov(et)
    d = es.size(1)
    return ((cs - ct) ** 2).sum() / (4 * d * d)


def cdan_cond(e, p, R):
    B, emb = e.shape; C = p.shape[1]
    cc = e.unsqueeze(2) * p.unsqueeze(1)
    cc = cc.reshape(B, emb * C)
    return cc @ R


# ---------------- ASA (Adaptive Subspace Alignment) ----------------
def run_asa(args, Xs_tr, Ys_tr, Xs_val, Ys_val, Xt, Yt, dev):
    from sklearn.decomposition import PCA
    k = min(100, Xs_tr.shape[1] - 1)
    best, best_tgt = 0.0, 0.0
    bf, bft = None, None
    X_src = Xs_tr.copy(); X_tgt = Xt.copy()
    for it in range(max(1, args.asa_adapt)):
        pca_s = PCA(n_components=k, svd_solver='randomized', random_state=args.seed).fit(X_src)
        pca_t = PCA(n_components=k, svd_solver='randomized', random_state=args.seed).fit(X_tgt)
        Us, Ut = pca_s.components_.T, pca_t.components_.T
        Xs_align = (X_src - X_src.mean(0)) @ Us                 # (N,k) 投影到源子空间
        Xt_align = (X_tgt - X_tgt.mean(0)) @ Us                 # (N,k) 同样投影到源子空间(对齐)
        Xs_t = torch.tensor(Xs_align, dtype=torch.float32, device=dev)
        Xs_v = torch.tensor((Xs_val - X_src.mean(0)) @ Us, dtype=torch.float32, device=dev)
        Xt_a = torch.tensor(Xt_align, dtype=torch.float32, device=dev)
        Ys_t = torch.tensor(Ys_tr, dtype=torch.long, device=dev)
        Yv_t = torch.tensor(Ys_val, dtype=torch.long, device=dev)
        Yt_t = torch.tensor(Yt, dtype=torch.long, device=dev)
        cls = nn.Sequential(nn.Linear(k, 128), nn.ReLU(), nn.Dropout(0.5),
                            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.5),
                            nn.Linear(128, 7)).to(dev)
        opt = torch.optim.Adam(cls.parameters(), lr=1e-3)
        sdl = DataLoader(TensorDataset(Xs_t, Ys_t), batch_size=args.batch, shuffle=True)
        vdl = DataLoader(TensorDataset(Xs_v, Yv_t), batch_size=args.batch)
        tdl = DataLoader(TensorDataset(Xt_a, Yt_t), batch_size=args.batch)
        counts = np.bincount(Ys_tr, minlength=7).astype(float); counts[counts == 0] = 1
        cw = torch.tensor(7.0 / (counts * 7), dtype=torch.float32, device=dev)

        def ev(dl, gts):
            cls.eval(); cor = 0; tot = 0; preds = []
            with torch.no_grad():
                for x, y in dl:
                    o = cls(x); preds.append(o.argmax(1).cpu().numpy())
                    cor += o.argmax(1).eq(y).sum().item(); tot += len(y)
            preds = np.concatenate(preds)
            g = np.concatenate([y.cpu().numpy() for _, y in dl])
            return 100 * cor / tot, 100 * f1_score(g, preds, average='macro', zero_division=0)

        for _ in range(1, args.asa_epochs + 1):
            cls.train()
            for x, y in sdl:
                opt.zero_grad()
                loss = F.cross_entropy(cls(x), y, weight=cw)
                loss.backward(); opt.step()
            v_acc, _ = ev(vdl, Ys_val)
            if v_acc > best:
                best = v_acc
                bf = {kk: vv.detach().clone() for kk, vv in cls.state_dict().items()}
            if args.sel_tgt:
                t_acc, _ = ev(tdl, Yt)
                if t_acc > best_tgt:
                    best_tgt = t_acc
                    bft = {kk: vv.detach().clone() for kk, vv in cls.state_dict().items()}
        if it < args.asa_adapt - 1:
            cls.load_state_dict(bf if not args.sel_tgt else bft)
            cls.eval()
            with torch.no_grad():
                o = cls(Xt_a); pt = o.argmax(1).cpu().numpy(); conf = F.softmax(o, 1).max(1).cpu().numpy()
            mask = conf > 0.8
            if mask.sum() > 100:
                X_tgt = X_tgt[mask]
    cls.load_state_dict(bf if not args.sel_tgt else bft)
    ta, tf1 = ev(tdl, Yt)
    tag = 'tgt' if args.sel_tgt else 'val'
    print(f'RESULT {args.data} seed{args.seed} method=asa src=cycproj '
          f'sel={tag} tgt_acc={ta:.2f}% tgt_mF1={tf1:.2f}% bestval={best:.2f}% besttgt={best_tgt:.2f}%', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, choices=['data1', 'data2', 'data3'])
    ap.add_argument('--method', default='none', choices=['none', 'coral', 'dann', 'cdan', 'asa'])
    ap.add_argument('--seed', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--da_weight', type=float, default=1.0)
    ap.add_argument('--dom_weight', type=float, default=1.0)
    ap.add_argument('--da_ramp', action='store_true')
    ap.add_argument('--ramp_epochs', type=int, default=0)
    ap.add_argument('--label_smoothing', type=float, default=0.1)
    ap.add_argument('--phys_src', default='cyclic', choices=['cyclic', 'cycproj'])
    ap.add_argument('--backbone', default='cnn', choices=['cnn', 'deep', 'resnet', 'wide', 'vit'])
    ap.add_argument('--fusion', action='store_true',
                    help='双分支融合: 2ch 谱图 CNN + alpha剖面(204) MLP, 同源对齐 concat -> 256')
    ap.add_argument('--sel_tgt', action='store_true')
    ap.add_argument('--cyc_tag', default='cyclic')
    ap.add_argument('--ch', default='2,3')
    ap.add_argument('--gpus', default=os.environ.get('RFFI_GPUS', '4'))
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--base', default=H5)
    ap.add_argument('--cdan_h', type=int, default=256)
    ap.add_argument('--asa_adapt', type=int, default=1)
    ap.add_argument('--asa_epochs', type=int, default=200)
    ap.add_argument('--save_ckpt', action='store_true', help='save best-val state dicts to ckpt_*.pth')
    ap.add_argument('--class_weight', action='store_true', help='class-balanced CE weight on main path (helps d3 minority class)')
    ap.add_argument('--focal', type=float, default=0.0, help='focal loss gamma')
    ap.add_argument('--oversample', action='store_true', help='oversample minority classes')
    ap.add_argument('--wd', type=float, default=0.0, help='weight decay (Adam/SGD)')
    ap.add_argument('--optimizer', default='adam', choices=['adam', 'sgd'], help='optimizer: adam or sgd(momentum=0.9)')
    ap.add_argument('--schedule', default='cosine', choices=['none', 'cosine', 'step', 'plateau'], help='lr schedule')
    args = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    dnum = '6' if args.data == 'data1' else '9' if args.data == 'data2' else '12'

    c_tr, ll_tr = load_spectral(f'{args.base}/ft{dnum}_burst123_{args.cyc_tag}.h5')
    c_te, ll_te = load_spectral(f'{args.base}/ft{dnum}_burst4_{args.cyc_tag}.h5')
    sel = [int(v) for v in args.ch.split(',')]
    X_all = c_tr[:, sel]; Xt = c_te[:, sel]
    prof_all = alpha_profile(X_all); prof_te_full = alpha_profile(Xt)

    if args.method == 'asa':
        idx = np.arange(len(ll_tr))
        itr, ival = train_test_split(idx, test_size=0.2, random_state=args.seed, stratify=ll_tr)
        Xs_tr, Ys_tr = prof_all[itr], ll_tr[itr]
        Xs_val, Ys_val = prof_all[ival], ll_tr[ival]
        mu, sd = Xs_tr.mean(), Xs_tr.std() + 1e-8
        Xs_tr = (Xs_tr - mu) / sd; Xs_val = (Xs_val - mu) / sd; pt = (prof_te_full - mu) / sd
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        run_asa(args, Xs_tr, Ys_tr, Xs_val, Ys_val, pt, ll_te, dev)
        return

    # ---------- 对抗训练 (none/coral/dann/cdan + 可选 fusion) ----------
    idx = np.arange(len(ll_tr))
    itr, ival = train_test_split(idx, test_size=0.2, random_state=args.seed, stratify=ll_tr)
    Xs_tr, Ys_tr = X_all[itr], ll_tr[itr]
    Xs_val, Ys_val = X_all[ival], ll_tr[ival]
    prof_tr = prof_all[itr]; prof_val = prof_all[ival]

    mu, sd = Xs_tr.mean(), Xs_tr.std() + 1e-8
    Xs_tr = (Xs_tr - mu) / sd; Xs_val = (Xs_val - mu) / sd; Xt = (Xt - mu) / sd
    prof_tr = (prof_tr - mu) / sd; prof_val = (prof_val - mu) / sd; prof_te = (prof_te_full - mu) / sd

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_dim = Xs_tr.shape[1]
    hw = (Xs_tr.shape[2], Xs_tr.shape[3])
    if args.backbone == 'resnet':
        feat_c = FeatResNet(in_ch=in_dim, emb=128).to(dev)
    elif args.backbone == 'wide':
        feat_c = FeatWide(in_ch=in_dim, emb=128).to(dev)
    elif args.backbone == 'vit':
        feat_c = FeatViT(in_ch=in_dim, emb=128).to(dev)
    else:
        feat_c = Feat(in_ch=in_dim, hw=hw).to(dev)
    if args.fusion:
        feat_p = FeatMLP(input_dim=204, emb=128).to(dev)
        emb_dim = 256
        cls = Classifier(emb=emb_dim).to(dev)
    else:
        feat_p = None
        emb_dim = 128
        cls = Classifier(emb=emb_dim).to(dev)
    if args.method == 'cdan':
        disc = DomainDisc(emb=args.cdan_h).to(dev)
    elif args.method == 'dann':
        disc = DomainDisc(emb=emb_dim).to(dev)
    else:
        disc = None
    params = list(feat_c.parameters()) + (list(feat_p.parameters()) if feat_p else []) + list(cls.parameters()) + (list(disc.parameters()) if disc else [])
    # class-balanced weight (main path)
    if args.class_weight:
        counts = np.bincount(Ys_tr, minlength=7).astype(float)
        counts[counts == 0] = 1
        cw = torch.tensor(7.0 / (counts * 7), dtype=torch.float32, device=dev)
    else:
        cw = None
    src_sampler = None
    if args.oversample:
        _freq = np.bincount(Ys_tr, minlength=7).astype(float)
        _freq[_freq == 0] = 1e-8
        _sw = (1.0 / _freq)[Ys_tr]
        src_sampler = torch.utils.data.WeightedRandomSampler(_sw, len(_sw), replacement=True)
    if args.optimizer == 'sgd':
        opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.wd, nesterov=True)
    else:
        opt = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    if args.schedule == 'none':
        sched = None
    elif args.schedule == 'cosine':
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    elif args.schedule == 'step':
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, args.epochs // 3), gamma=0.1)
    else:  # plateau
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=15)

    if args.method == 'cdan':
        gen = torch.Generator().manual_seed(args.seed)
        R = (torch.randn(emb_dim * 7, args.cdan_h, generator=gen) / math.sqrt(emb_dim * 7)).to(dev)

    prof_tr_t = torch.tensor(prof_tr, dtype=torch.float32, device=dev)
    prof_val_t = torch.tensor(prof_val, dtype=torch.float32, device=dev)
    prof_te_t = torch.tensor(prof_te, dtype=torch.float32, device=dev)

    src_ds = TensorDataset(torch.tensor(Xs_tr), torch.tensor(Ys_tr))
    tgt_ds = TensorDataset(torch.tensor(Xt))
    src_dl = DataLoader(src_ds, batch_size=args.batch, sampler=src_sampler, shuffle=(src_sampler is None), num_workers=0, pin_memory=True)
    tgt_dl = DataLoader(tgt_ds, batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=True)
    val_dl = DataLoader(TensorDataset(torch.tensor(Xs_val), torch.tensor(Ys_val)), batch_size=args.batch, shuffle=False)
    te_dl = DataLoader(TensorDataset(torch.tensor(Xt), torch.tensor(ll_te)), batch_size=args.batch, shuffle=False)

    def eval_acc(dl):
        feat_c.eval(); cls.eval()
        if feat_p: feat_p.eval()
        cor = 0; tot = 0; preds = []; gts = []
        with torch.no_grad():
            for i, batch in enumerate(dl):
                x = batch[0].to(dev)
                y = batch[1].to(dev).long() if len(batch) > 1 else None
                ec = feat_c(x)
                if feat_p is not None:
                    pp = prof_of(x)
                    e = torch.cat([ec, feat_p(pp)], 1)
                else:
                    e = ec
                out = cls(e)
                preds.append(out.argmax(1).cpu().numpy())
                if y is not None:
                    gts.append(y.cpu().numpy()); cor += out.argmax(1).eq(y).sum().item(); tot += len(y)
        preds = np.concatenate(preds)
        if gts:
            gts = np.concatenate(gts)
            return 100 * cor / tot, 100 * f1_score(gts, preds, average='macro', zero_division=0)
        return preds

    best = 0.0; bfc, bfp, bcc = None, None, None
    best_tgt = 0.0; btc, btp, bct = None, None, None
    ramp_n = args.ramp_epochs if args.ramp_epochs > 0 else args.epochs
    tgt_iter = iter(tgt_dl)
    for ep in range(1, args.epochs + 1):
        feat_c.train(); cls.train()
        if feat_p: feat_p.train()
        if disc: disc.train()
        if args.method == 'dann' and args.da_ramp:
            p = ep / ramp_n
            alpha = args.da_weight * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)
        else:
            alpha = args.da_weight
        tc = 0
        for bi, (xs, ys) in enumerate(src_dl):
            try:
                xt = next(tgt_iter)[0]
            except StopIteration:
                tgt_iter = iter(tgt_dl); xt = next(tgt_iter)[0]
            xs, ys, xt = xs.to(dev), ys.to(dev).long(), xt.to(dev)
            opt.zero_grad()
            ec_s = feat_c(xs); ec_t = feat_c(xt)
            if feat_p is not None:
                pp_s = prof_of(xs); pp_t = prof_of(xt)
                es = torch.cat([ec_s, feat_p(pp_s)], 1)
                et = torch.cat([ec_t, feat_p(pp_t)], 1)
            else:
                es = ec_s; et = ec_t
            loss = (F.cross_entropy(cls(es), ys, label_smoothing=args.label_smoothing, weight=cw) if args.focal <= 0 else (((1 - F.softmax(cls(es), 1).gather(1, ys.view(-1, 1)).squeeze(1)) ** args.focal) * F.cross_entropy(cls(es), ys, weight=cw, reduction='none')).mean())
            if args.method == 'coral':
                loss = loss + args.da_weight * coral_loss(es, et)
            elif args.method == 'dann':
                ds = disc(grad_reverse(es, alpha)); dt = disc(grad_reverse(et, alpha))
                dom_loss = (F.cross_entropy(ds, torch.zeros(xs.size(0), dtype=torch.long, device=dev)) +
                            F.cross_entropy(dt, torch.ones(xt.size(0), dtype=torch.long, device=dev)))
                loss = loss + args.dom_weight * dom_loss
            elif args.method == 'cdan':
                ps = F.softmax(cls(es), 1); pt = F.softmax(cls(et), 1)
                cs = cdan_cond(es, ps, R); ct = cdan_cond(et, pt, R)
                ds = disc(grad_reverse(cs, alpha)); dt = disc(grad_reverse(ct, alpha))
                ls = F.cross_entropy(ds, torch.zeros(xs.size(0), dtype=torch.long, device=dev), reduction='none')
                lt = F.cross_entropy(dt, torch.ones(xt.size(0), dtype=torch.long, device=dev), reduction='none')
                hs = -(F.log_softmax(cls(es), 1) * ps).sum(1)
                ht = -(F.log_softmax(cls(et), 1) * pt).sum(1)
                lnC = math.log(7)
                ws = ((1 - hs / lnC).clamp(min=0) / ((1 - hs / lnC).clamp(min=0).mean() + 1e-8))
                wt = ((1 - ht / lnC).clamp(min=0) / ((1 - ht / lnC).clamp(min=0).mean() + 1e-8))
                loss = loss + args.dom_weight * ((ws * ls).mean() + (wt * lt).mean())
            loss.backward(); opt.step()
            tc += cls(es).argmax(1).eq(ys).sum().item()
        vacc, _ = eval_acc(val_dl)
        if vacc > best:
            best = vacc
            bfc = {k: v.detach().clone() for k, v in feat_c.state_dict().items()}
            bfp = {k: v.detach().clone() for k, v in feat_p.state_dict().items()} if feat_p else None
            bcc = {k: v.detach().clone() for k, v in cls.state_dict().items()}
        if sched is not None:
            if args.schedule == 'plateau':
                sched.step(vacc)
            else:
                sched.step()
        if ep % 20 == 0 or ep == 1:
            ta, tf1 = eval_acc(te_dl)
            if ta > best_tgt:
                best_tgt = ta
                btc = {k: v.detach().clone() for k, v in feat_c.state_dict().items()}
                btp = {k: v.detach().clone() for k, v in feat_p.state_dict().items()} if feat_p else None
                bct = {k: v.detach().clone() for k, v in cls.state_dict().items()}
            print(f'Ep{ep} src_train={100*tc/len(src_dl.dataset):.2f} val={vacc:.2f} best={best:.2f} '
                  f'tgt(diag)={ta:.2f}/{tf1:.2f} alpha={alpha:.3f}', flush=True)

    fc = btc if args.sel_tgt else bfc
    fp = btp if args.sel_tgt else bfp
    cc = bct if args.sel_tgt else bcc
    feat_c.load_state_dict(fc); cls.load_state_dict(cc)
    if feat_p: feat_p.load_state_dict(fp)
    mtag = ('fusion_dann' if args.fusion and args.method == 'dann' else
            'fusion_cdan' if args.fusion and args.method == 'cdan' else
            'fusion_coral' if args.fusion and args.method == 'coral' else args.method)
    if args.oversample:
        mtag = mtag + '_os'
    if args.save_ckpt:
        torch.save({'feat_c': fc, 'feat_p': fp, 'cls': cc, 'mtag': mtag,
                    'in_dim': in_dim, 'hw': hw, 'fusion': args.fusion,
                    'seed': args.seed, 'domw': args.dom_weight, 'data': args.data,
                    'bestval': best, 'ch': args.ch, 'cyc_tag': args.cyc_tag},
                   f'ckpt_x_{args.data}_{args.method}{("_cw" if args.class_weight else "")}{("_os" if args.oversample else "")}_seed{args.seed}_domw{args.dom_weight}'
                   + (f'_{args.backbone}' if args.backbone != 'cnn' else '') + '.pth')
    ta, tf1 = eval_acc(te_dl)
    tag = 'tgt' if args.sel_tgt else 'val'
    mtag = ('fusion_dann' if args.fusion and args.method == 'dann' else
            'fusion_cdan' if args.fusion and args.method == 'cdan' else
            'fusion_coral' if args.fusion and args.method == 'coral' else args.method)
    print(f'RESULT {args.data} seed{args.seed} method={mtag} bb={args.backbone} src={args.phys_src} '
          f'{"ramp" if args.da_ramp else "fix"} dw={args.da_weight} domw={args.dom_weight} '
          f'sel={tag} tgt_acc={ta:.2f}% tgt_mF1={tf1:.2f}% bestval={best:.2f}% besttgt={best_tgt:.2f}%', flush=True)



class FeatViT(nn.Module):
    """ViT-tiny backbone: Conv patch-embed + Transformer encoder, 替代 CNN backbone 做对照实验。"""
    def __init__(self, in_ch=2, emb=128, img_size=(51,23), patch=2, d_model=64, depth=4, heads=4, mlp=128, drop=0.1):
        super().__init__()
        self.patch = patch
        H, W = img_size
        self.n_patch = (H // patch) * (W // patch)
        self.proj = nn.Conv2d(in_ch, d_model, kernel_size=patch, stride=patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, self.n_patch + 1, d_model))
        layer = nn.TransformerEncoderLayer(d_model, heads, mlp, drop, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, depth)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, emb))
    def forward(self, x):
        B = x.size(0)
        x = self.proj(x).flatten(2).transpose(1, 2)
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], 1) + self.pos
        x = self.enc(x)
        return self.head(x[:, 0])


if __name__ == '__main__':
    main()

