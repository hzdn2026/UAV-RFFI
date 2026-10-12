import sys, os, glob, argparse, torch, numpy as np
import tta as tb

ap = argparse.ArgumentParser()
ap.add_argument('--data', default='data1', choices=['data1','data2','data3'], help='distance tag')
ap.add_argument('--topk', type=int, default=18, help='bestval top-k for headline')
args = ap.parse_args()
CKPT = os.environ.get('RFFI_CKPT', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints'))

from sklearn.metrics import f1_score

CACHE = os.path.join(CKPT, f'tent_softs_cache_{args.data}.pt')
files = sorted(set(glob.glob(f'{CKPT}/ckpt_x_{args.data}_*.pth'))
                   | set(glob.glob(f'{CKPT}/ckpt_{args.data}_*.pth')))
files = [f for f in files if '_resnet' not in os.path.basename(f)
         and '_wide' not in os.path.basename(f)]
keep = []
for f in files:
    bn = os.path.basename(f)
    if bn.startswith('ckpt_data1_'):
        keep.append(f)
    else:
        m = torch.load(f, map_location='cpu', weights_only=False).get('mtag')
        if m in ('cdan','none'):
            keep.append(f)
files = keep

def report(preds, name):
    acc = 100 * (preds == ll_te).float().mean().item()
    f1 = 100 * f1_score(ll_te.numpy(), preds.numpy(), average='macro', zero_division=0)
    print(f'  [{name}] acc={acc:.2f}% macroF1={f1:.2f}%')
    return acc, f1

if os.path.exists(CACHE):
    print('LOAD CACHE'); d = torch.load(CACHE)
    tent_softs, confs, bestvals, tags = d['softs'], d['confs'], d['bestvals'], d['tags']
    ll_te = d['ll_te']; files = d['files']
else:
    tent_softs, confs, bestvals, tags = [], [], [], []
    ll_te = None
    for f in files:
        ck = torch.load(f, map_location='cpu', weights_only=False)
        Xt_n, ll_te = tb.build_prep(int(ck['seed']), args.data)
        Xt_t = torch.tensor(Xt_n).to(tb.DEV)
        feat, cls = tb.load_model(ck)
        feat_t, cls_t = tb.tta_tent(feat, cls, Xt_t, 40, 1e-3)
        soft = tb.soft_of(feat_t, cls_t, Xt_t).cpu()
        tent_softs.append(soft)
        confs.append(soft.max(1).values.mean().item())
        bestvals.append(ck['bestval'])
        tags.append('OLD' if ck['seed'] in (123, 2024, 32, 777) else 'NEW')
    tent_softs = torch.stack(tent_softs)
    confs = torch.tensor(confs); bestvals = torch.tensor(bestvals)
    torch.save({'softs': tent_softs, 'confs': confs, 'bestvals': bestvals,
                'tags': tags, 'll_te': torch.tensor(ll_te), 'files': files}, CACHE)
    print('SAVED CACHE')
ll_te = torch.tensor(ll_te); M = len(files)

print(f'# {M} models; OLD={tags.count("OLD")} NEW={tags.count("NEW")}')
report(tent_softs.mean(0).argmax(1), 'ALL_eq(42)')
# bestval top-k full scan
print('--- topk_bestval scan ---')
for k in [10,12,14,16,18,20,22,24,26,28,30,35,42]:
    order = bestvals.argsort(descending=True)[:k]
    report(tent_softs[order].mean(0).argmax(1), f'topk_bestval={k}')
head_order = bestvals.argsort(descending=True)[:args.topk]
report(tent_softs[head_order].mean(0).argmax(1), f'HEADLINE topk_bestval={args.topk}')
# bootstrap: majority vote pseudo-label + agreement*conf
maj = tent_softs.mean(0).argmax(1)
agree = (tent_softs.argmax(2) == maj.unsqueeze(0)).float().mean(1)
score = agree * confs
print('--- topk_agreeconf scan ---')
for k in [13,15,17,20,25,30]:
    order = score.argsort(descending=True)[:k]
    report(tent_softs[order].mean(0).argmax(1), f'topk_agreeconf={k}')
print('DONE')
