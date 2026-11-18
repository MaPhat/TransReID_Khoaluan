"""
LiteGCR stage 2: train the GCN on FEATURES of a frozen, already-trained backbone.

The backbone is not needed here - only cached features, identity labels and camera labels:

    refined = GCN(features, graph built on the batch)          # graph from features (+ camera IDs)
    loss    = w_tri * triplet(refined, ids) + w_id * softmax_ID(head(refined), ids)

`head` is a NEW classifier (BNNeck + Linear) on top of the refined features: the backbone's own
classifier sits before the GCN, so without a new head the ID loss would never reach the GCN.
The head is only used for training and is discarded afterwards; the saved file contains the GCN only.

Identical copy in TransReID / RPTM / MBR4G (each repo only provides feature extraction + the
triplet / ID loss objects that its backbone already uses).
"""
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # MBR4G layout
    from litegcr import GCNRefiner, build_graphs, graph_reranking
except ImportError:  # TransReID / RPTM layout
    from utils.graph_reranking import GCNRefiner, build_graphs, graph_reranking


def log_progress(loader, desc, log=print, every=30.0):
    """Iterate a DataLoader and log throughput + ETA every `every` seconds (and on the first batch)."""
    total, bs = len(loader), getattr(loader, 'batch_size', None) or 1
    t0 = last = time.time()
    for i, batch in enumerate(loader, 1):
        yield batch
        now = time.time()
        if i == 1 or i == total or now - last >= every:
            rate = i / (now - t0)
            log(f'[{desc}] {i}/{total} batches ({100 * i / total:.0f}%) | {rate * bs:.0f} img/s | '
                f'ETA {(total - i) / rate / 60:.1f} min')
            last = now


class IDHead(nn.Module):
    """BNNeck + linear classifier (same design as the ReID baselines)."""

    def __init__(self, dim, num_classes):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim)
        self.bn.bias.requires_grad_(False)
        self.fc = nn.Linear(dim, num_classes, bias=False)
        nn.init.normal_(self.fc.weight, std=0.001)

    def forward(self, x):
        return self.fc(self.bn(x))


def pk_indices(pid_to_idx, pids, P, K, rng):
    """P identities x K instances (with replacement if an identity has fewer than K images)."""
    chosen = rng.choice(len(pids), size=min(P, len(pids)), replace=False)
    out = []
    for c in chosen:
        pool = pid_to_idx[pids[c]]
        out.extend(rng.choice(pool, size=K, replace=len(pool) < K))
    return np.asarray(out)


def prepare_descriptor(x, dim, normalize_rest):
    """Descriptor as the plain (no graph) pipeline uses it.
    normalize_rest=True  (TransReID / RPTM): the WHOLE descriptor is L2-normalised (also with JPM local parts).
    normalize_rest=False (MBR): global part normalised, the other branches are already normalised."""
    x = x.float()
    if normalize_rest:
        return F.normalize(x, p=2, dim=1)
    g = F.normalize(x[:, :dim], p=2, dim=1)
    r = x[:, dim:]
    return torch.cat([g, r], 1) if r.numel() else g


def descriptor_distmat(qf, gf, dim, normalize_rest):
    """Plain euclidean distance on the descriptor - no graph, no GCN."""
    return torch.cdist(prepare_descriptor(qf, dim, normalize_rest),
                       prepare_descriptor(gf, dim, normalize_rest), p=2).cpu().numpy()


def evaluate(gcn, d, eval_fn, dim, use_camera, normalize_rest, k, gamma, device):
    gcn.eval()
    with torch.no_grad():
        dist = graph_reranking(d['q_desc'], d['g_desc'], d['q_cams'], d['g_cams'], k=k, gamma=gamma,
                               gcn_model=gcn, use_camera=use_camera, normalize_rest=normalize_rest)
    cmc, mAP = eval_fn(dist, d['q_pids'], d['g_pids'], d['q_cams'], d['g_cams'])
    return cmc, mAP


def train_gcn_on_features(d, make_triplet, make_id_loss, eval_fn, out_dir, feat_dim, num_classes,
                          use_camera=True, normalize_rest=True, epochs=30, patience=5,
                          P=64, K=4, lr=1e-3, weight_decay=5e-4, w_id=1.0, w_tri=1.0,
                          k=20, gamma=0.5, device='cuda', seed=0, log=print):
    """d: dict with train_feats (N,>=feat_dim) / train_pids / train_cams and the validation
    descriptors q_desc, g_desc (+ q_pids, g_pids, q_cams, g_cams).
    make_triplet(): callable (feat, ids) -> scalar loss.   make_id_loss(): callable (logits, ids) -> scalar loss.
    eval_fn(distmat, q_pids, g_pids, q_cams, g_cams) -> (cmc, mAP)."""
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    device = torch.device(device)

    feats = d['train_feats'][:, :feat_dim].contiguous().to(device)
    pids = np.asarray(d['train_pids']).astype(np.int64)
    cams = torch.as_tensor(np.asarray(d['train_cams']), dtype=torch.long, device=device)
    pids_t = torch.as_tensor(pids, device=device)
    pid_to_idx = {}
    for i, p in enumerate(pids):
        pid_to_idx.setdefault(int(p), []).append(i)
    id_list = list(pid_to_idx)
    pid_to_idx = {p: np.asarray(v) for p, v in pid_to_idx.items()}
    iters = max(1, len(pids) // (P * K))

    gcn = GCNRefiner(feat_dim).to(device)
    head = IDHead(feat_dim, num_classes).to(device)
    triplet, id_loss = make_triplet(), make_id_loss()
    params = [p for p in list(gcn.parameters()) + list(head.parameters()) if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)

    base_cmc, base_map = eval_fn(descriptor_distmat(d['q_desc'], d['g_desc'], feat_dim, normalize_rest),
                                 d['q_pids'], d['g_pids'], d['q_cams'], d['g_cams'])
    log(f'[baseline, no GCN / no re-ranking]  mAP {base_map:.2%}  R1 {base_cmc[0]:.2%}')

    best, bad, best_path = -1.0, 0, os.path.join(out_dir, 'gcn_best.pth')
    for ep in range(1, epochs + 1):
        gcn.train(); head.train()
        t0, s_tri, s_id, s_acc = time.time(), 0.0, 0.0, 0.0
        for _ in range(iters):
            idx = torch.as_tensor(pk_indices(pid_to_idx, id_list, P, K, rng), device=device)
            f, y, c = feats[idx].float(), pids_t[idx], cams[idx]
            A_g, A_c = build_graphs(f, c if use_camera else None, k, gamma)   # graph from the batch
            refined = gcn(f, A_g, A_c)
            l_tri = triplet(refined, y)
            logits = head(refined)
            l_id = id_loss(logits, y)
            loss = w_tri * l_tri + w_id * l_id
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            s_tri += l_tri.item(); s_id += l_id.item(); s_acc += (logits.argmax(1) == y).float().mean().item()
        sched.step()
        cmc, mAP = evaluate(gcn, d, eval_fn, feat_dim, use_camera, normalize_rest, k, gamma, device)
        log(f'Epoch {ep}/{epochs} | tri {s_tri/iters:.4f} id {s_id/iters:.4f} acc {s_acc/iters:.3f} | '
            f'val mAP {mAP:.2%} R1 {cmc[0]:.2%} (baseline mAP {base_map:.2%}) | {time.time()-t0:.0f}s')
        if mAP > best:
            best, bad = mAP, 0
            torch.save(gcn.state_dict(), best_path)
            log(f'  best mAP {best:.2%} -> saved {best_path}')
        else:
            bad += 1
            if bad >= patience:
                log(f'Early stopping at epoch {ep} (best mAP {best:.2%})')
                break
    torch.save(gcn.state_dict(), os.path.join(out_dir, 'gcn_last.pth'))
    log(f'DONE best val mAP {best:.2%} (baseline {base_map:.2%})')
    return best_path


def evaluate_rows(d, eval_fn, feat_dim, use_camera, normalize_rest, rerank_fn=None, gcn_weights=None,
                  k=20, gamma=0.5, out_path=None, log=print):
    """Quick test from cached features (no backbone): Baseline / k-reciprocal / Rule-based graph / LiteGCR.
    Same functions as the full pipelines use, on the cached query/gallery split."""
    args = (d['q_pids'], d['g_pids'], d['q_cams'], d['g_cams'])
    rows = {}
    rows['Baseline (None)'] = eval_fn(descriptor_distmat(d['q_desc'], d['g_desc'], feat_dim, normalize_rest), *args)
    if rerank_fn is not None:
        q = prepare_descriptor(d['q_desc'], feat_dim, normalize_rest)
        g = prepare_descriptor(d['g_desc'], feat_dim, normalize_rest)
        rows['k-Reciprocal Re-ranking'] = eval_fn(np.asarray(rerank_fn(q, g)), *args)
    dist = graph_reranking(d['q_desc'], d['g_desc'], d['q_cams'], d['g_cams'], k=k, gamma=gamma, gcn_model=None,
                           use_camera=use_camera, normalize_rest=normalize_rest)
    rows['Rule-based Graph Re-ranking'] = eval_fn(dist, *args)
    if gcn_weights:
        gcn = GCNRefiner(feat_dim)
        gcn.load_param(gcn_weights)
        rows['LiteGCR (Ours)'] = evaluate(gcn, d, eval_fn, feat_dim, use_camera, normalize_rest, k, gamma, None)
    lines = ['| Method | mAP | Rank-1 | Rank-5 | Rank-10 |', '|---|---|---|---|---|']
    for name, (cmc, mAP) in rows.items():
        lines.append(f'| {name} | {100*mAP:.2f} | {100*cmc[0]:.2f} | {100*cmc[4]:.2f} | {100*cmc[9]:.2f} |')
    text = '\n'.join(lines)
    log('Quick test on cached query/gallery features (single split):\n' + text)
    if out_path:
        with open(out_path, 'w') as f:
            f.write(text + '\n')
    return rows
