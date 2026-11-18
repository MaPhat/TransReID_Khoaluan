"""LiteGCR stage 2 for TransReID: train the GCN on features of a FROZEN, already-trained backbone.

  python train_gcn.py --config_file configs/VehicleID/vit_base.yml --out_dir ./gcn_out \
      TEST.WEIGHT <transformer_best.pth> MODEL.PRETRAIN_PATH <vit weights> DATASETS.ROOT_DIR <parent of VehicleID_V1.0>

Loss = backbone's triplet loss + softmax ID loss on a new head, both on the GCN-refined features.
Features are extracted once and cached in <out_dir>/features.pt.
"""
import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from config import cfg
from datasets import make_dataloader
from loss.softmax_loss import CrossEntropyLabelSmooth
from loss.triplet_loss import TripletLoss
from model import make_model
from utils.gcn_stage2 import evaluate_rows, log_progress, train_gcn_on_features
from utils.logger import setup_logger
from utils.metrics import eval_func
from utils.reranking import re_ranking


@torch.no_grad()
def extract(model, loader, device, desc, amp=False, log=print):
    feats, pids, cams = [], [], []
    for img, pid, camid, camids, view, _ in log_progress(loader, desc, log):
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp and device == 'cuda'):
            feat = model(img.to(device), cam_label=camids.to(device), view_label=view.to(device))
        feats.append(feat.float().cpu())
        pids.extend(pid)
        cams.extend(camid)
    return torch.cat(feats), np.asarray(pids), np.asarray(cams)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LiteGCR stage 2 (frozen backbone)')
    parser.add_argument('--config_file', default='', type=str)
    parser.add_argument('--out_dir', required=True, type=str)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--P', default=64, type=int, help='identities per batch')
    parser.add_argument('--K', default=4, type=int, help='images per identity')
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--no_amp', action='store_true', help='extract train features in FP32 (slower)')
    parser.add_argument('--eval_only', action='store_true', help='quick test from cached features, no training')
    parser.add_argument('--gcn_weights', default=None, help='GCN weights for --eval_only (default: <out_dir>/gcn_best.pth)')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    os.makedirs(args.out_dir, exist_ok=True)
    logger = setup_logger('transreid', args.out_dir, if_train=False)
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = True
    random.seed(cfg.SOLVER.SEED); np.random.seed(cfg.SOLVER.SEED); torch.manual_seed(cfg.SOLVER.SEED)

    cache = os.path.join(args.out_dir, 'features.pt')
    if os.path.exists(cache):
        logger.info('Using cached features: {}'.format(cache))
        d = torch.load(cache, weights_only=False)
    else:
        train_loader, train_loader_normal, val_loader, num_query, num_classes, cam_num, view_num = make_dataloader(cfg)
        model = make_model(cfg, num_class=num_classes, camera_num=cam_num, view_num=view_num)
        model.load_param(cfg.TEST.WEIGHT)
        model.to(device).eval()
        feat_dim = model.in_planes
        logger.info('Extracting train features ...')
        tf, tp, tc = extract(model, train_loader_normal, device, 'train feats', amp=not args.no_amp, log=logger.info)
        logger.info('Extracting query/gallery features ...')
        vf, vp, vc = extract(model, val_loader, device, 'query+gallery feats', log=logger.info)
        d = dict(train_feats=tf[:, :feat_dim].half(), train_pids=tp, train_cams=tc,
                 q_desc=vf[:num_query], g_desc=vf[num_query:], q_pids=vp[:num_query], g_pids=vp[num_query:],
                 q_cams=vc[:num_query], g_cams=vc[num_query:], feat_dim=feat_dim, num_classes=num_classes)
        torch.save(d, cache)
        del model
        torch.cuda.empty_cache()

    use_camera = cfg.DATASETS.NAMES != 'VehicleID'
    eval_fn = lambda dist, qp, gp, qc, gc: eval_func(dist, qp, gp, qc, gc)
    if args.eval_only:
        evaluate_rows(d, eval_fn, d['feat_dim'], use_camera, True,
                      rerank_fn=lambda q, g: re_ranking(q, g, k1=20, k2=6, lambda_value=0.3),   # TransReID's setting
                      gcn_weights=args.gcn_weights or os.path.join(args.out_dir, 'gcn_best.pth'),
                      out_path=os.path.join(args.out_dir, 'quick_results.md'), log=logger.info)
        raise SystemExit(0)

    triplet = TripletLoss() if cfg.MODEL.NO_MARGIN else TripletLoss(cfg.SOLVER.MARGIN)
    make_triplet = lambda: (lambda feat, y: triplet(feat, y)[0])
    make_id_loss = (lambda: CrossEntropyLabelSmooth(num_classes=d['num_classes'])) \
        if cfg.MODEL.IF_LABELSMOOTH == 'on' else (lambda: F.cross_entropy)

    train_gcn_on_features(
        d, make_triplet, make_id_loss,
        eval_fn=eval_fn,
        out_dir=args.out_dir, feat_dim=d['feat_dim'], num_classes=d['num_classes'],
        use_camera=use_camera, normalize_rest=True,
        epochs=args.epochs, patience=args.patience, P=args.P, K=args.K, lr=args.lr,
        w_id=cfg.MODEL.ID_LOSS_WEIGHT, w_tri=cfg.MODEL.TRIPLET_LOSS_WEIGHT,
        device=device, seed=cfg.SOLVER.SEED, log=logger.info)
