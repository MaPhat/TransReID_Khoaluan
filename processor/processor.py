import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist

def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, 
             local_rank,
             gcn_model=None):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    if gcn_model is not None:
        evaluator = R1_mAP_eval(num_query, max_rank=50, 
                                feat_norm=cfg.TEST.FEAT_NORM, 
                                reranking=cfg.TEST.RE_RANKING, 
                                is_graph=cfg.TEST.GRAPH_RE_RANKING,
                                is_learngraph=cfg.MODEL.LEARN_BASED_GCN,
                                gcn_model=gcn_model)
    else:
        evaluator = R1_mAP_eval(num_query, max_rank=50, 
                                feat_norm=cfg.TEST.FEAT_NORM, 
                                reranking=cfg.TEST.RE_RANKING, 
                                is_graph=cfg.TEST.GRAPH_RE_RANKING)
    scaler = amp.GradScaler()

    # Early stopping state
    early_stopping_patience = cfg.SOLVER.EARLY_STOPPING_PATIENCE
    best_mAP = 0.0
    no_improve_count = 0

    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            with amp.autocast(enabled=True):
                score, feat = model(img, target, cam_label=target_cam, view_label=target_view )
                if cfg.MODEL.LEARN_BASED_GCN:
                    feat_is_list = isinstance(feat, list)
                    global_feature = feat[0] if feat_is_list else feat
                    A_g, A_c = build_graphs_for_batch(global_feature, target_cam)
                    global_feature_gcn = gcn_model(global_feature, A_g, A_c)

                    if feat_is_list:
                        feat[0] = global_feature_gcn
                    else:
                        feat = global_feature_gcn
                loss = loss_fn(score, feat, target, target_cam)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            
            if cfg.MODEL.LEARN_BASED_GCN:
                torch.save(gcn_model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, 'GCN_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()

                    if cfg.MODEL.LEARN_BASED_GCN:
                        gcn_model.eval()

                    if cfg.MODEL.LEARN_BASED_GCN and gcn_model is not None:
                        evaluator.gcn_model = gcn_model

                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()

                if cfg.MODEL.LEARN_BASED_GCN:
                        gcn_model.eval()

                if cfg.MODEL.LEARN_BASED_GCN and gcn_model is not None:
                    evaluator.gcn_model = gcn_model

                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        feat = model(img, cam_label=camids, view_label=target_view)                            
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()

                # Early stopping
                if early_stopping_patience > 0:
                    if mAP > best_mAP:
                        best_mAP = mAP
                        no_improve_count = 0
                        torch.save(model.state_dict(),
                                   os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_best.pth'))
                        if cfg.MODEL.LEARN_BASED_GCN and gcn_model is not None:
                            torch.save(gcn_model.state_dict(),
                                       os.path.join(cfg.OUTPUT_DIR, 'GCN_best.pth'))
                        logger.info("Best mAP updated: {:.1%} — model saved.".format(best_mAP))
                    else:
                        no_improve_count += 1
                        logger.info("No improvement for {}/{} eval periods.".format(
                            no_improve_count, early_stopping_patience))
                        if no_improve_count >= early_stopping_patience:
                            logger.info(
                                "Early stopping triggered at epoch {}. "
                                "Best mAP: {:.1%}".format(epoch, best_mAP))
                            return


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query,
                 gcn_model=None):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    if gcn_model is not None and cfg.MODEL.LEARN_BASED_GCN:
        evaluator = R1_mAP_eval(num_query, max_rank=50, 
                            feat_norm=cfg.TEST.FEAT_NORM, 
                            reranking=cfg.TEST.RE_RANKING, 
                            is_graph=cfg.TEST.GRAPH_RE_RANKING,
                            is_learngraph=cfg.MODEL.LEARN_BASED_GCN,
                            gcn_model=gcn_model)
    else:
        evaluator = R1_mAP_eval(num_query, max_rank=50, 
                            feat_norm=cfg.TEST.FEAT_NORM, 
                            reranking=cfg.TEST.RE_RANKING, 
                            is_graph=cfg.TEST.GRAPH_RE_RANKING)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return mAP, cmc[0], cmc[4], cmc[9]

def build_graphs_for_batch(feat, camids, k=5, gamma=2.0):
    """
    Xây dựng đồ thị động cho một Batch huấn luyện.
    feat: (N, D) - Đặc trưng trích xuất từ Backbone
    camids: (N,) - ID camera của từng ảnh trong batch
    k: Số lượng lân cận gần nhất
    """
    # feat = torch.Tensor(feat)
    if feat.dim() == 1:
        feat = feat.unsqueeze(0)
    N = feat.size(0)
    device = feat.device
    k = min(k, N)

    dist_mat = torch.cdist(feat, feat, p=2)

    _, nn_idx = torch.topk(dist_mat, k=k, largest=False)
    
    A_g = torch.zeros(N, N, device=device)
    weights = torch.exp(-dist_mat / gamma)
    A_g.scatter_(1, nn_idx, weights.gather(1, nn_idx))

    camids = camids.view(-1, 1)
    mask_cross = (camids != camids.T).float()
    
    A_c = A_g * mask_cross

    def normalize(A):
        row_sum = A.sum(1, keepdim=True)
        A_norm = A / (row_sum + 1e-6)
        return A_norm

    return normalize(A_g), normalize(A_c)
