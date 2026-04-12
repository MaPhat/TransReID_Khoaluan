import os
from config import cfg
import argparse
from datasets import make_dataloader
from model import make_model
from processor import do_inference
from utils.logger import setup_logger
from utils.graph_reranking import GCNRefiner


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()



    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)
    model.load_param(cfg.TEST.WEIGHT)

    if cfg.MODEL.LEARN_BASED_GCN:
        feature_dim = model.in_planes
        gcn_model = GCNRefiner(feature_dim=feature_dim)
        gcn_model.load_param(cfg.TEST.GCN_WEIGHT)


    if cfg.DATASETS.NAMES == 'VehicleID':
        for trial in range(10):
            train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
            if cfg.MODEL.LEARN_BASED_GCN:
                mAP, rank_1, rank5, rank10 = do_inference(cfg,
                                            model,
                                            val_loader,
                                            num_query,
                                            gcn_model)
            else:
                mAP, rank_1, rank5, rank10 = do_inference(cfg,
                                            model,
                                            val_loader,
                                            num_query)
            if trial == 0:
                all_mAP = mAP
                all_rank_1 = rank_1
                all_rank_5 = rank5
                all_rank_10 = rank10
            else:
                all_mAP = all_mAP + mAP
                all_rank_1 = all_rank_1 + rank_1
                all_rank_5 = all_rank_5 + rank5
                all_rank_10 = all_rank_10 + rank10

            logger.info("mAP:{:.1%}, rank_1:{}, rank_5 {}, rank_10 {} : trial : {}".format(mAP, rank_1, rank5, rank10, trial))
        logger.info("sum_mAP:{:.1%}, sum_rank_1:{:.1%}, sum_rank_5 {:.1%}, sum_rank_10 {:.1%}".format(all_mAP.sum()/10.0, all_rank_1.sum()/10.0, all_rank_5.sum()/10.0, all_rank_10.sum()/10.0))
    else:
        if cfg.MODEL.LEARN_BASED_GCN:
            do_inference(cfg,
                        model,
                        val_loader,
                        num_query,
                        gcn_model)
        else:
            do_inference(cfg,
                        model,
                        val_loader,
                        num_query)

