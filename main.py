###################################################################################
### PARTS OF THIS CODE ARE ORIGINALLY PROVIDED BY THE CoLA REPOSITORY ON GITHUB ###
### https://github.com/zhang-can/CoLA                                           ###
###################################################################################

import os
import sys
import time
import copy
import json

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import core.utils.utils as utils
import core.utils.init_model as init_utils
from core.config import cfg
from core.dataloaders.dataset import NpyFeature
from core.utils.losses import SPNTotalLoss, TotalLoss
from core.utils.eval import AverageCounter, ANETdetection


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def train_one_step_spn(net, loader_iter, optimizer, criterion, writer, step):
    net.train()
    
    data, label, _, _, _ = next(loader_iter)
    data = data.cuda()
    label = label.cuda()

    optimizer.zero_grad()
    _, labels, _, _, _, contrast_pairs = net(data)
    cost, loss = criterion(labels, label, contrast_pairs)

    cost.backward()
    optimizer.step()

    for key in loss.keys():
        writer.add_scalar(key, loss[key].cpu().item(), step)
    return cost

def train_spn(net, cfg, train_loader, writer):
    cfg.LR_SPN = eval(cfg.LR_SPN)

    criterion = SPNTotalLoss()

    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=cfg.LR_SPN[0],
        betas=(0.9, 0.999),
        weight_decay=0.0005
    )

    # TRAIN LOOP
    for step in range(1, cfg.NUM_ITERS + 1):
        if step > 1 and cfg.LR_SPN[step - 1] != cfg.LR_SPN[step - 2]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = cfg.LR_SPN[step - 1]

        if (step - 1) % len(train_loader) == 0:
            loader_iter = iter(train_loader)

        # balance losses
        if (step == 1000):
            criterion.balance(0.01)

        batch_time = AverageCounter()
        losses = AverageCounter()
        
        end = time.time()
        cost = train_one_step_spn(net, loader_iter, optimizer, criterion, writer, step)
        losses.update(cost.item(), cfg.BATCH_SIZE)
        batch_time.update(time.time() - end)
        end = time.time()

        if step == 1 or step % cfg.PRINT_FREQ == 0:
            print(('Step: [{0:04d}/{1}]\t' \
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    step, cfg.NUM_ITERS, batch_time=batch_time, loss=losses)))

    return net

def train_one_step_soi(net, loader_iter, optimizer, criterion, writer, step):
    net.train()
    
    data, label, _, _, _ = next(loader_iter)
    data = data.cuda()
    label = label.cuda()

    optimizer.zero_grad()
    video_scores, contrast_pairs, _, _ = net(data)
    cost, loss = criterion(video_scores, label, contrast_pairs)

    cost.backward()
    optimizer.step()

    for key in loss.keys():
        writer.add_scalar(key, loss[key].cpu().item(), step)
    return cost

def train_soi(net, cfg, train_loader, test_loader, test_info, writer):
    cfg.LR_SOI = eval(cfg.LR_SOI)

    best_mAP = -1

    criterion = TotalLoss()

    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=cfg.LR_SOI[0],
        betas=(0.9, 0.999),
        weight_decay=0.0005
    )

    # TRAIN LOOP
    for step in range(1, cfg.NUM_ITERS + 1):
        if step > 1 and cfg.LR_SOI[step - 1] != cfg.LR_SOI[step - 2]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = cfg.LR_SOI[step - 1]

        if (step - 1) % len(train_loader) == 0:
            loader_iter = iter(train_loader)

        # balance losses
        if (step == 500):
            criterion.balance(0.01)

        batch_time = AverageCounter()
        losses = AverageCounter()
        
        end = time.time()
        cost = train_one_step_soi(net, loader_iter, optimizer, criterion, writer, step)
        losses.update(cost.item(), cfg.BATCH_SIZE)
        batch_time.update(time.time() - end)
        end = time.time()

        if step == 1 or step % cfg.PRINT_FREQ == 0:
            print(('Step: [{0:04d}/{1}]\t' \
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    step, cfg.NUM_ITERS, batch_time=batch_time, loss=losses)))
            
        if step > -1 and step % cfg.TEST_FREQ == 0:

            mAP_50, mAP_AVG = test(
                net,
                cfg,
                test_loader,
                test_info,
                step,
                writer
            )

            if test_info["average_mAP"][-1] > best_mAP:
                best_mAP = test_info["average_mAP"][-1]

                utils.save_best_record_thumos(
                    test_info, 
                    os.path.join(cfg.OUTPUT_PATH, "best_results.txt")
                )

                torch.save(
                    net.state_dict(),
                    os.path.join(cfg.MODEL_PATH, "model_best.pth.tar")
                )

            print(('- Test result: \t' \
                   'mAP@0.5 {mAP_50:.2%}\t' \
                   'mAP@AVG {mAP_AVG:.2%} (best: {best_mAP:.2%})'.format(
                   mAP_50=mAP_50, mAP_AVG=mAP_AVG, best_mAP=best_mAP)))


def train(cfg, train_loader, test_loader, test_info):
    writer = SummaryWriter(cfg.LOG_PATH)

    print('=> training spn module')
    print('=> test frequency: {} steps'.format(cfg.TEST_FREQ))
    print('=> start training...')

    spn = init_utils.init_spn_model()
    spn = spn.cuda()

    train_spn(spn, cfg, train_loader, writer)

    print('=> training soi module')
    print('=> test frequency: {} steps'.format(cfg.TEST_FREQ))
    print('=> start training...')

    soi = init_utils.init_soi_model()
    soi = soi.cuda()

    net = init_utils.init_model(spn, soi)
    net = net.cuda()

    train_soi(net, cfg, train_loader, test_loader, test_info, writer)


@torch.no_grad()
def test(net, cfg, test_loader, test_info, step, writer=None, model_file=None):
    net.eval()

    # load model from file
    if model_file:
        print('=> loading model: {}'.format(model_file))
        net.load_state_dict(torch.load(model_file))
        print('=> tesing model...')

    final_res = {'method': 'PLACEHOLDER', 'results': {}}
    
    acc = AverageCounter()

    # TEST LOOP
    for data, label, _, vid, vid_num_seg in test_loader:
        data, label = data.cuda(), label.cuda()
        vid_num_seg = vid_num_seg[0].cpu().item()

        video_scores, _, actionness, cas = net(data)

        label_np = label.cpu().data.numpy()
        score_np = video_scores[0].cpu().data.numpy()
        
        pred_np = np.where(score_np < cfg.CLASS_THRESH, 0, 1)
        correct_pred = np.sum(label_np == pred_np, axis=1)
        acc.update(
            float(np.sum((correct_pred == cfg.NUM_CLASSES))),
            correct_pred.shape[0]
        )

        pred = np.where(score_np >= cfg.CLASS_THRESH)[0]
        if len(pred) == 0:
            pred = np.array([np.argmax(score_np)])
        
        cas_pred = utils.get_pred_activations(cas, pred, cfg)
        aness_pred = utils.get_pred_activations(actionness, pred, cfg)
        proposal_dict = utils.get_proposal_dict(
            cas_pred,
            aness_pred,
            pred,
            score_np,
            vid_num_seg,
            cfg
        )

        final_proposals = [utils.nms(v, cfg.NMS_THRESH) for _,v in proposal_dict.items()]
        final_res['results'][vid[0]] = utils.result2json(final_proposals, cfg.classes)

    json_path = os.path.join(cfg.OUTPUT_PATH, 'result.json')
    json.dump(final_res, open(json_path, 'w'))
    
    anet_detection = ANETdetection(
        cfg.GT_PATH,
        json_path,
        subset='test',
        tiou_thresholds=cfg.TIOU_THRESH,
        verbose=False,
    )
    mAP, average_mAP = anet_detection.evaluate()

    if writer:
        writer.add_scalar('Test Performance/Accuracy', acc.avg, step)
        writer.add_scalar('Test Performance/mAP@AVG', average_mAP, step)
        for i in range(cfg.TIOU_THRESH.shape[0]):
            writer.add_scalar('mAP@tIOU/mAP@{:.1f}'.format(
                cfg.TIOU_THRESH[i]),
                mAP[i],
                step)

    test_info["step"].append(step)
    test_info["test_acc"].append(acc.avg)
    test_info["average_mAP"].append(average_mAP)

    for i in range(cfg.TIOU_THRESH.shape[0]):
        test_info["mAP@{:.1f}".format(cfg.TIOU_THRESH[i])].append(mAP[i])

    return test_info['mAP@0.5'][-1], average_mAP

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU_ID

    worker_init_fn = None
    if cfg.SEED >= 0:
        init_utils.set_seed(cfg.SEED)
        worker_init_fn = np.random.seed(cfg.SEED)

    init_utils.set_path(cfg)
    init_utils.save_config(cfg)

    train_loader = torch.utils.data.DataLoader(
        NpyFeature(
            data_path=cfg.DATA_PATH,
            mode='train',
            feature_fps=cfg.FEATS_FPS,
            num_segments=cfg.NUM_SEGMENTS,
            supervision='weak',
            class_dict=cfg.classes,
            seed=cfg.SEED,
            sampling='random'
        ),
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        worker_init_fn=worker_init_fn
    )

    test_loader = torch.utils.data.DataLoader(
        NpyFeature(
            data_path=cfg.DATA_PATH,
            mode='test',
            feature_fps=cfg.FEATS_FPS,
            num_segments=cfg.NUM_SEGMENTS,
            supervision='weak',
            class_dict=cfg.classes,
            seed=cfg.SEED,
            sampling='uniform'
        ),
        batch_size=1,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        worker_init_fn=worker_init_fn
    )
    
    test_info = {
        "step": [],
        "test_acc": [],
        "average_mAP": [],
        "mAP@0.1": [],
        "mAP@0.2": [],
        "mAP@0.3": [], 
        "mAP@0.4": [],
        "mAP@0.5": [],
        "mAP@0.6": [],
        "mAP@0.7": [],
        "mAP@0.8": [],
        "mAP@o.9": []
    }
    
    # TEST
    if cfg.MODE == 'test':
        spn = init_utils.init_spn_model()
        soi = init_utils.init_soi_model()
        net = init_utils.init_model(spn, soi)
        _,_ = test(net, cfg, test_loader, test_info, 0, None, cfg.MODEL_FILE)
        # net, cfg, test_loader, test_info, 0, None, cfg.MODEL_FILE
        return

    # TRAIN
    train(cfg, train_loader, test_loader, test_info)



if __name__ == "__main__":
    assert len(sys.argv)>=2 and sys.argv[1] in ['train', 'test'], 'Please set mode ([train] or [test])'
    cfg.MODE = sys.argv[1]
    if cfg.MODE == 'test':
        assert len(sys.argv) == 3, 'Please set model path'
        cfg.MODEL_FILE = sys.argv[2]
    main()