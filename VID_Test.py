import argparse
import numpy as np
import torch
from torch.autograd import Variable

from Dataloader import dataloader
from VID_Trans_model import VID_Trans


def evaluate(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=21):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print('Note: number of gallery samples is quite small, got {}'.format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc = []
    all_AP = []
    num_valid_q = 0.
    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            continue
        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    return all_cmc, mAP


def test(model, queryloader, galleryloader, pool='avg', use_gpu=True, ranks=[1, 5, 10, 20]):
    model.eval()
    qf, q_pids, q_camids = [], [], []
    with torch.no_grad():
        for _, (imgs, pids, camids, _) in enumerate(queryloader):
            if use_gpu:
                imgs = imgs.cuda()
            imgs = Variable(imgs)
            imgs = imgs.squeeze(0)
            num_clips = imgs.size(0)
            features = model(imgs, None)
            features = features.view(num_clips, -1)
            features = torch.mean(features, 0)
            qf.append(features.cpu())
            q_pids.append(int(pids[0]))
            q_camids.append(int(camids.view(-1)[0].item()))

        qf = torch.stack(qf)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)
        print('Extracted features for query set, obtained {}-by-{} matrix'.format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        for _, (imgs, pids, camids, _) in enumerate(galleryloader):
            if use_gpu:
                imgs = imgs.cuda()
            imgs = Variable(imgs)
            imgs = imgs.squeeze(0)
            num_clips = imgs.size(0)
            features = model(imgs, None)
            features = features.view(num_clips, -1)
            if pool == 'avg':
                features = torch.mean(features, 0)
            else:
                features, _ = torch.max(features, 0)
            gf.append(features.cpu())
            g_pids.append(int(pids[0]))
            g_camids.append(int(camids.view(-1)[0].item()))

    gf = torch.stack(gf)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)
    print('Extracted features for gallery set, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))
    print('Computing distance matrix')
    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(qf, gf.t(), beta=1, alpha=-2)
    distmat = distmat.numpy()
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    print('Results ----------')
    print('mAP: {:.1%}'.format(mAP))
    print('CMC curve')
    print('Rank-1  : {:.1%}'.format(cmc[0]))
    if len(cmc) > 4: print('Rank-5  : {:.1%}'.format(cmc[4]))
    if len(cmc) > 9: print('Rank-10 : {:.1%}'.format(cmc[9]))
    if len(cmc) > 19: print('Rank-20 : {:.1%}'.format(cmc[19]))
    return cmc[0], mAP


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VID-Trans-ReID no-camera baseline test')
    parser.add_argument('--Dataset_name', required=True, type=str)
    parser.add_argument('--model_path', required=True, type=str)
    args = parser.parse_args()

    _, _, num_classes, camera_num, _, q_val_loader, g_val_loader = dataloader(args.Dataset_name)
    model = VID_Trans(num_classes=num_classes, camera_num=camera_num, pretrainpath=None).cuda()
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint, strict=True)
    cmc, mAP = test(model, q_val_loader, g_val_loader)
    print('CMC: %.4f, mAP : %.4f' % (cmc, mAP))
