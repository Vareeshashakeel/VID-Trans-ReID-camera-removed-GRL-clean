import argparse
import os
import random
import time

import numpy as np
import torch
from torch.cuda import amp

from Dataloader import dataloader
from Loss_fun import make_loss
from VID_Test import test
from VID_Trans_model import VID_Trans
from utility import AverageMeter, optimizer, scheduler


def set_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_grl_lambda(epoch, total_epochs, warmup_epochs=15, max_lambda=0.3):
    if epoch <= warmup_epochs:
        return 0.0

    progress = (epoch - warmup_epochs) / max(1, (total_epochs - warmup_epochs))
    progress = min(max(progress, 0.0), 1.0)
    lam = 2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0
    return float(max_lambda * lam)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clean GRL camera-adversarial training')

    parser.add_argument('--Dataset_name', required=True, type=str)
    parser.add_argument('--model_path', required=True, type=str, help='ViT pretrained weight path')
    parser.add_argument('--output_dir', default='./output_camera_removed_grl_clean', type=str)

    parser.add_argument('--epochs', default=120, type=int)
    parser.add_argument('--eval_every', default=10, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--seq_len', default=4, type=int)

    parser.add_argument('--center_w', default=0.0005, type=float)
    parser.add_argument('--cam_loss_w', default=0.02, type=float)
    parser.add_argument('--grl_max_lambda', default=0.3, type=float)
    parser.add_argument('--cam_warmup_epochs', default=15, type=int)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(1234)

    train_loader, _, num_classes, camera_num, view_num, q_val_loader, g_val_loader = dataloader(
        args.Dataset_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seq_len=args.seq_len
    )

    model = VID_Trans(
        num_classes=num_classes,
        camera_num=camera_num,
        pretrainpath=args.model_path
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    loss_fun, center_criterion = make_loss(
        num_classes=num_classes,
        camera_num=camera_num,
        use_gpu=(device == 'cuda')
    )
    center_criterion = center_criterion.to(device)

    optimizer_main = optimizer(model)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=0.5)

    lr_scheduler = scheduler(optimizer_main)
    scaler = amp.GradScaler(enabled=(device == 'cuda'))

    total_loss_meter = AverageMeter()
    idtri_loss_meter = AverageMeter()
    center_loss_meter = AverageMeter()
    cam_loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    best_rank1 = 0.0

    print(
        'GRL camera-adversarial training enabled: True | '
        f'cam_loss_w={args.cam_loss_w:.4f} | '
        f'grl_max_lambda={args.grl_max_lambda:.3f} | '
        f'warmup={args.cam_warmup_epochs} | '
        f'center_w={args.center_w}'
    )

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        total_loss_meter.reset()
        idtri_loss_meter.reset()
        center_loss_meter.reset()
        cam_loss_meter.reset()
        acc_meter.reset()

        lr_scheduler.step(epoch)
        model.train()

        grl_lambda = get_grl_lambda(
            epoch=epoch,
            total_epochs=args.epochs,
            warmup_epochs=args.cam_warmup_epochs,
            max_lambda=args.grl_max_lambda
        )

        for iteration, (img, pid, target_cam, labels2) in enumerate(train_loader, start=1):
            optimizer_main.zero_grad()
            optimizer_center.zero_grad()

            img = img.to(device, non_blocking=True)
            pid = pid.to(device, non_blocking=True)
            labels2 = labels2.to(device, non_blocking=True)

            target_cam = target_cam.to(device, non_blocking=True)
            if target_cam.dim() == 2:
                target_cam_seq = target_cam[:, 0].contiguous().long()
            elif target_cam.dim() == 1:
                if target_cam.size(0) == img.size(0):
                    target_cam_seq = target_cam.contiguous().long()
                else:
                    target_cam_seq = target_cam.view(img.size(0), -1)[:, 0].contiguous().long()
            else:
                raise ValueError(f'Unexpected target_cam shape: {target_cam.shape}')

            # ---------------------------
            # Main loss under AMP
            # ---------------------------
            with amp.autocast(enabled=(device == 'cuda')):
                score, feat, a_vals, cam_logits = model(
                    img,
                    pid,
                    cam_label=target_cam_seq,
                    grl_lambda=grl_lambda
                )

                attn_noise = a_vals * labels2
                attn_loss = attn_noise.sum(1).mean()

                idtri_loss, cam_loss = loss_fun(
                    score,
                    feat,
                    pid,
                    cam_logits=cam_logits,
                    target_cam=target_cam_seq
                )

                main_loss = idtri_loss + attn_loss + args.cam_loss_w * cam_loss

            scaler.scale(main_loss).backward(retain_graph=(args.center_w > 0))
            scaler.step(optimizer_main)
            scaler.update()

            # ---------------------------
            # Center loss in full precision
            # ---------------------------
            if args.center_w > 0:
                center_loss = center_criterion(feat, pid)
                (args.center_w * center_loss).backward()

                for param in center_criterion.parameters():
                    if param.grad is not None:
                        param.grad.data *= (1.0 / args.center_w)

                optimizer_center.step()
            else:
                center_loss = idtri_loss.new_tensor(0.0)

            total_loss = main_loss.detach() + args.center_w * center_loss.detach()

            if isinstance(score, list):
                acc = (score[0].max(1)[1] == pid).float().mean()
            else:
                acc = (score.max(1)[1] == pid).float().mean()

            batch_size = img.shape[0]

            total_loss_meter.update(total_loss.item(), batch_size)
            idtri_loss_meter.update(idtri_loss.item(), batch_size)
            center_loss_meter.update(center_loss.item(), batch_size)
            cam_loss_meter.update(cam_loss.item(), batch_size)
            acc_meter.update(acc.item(), 1)

            if device == 'cuda':
                torch.cuda.synchronize()

            if iteration % 50 == 0:
                print(
                    'Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e} | '
                    'id={:.3f} center={:.3f} cam={:.3f} grl={:.3f}'.format(
                        epoch,
                        iteration,
                        len(train_loader),
                        total_loss_meter.avg,
                        acc_meter.avg,
                        lr_scheduler._get_lr(epoch)[0],
                        idtri_loss_meter.avg,
                        center_loss_meter.avg,
                        cam_loss_meter.avg,
                        grl_lambda
                    )
                )

        print('Epoch {} finished in {:.1f}s'.format(epoch, time.time() - start_time))

        if epoch % args.eval_every == 0:
            model.eval()
            rank1, mAP = test(model, q_val_loader, g_val_loader)
            print('CMC: %.4f, mAP : %.4f' % (rank1, mAP))

            latest_path = os.path.join(
                args.output_dir,
                f'{args.Dataset_name}_camera_removed_grl_latest.pth'
            )
            torch.save(model.state_dict(), latest_path)

            if best_rank1 < rank1:
                best_rank1 = rank1
                best_path = os.path.join(
                    args.output_dir,
                    f'{args.Dataset_name}_camera_removed_grl_best.pth'
                )
                torch.save(model.state_dict(), best_path)
                print(f'[OK] Saved best checkpoint: {best_path}')
