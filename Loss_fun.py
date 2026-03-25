import torch
import torch.nn as nn

from loss.softmax_loss import CrossEntropyLabelSmooth
from loss.triplet_loss import TripletLoss
from loss.center_loss import CenterLoss


class CombinedCenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim_global=768, feat_dim_local=3072, use_gpu=True):
        super().__init__()
        self.global_center = CenterLoss(
            num_classes=num_classes,
            feat_dim=feat_dim_global,
            use_gpu=use_gpu
        )
        self.local_centers = nn.ModuleList([
            CenterLoss(
                num_classes=num_classes,
                feat_dim=feat_dim_local,
                use_gpu=use_gpu
            )
            for _ in range(4)
        ])

    def forward(self, feat, target):
        if isinstance(feat, list):
            global_loss = self.global_center(feat[0], target)

            local_losses = []
            for c, f in zip(self.local_centers, feat[1:]):
                local_losses.append(c(f, target))

            local_loss = sum(local_losses) / max(1, len(local_losses))
            return 0.75 * global_loss + 0.25 * local_loss

        return self.global_center(feat, target)


def make_loss(num_classes, camera_num=0, use_gpu=True):
    center_criterion = CombinedCenterLoss(num_classes=num_classes, use_gpu=use_gpu)

    triplet = TripletLoss()
    xent = CrossEntropyLabelSmooth(num_classes=num_classes)

    cam_xent = None
    if camera_num is not None and camera_num > 1:
        cam_xent = CrossEntropyLabelSmooth(num_classes=camera_num)

    def loss_func(score, feat, target, cam_logits=None, target_cam=None):
        # -----------------------------
        # ID classification loss
        # -----------------------------
        if isinstance(score, list):
            id_loss_list = [xent(scor, target) for scor in score[1:]]
            id_loss_local = sum(id_loss_list) / len(id_loss_list)
            id_loss = 0.25 * id_loss_local + 0.75 * xent(score[0], target)
        else:
            id_loss = xent(score, target)

        # -----------------------------
        # Triplet loss
        # -----------------------------
        if isinstance(feat, list):
            tri_loss_list = [triplet(fea, target)[0] for fea in feat[1:]]
            tri_loss_local = sum(tri_loss_list) / len(tri_loss_list)
            tri_loss = 0.25 * tri_loss_local + 0.75 * triplet(feat[0], target)[0]
        else:
            tri_loss = triplet(feat, target)[0]

        # -----------------------------
        # Camera adversarial loss
        # IMPORTANT: target_cam must be sequence-level [B]
        # cam_logits must be [B, camera_num]
        # -----------------------------
        if cam_logits is not None and cam_xent is not None and target_cam is not None:
            target_cam = target_cam.long().view(-1)
            cam_loss = cam_xent(cam_logits, target_cam)
        else:
            cam_loss = id_loss.new_tensor(0.0)

        return id_loss + tri_loss, cam_loss

    return loss_func, center_criterion
