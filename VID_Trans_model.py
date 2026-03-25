import copy
from functools import partial

import torch
import torch.nn as nn
from torch.nn import functional as F

from vit_ID import Block, TransReID


def TCSS(features, shift, b, t):
    features = features.view(b, features.size(1), t * features.size(2))
    token = features[:, 0:1]

    batchsize = features.size(0)
    dim = features.size(-1)

    features = torch.cat([features[:, shift:], features[:, 1:shift]], dim=1)

    try:
        features = features.view(batchsize, 2, -1, dim)
    except Exception:
        features = torch.cat([features, features[:, -2:-1, :]], dim=1)
        features = features.view(batchsize, 2, -1, dim)

    features = torch.transpose(features, 1, 2).contiguous()
    features = features.view(batchsize, -1, dim)
    return features, token


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class GradientReverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x, lambd=1.0):
    return GradientReverseFunction.apply(x, lambd)


class CameraAdversary(nn.Module):
    def __init__(self, in_dim, camera_num, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, camera_num),
        )
        self.net.apply(weights_init_kaiming)
        self.net[-1].apply(weights_init_classifier)

    def forward(self, x, grl_lambda=1.0):
        x = grad_reverse(x, grl_lambda)
        return self.net(x)


class VID_Trans(nn.Module):
    def __init__(self, num_classes, camera_num=None, pretrainpath=None, cam_hidden_dim=256):
        super(VID_Trans, self).__init__()
        self.in_planes = 768
        self.num_classes = num_classes
        self.camera_num = int(camera_num) if camera_num is not None else 0

        self.base = TransReID(
            img_size=[256, 128], patch_size=16, stride_size=[16, 16],
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
            drop_path_rate=0.1, drop_rate=0.0, attn_drop_rate=0.0,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )

        if pretrainpath is not None:
            state_dict = torch.load(pretrainpath, map_location='cpu')
            self.base.load_param(state_dict, load=True)

        block = self.base.blocks[-1]
        layer_norm = self.base.norm
        self.b1 = nn.Sequential(copy.deepcopy(block), copy.deepcopy(layer_norm))

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        dpr = [x.item() for x in torch.linspace(0, 0, 12)]
        self.block1 = Block(
            dim=3072, num_heads=12, mlp_ratio=4, qkv_bias=True, qk_scale=None,
            drop=0, attn_drop=0, drop_path=dpr[11],
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )
        self.b2 = nn.Sequential(self.block1, nn.LayerNorm(3072))

        self.bottleneck_1 = nn.BatchNorm1d(3072)
        self.bottleneck_1.bias.requires_grad_(False)
        self.bottleneck_1.apply(weights_init_kaiming)
        self.bottleneck_2 = nn.BatchNorm1d(3072)
        self.bottleneck_2.bias.requires_grad_(False)
        self.bottleneck_2.apply(weights_init_kaiming)
        self.bottleneck_3 = nn.BatchNorm1d(3072)
        self.bottleneck_3.bias.requires_grad_(False)
        self.bottleneck_3.apply(weights_init_kaiming)
        self.bottleneck_4 = nn.BatchNorm1d(3072)
        self.bottleneck_4.bias.requires_grad_(False)
        self.bottleneck_4.apply(weights_init_kaiming)

        self.classifier_1 = nn.Linear(3072, self.num_classes, bias=False)
        self.classifier_1.apply(weights_init_classifier)
        self.classifier_2 = nn.Linear(3072, self.num_classes, bias=False)
        self.classifier_2.apply(weights_init_classifier)
        self.classifier_3 = nn.Linear(3072, self.num_classes, bias=False)
        self.classifier_3.apply(weights_init_classifier)
        self.classifier_4 = nn.Linear(3072, self.num_classes, bias=False)
        self.classifier_4.apply(weights_init_classifier)

        self.middle_dim = 256
        self.attention_conv = nn.Conv2d(self.in_planes, self.middle_dim, [1, 1])
        self.attention_tconv = nn.Conv1d(self.middle_dim, 1, 3, padding=1)
        self.attention_conv.apply(weights_init_kaiming)
        self.attention_tconv.apply(weights_init_kaiming)

        self.shift_num = 5
        self.camera_classifier = None
        if self.camera_num > 1:
            self.camera_classifier = CameraAdversary(self.in_planes, self.camera_num, hidden_dim=cam_hidden_dim)

    def forward(self, x, label=None, cam_label=None, view_label=None, grl_lambda=0.0):
        b = x.size(0)
        t = x.size(1)

        x = x.view(x.size(0) * x.size(1), x.size(2), x.size(3), x.size(4))
        features = self.base(x)

        b1_feat = self.b1(features)
        global_feat = b1_feat[:, 0]

        global_feat = global_feat.unsqueeze(dim=2).unsqueeze(dim=3)
        a = F.relu(self.attention_conv(global_feat))
        a = a.view(b, t, self.middle_dim)
        a = a.permute(0, 2, 1)
        a = F.relu(self.attention_tconv(a))
        a = a.view(b, t)
        a_vals = a

        a = F.softmax(a, dim=1)
        x = global_feat.view(b, t, -1)
        a = torch.unsqueeze(a, -1).expand(b, t, self.in_planes)
        att_x = torch.mul(x, a)
        att_x = torch.sum(att_x, 1)

        global_feat = att_x.view(b, self.in_planes)
        global_bn_feat = self.bottleneck(global_feat)

        feature_length = features.size(1) - 1
        patch_length = feature_length // 4
        x, token = TCSS(features, self.shift_num, b, t)

        part1 = x[:, :patch_length]
        part1 = self.b2(torch.cat((token, part1), dim=1))
        part1_f = part1[:, 0]

        part2 = x[:, patch_length:patch_length * 2]
        part2 = self.b2(torch.cat((token, part2), dim=1))
        part2_f = part2[:, 0]

        part3 = x[:, patch_length * 2:patch_length * 3]
        part3 = self.b2(torch.cat((token, part3), dim=1))
        part3_f = part3[:, 0]

        part4 = x[:, patch_length * 3:patch_length * 4]
        part4 = self.b2(torch.cat((token, part4), dim=1))
        part4_f = part4[:, 0]

        part1_bn = self.bottleneck_1(part1_f)
        part2_bn = self.bottleneck_2(part2_f)
        part3_bn = self.bottleneck_3(part3_f)
        part4_bn = self.bottleneck_4(part4_f)

        if self.training:
            global_id = self.classifier(global_bn_feat)
            local_id1 = self.classifier_1(part1_bn)
            local_id2 = self.classifier_2(part2_bn)
            local_id3 = self.classifier_3(part3_bn)
            local_id4 = self.classifier_4(part4_bn)

            cam_logits = None
            if self.camera_classifier is not None:
                cam_logits = self.camera_classifier(global_bn_feat, grl_lambda=grl_lambda)

            return (
                [global_id, local_id1, local_id2, local_id3, local_id4],
                [global_feat, part1_f, part2_f, part3_f, part4_f],
                a_vals,
                cam_logits,
            )
        else:
            return torch.cat([global_bn_feat, part1_bn / 4, part2_bn / 4, part3_bn / 4, part4_bn / 4], dim=1)

    def load_param(self, trained_path, load=False):
        if not load:
            param_dict = torch.load(trained_path)
            for i in param_dict:
                self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
            print('Loading pretrained model from {}'.format(trained_path))
        else:
            param_dict = trained_path
            for i in param_dict:
                if i in self.state_dict() and 'classifier' not in i and 'sie_embed' not in i and 'camera_classifier' not in i:
                    self.state_dict()[i].copy_(param_dict[i])

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))
