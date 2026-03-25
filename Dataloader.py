import random
import numpy as np
from PIL import Image, ImageFile

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

from utility import RandomIdentitySampler, RandomErasing3
from Datasets.MARS_dataset import Mars
from Datasets.iLDSVID import iLIDSVID
from Datasets.PRID_dataset import PRID

ImageFile.LOAD_TRUNCATED_IMAGES = True

__factory = {
    'Mars': Mars,
    'iLIDSVID': iLIDSVID,
    'PRID': PRID,
}


def train_collate_fn(batch):
    imgs, pids, camids, labels = zip(*batch)
    return torch.stack(imgs, dim=0), torch.tensor(pids, dtype=torch.int64), torch.tensor(camids, dtype=torch.int64), torch.stack(labels, dim=0)


def val_collate_fn(batch):
    imgs, pids, camids, img_paths = zip(*batch)
    return torch.stack(imgs, dim=0), pids, torch.tensor(camids, dtype=torch.int64), img_paths


def dataloader(Dataset_name, batch_size=64, num_workers=4, seq_len=4):
    train_transforms = T.Compose([
        T.Resize([256, 128], interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop([256, 128]),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    val_transforms = T.Compose([
        T.Resize([256, 128]),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    dataset = __factory[Dataset_name]()
    train_set = VideoDataset_inderase(dataset.train, seq_len=seq_len, sample='intelligent', transform=train_transforms)
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=RandomIdentitySampler(dataset.train, batch_size, 4),
        num_workers=num_workers,
        collate_fn=train_collate_fn,
        pin_memory=True,
    )

    q_val_set = VideoDataset(dataset.query, seq_len=seq_len, sample='dense', transform=val_transforms)
    g_val_set = VideoDataset(dataset.gallery, seq_len=seq_len, sample='dense', transform=val_transforms)

    q_val_loader = DataLoader(q_val_set, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=val_collate_fn, pin_memory=True)
    g_val_loader = DataLoader(g_val_set, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=val_collate_fn, pin_memory=True)

    return train_loader, len(dataset.query), num_classes, cam_num, view_num, q_val_loader, g_val_loader


def read_image(img_path):
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


def _pad_indices(indices, seq_len):
    indices = list(indices)
    if len(indices) == 0:
        raise RuntimeError('Empty tracklet encountered.')
    while len(indices) < seq_len:
        indices.append(indices[-1])
    return indices


class VideoDataset(Dataset):
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset, seq_len=15, sample='evenly', transform=None, max_length=40):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)

        if self.sample == 'random':
            frame_indices = list(range(num))
            rand_end = max(0, len(frame_indices) - self.seq_len)
            begin_index = random.randint(0, rand_end)
            indices = frame_indices[begin_index:begin_index + self.seq_len]
            indices = _pad_indices(indices, self.seq_len)
            imgs, targt_cam = [], []
            for idx in indices:
                img = read_image(img_paths[int(idx)])
                if self.transform is not None:
                    img = self.transform(img)
                imgs.append(img.unsqueeze(0))
                targt_cam.append(camid)
            imgs = torch.cat(imgs, dim=0)
            return imgs, pid, targt_cam

        elif self.sample == 'dense':
            cur_index = 0
            frame_indices = list(range(num))
            indices_list = []
            while num - cur_index > self.seq_len:
                indices_list.append(frame_indices[cur_index:cur_index + self.seq_len])
                cur_index += self.seq_len
            last_seq = frame_indices[cur_index:]
            last_seq = _pad_indices(last_seq, self.seq_len)
            indices_list.append(last_seq)

            imgs_list = []
            for indices in indices_list:
                if len(imgs_list) > self.max_length:
                    break
                imgs = []
                for idx in indices:
                    img = read_image(img_paths[int(idx)])
                    if self.transform is not None:
                        img = self.transform(img)
                    imgs.append(img.unsqueeze(0))
                imgs = torch.cat(imgs, dim=0)
                imgs_list.append(imgs)
            imgs_array = torch.stack(imgs_list)
            return imgs_array, pid, camid, img_paths

        elif self.sample == 'dense_subset':
            frame_indices = list(range(num))
            rand_end = max(0, len(frame_indices) - self.max_length - 1)
            begin_index = random.randint(0, rand_end)
            cur_index = begin_index
            indices_list = []
            while num - cur_index > self.seq_len:
                indices_list.append(frame_indices[cur_index:cur_index + self.seq_len])
                cur_index += self.seq_len
            last_seq = frame_indices[cur_index:]
            last_seq = _pad_indices(last_seq, self.seq_len)
            indices_list.append(last_seq)
            imgs_list = []
            for indices in indices_list:
                if len(imgs_list) > self.max_length:
                    break
                imgs = []
                for idx in indices:
                    img = read_image(img_paths[int(idx)])
                    if self.transform is not None:
                        img = self.transform(img)
                    imgs.append(img.unsqueeze(0))
                imgs = torch.cat(imgs, dim=0)
                imgs_list.append(imgs)
            imgs_array = torch.stack(imgs_list)
            return imgs_array, pid, camid

        elif self.sample == 'intelligent_random':
            indices = []
            each = max(num // self.seq_len, 1)
            for i in range(self.seq_len):
                if i != self.seq_len - 1:
                    indices.append(random.randint(min(i * each, num - 1), min((i + 1) * each - 1, num - 1)))
                else:
                    indices.append(random.randint(min(i * each, num - 1), num - 1))
            imgs = []
            for idx in indices:
                img = read_image(img_paths[int(idx)])
                if self.transform is not None:
                    img = self.transform(img)
                imgs.append(img.unsqueeze(0))
            imgs = torch.cat(imgs, dim=0)
            return imgs, pid, camid
        else:
            raise KeyError('Unknown sample method: {}.'.format(self.sample))


class VideoDataset_inderase(Dataset):
    def __init__(self, dataset, seq_len=15, sample='evenly', transform=None, max_length=40):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform
        self.max_length = max_length
        self.erase = RandomErasing3(probability=0.5, mean=[0.485, 0.456, 0.406])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)
        if self.sample != 'intelligent':
            frame_indices = list(range(num))
            rand_end = max(0, len(frame_indices) - self.seq_len)
            begin_index = random.randint(0, rand_end)
            indices = frame_indices[begin_index:begin_index + self.seq_len]
            indices = _pad_indices(indices, self.seq_len)
        else:
            indices = []
            each = max(num // self.seq_len, 1)
            for i in range(self.seq_len):
                if i != self.seq_len - 1:
                    indices.append(random.randint(min(i * each, num - 1), min((i + 1) * each - 1, num - 1)))
                else:
                    indices.append(random.randint(min(i * each, num - 1), num - 1))

        imgs, labels, targt_cam = [], [], []
        for idx in indices:
            img = read_image(img_paths[int(idx)])
            if self.transform is not None:
                img = self.transform(img)
            img, temp = self.erase(img)
            labels.append(temp)
            imgs.append(img.unsqueeze(0))
            targt_cam.append(camid)

        labels = torch.tensor(labels)
        imgs = torch.cat(imgs, dim=0)
        return imgs, pid, targt_cam, labels
