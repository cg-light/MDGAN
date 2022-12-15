# -*- coding: utf-8 -*-
"""
@ Project Name: GAN_exp
@ Author: Jing
@ TIME: 10:58/24/02/2022
"""
import torch
import os
import glob
import os
import random

import numpy as np
import math
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as ttf


def rand_perlin_2d(shape, res, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    grid = torch.stack(torch.meshgrid(torch.arange(0, res[0], delta[0]), torch.arange(0, res[1], delta[1])), dim=-1) % 1
    angles = 2 * math.pi * torch.rand(res[0] + 1, res[1] + 1)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    tile_grads = lambda slice1, slice2: gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]].repeat_interleave(d[0],
                                                                                                              0).repeat_interleave(
        d[1], 1)

    dot = lambda grad, shift: (
            torch.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]),
                        dim=-1) * grad[:shape[0], :shape[1]]).sum(dim=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[:shape[0], :shape[1]])
    noise = math.sqrt(2) * torch.lerp(torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1])
    noise[noise > 0.2] = 1.0
    noise[noise != 1] = 0.0
    return 1.0 - noise


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


def filp(img1, img2, img3):
    if random.random() < 0.5:
        img1 = ttf.hflip(img1)
        img2 = ttf.hflip(img2)
        img3 = ttf.hflip(img3)
    elif random.random() > 0.5:
        img1 = ttf.vflip(img1)
        img2 = ttf.vflip(img2)
        img3 = ttf.vflip(img3)

    # img1 = ttf.to_tensor(img1)
    # img2 = ttf.to_tensor(img2)
    return img1, img2, img3


def rotate(img1, img2, img3):
    angle = transforms.RandomRotation.get_params([-180, 180])
    img1 = img1.rotate(angle)
    img2 = img2.rotate(angle)
    img3 = img3.rotate(angle)

    return img1, img2, img3


def crop(img1, img2, img3, in_size):
    w, h = int(1.15 * in_size[0]), int(1.15 * in_size[1])
    nums_i = int(0.15 * in_size[0])
    nums_j = int(0.15 * in_size[1])
    i = random.randint(0, nums_i)
    j = random.randint(0, nums_j)
    img1 = ttf.resize(img1, [w, h])
    img2 = ttf.resize(img2, [w, h])
    img3 = ttf.resize(img3, [w, h])
    img1 = ttf.crop(img1, i, j, in_size[0], in_size[1])
    img2 = ttf.crop(img2, i, j, in_size[0], in_size[1])
    img3 = ttf.crop(img3, i, j, in_size[0], in_size[1])
    return img1, img2, img3


class ImageDataset(Dataset):
    def __init__(self, root, input_shape, mode="train", unaligned=False):
        self.input_shape = input_shape
        self.mode = mode
        self.unaligned = unaligned
        transform = [transforms.Resize((input_shape[1], input_shape[2]), Image.BICUBIC),
                     transforms.ToTensor()]  # (0, 255)->(0, 1)
        if input_shape[0] == 1:
            transform.append(transforms.Normalize((0.5,), (0.5,)))
            # norm = transforms.Normalize((0.5,), (0.5,))
        else:
            # norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            transform.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))  # (0, 1) -> (-1, 1)
        # self.norm = norm
        self.transform = transforms.Compose(transform)
        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
        # print(os.path.join(root, mode))

    def __getitem__(self, index):
        if self.unaligned:
            all = Image.open(self.files[random.randint(0, (len(self.files) - 1))])
        else:
            all = Image.open(self.files[index % len(self.files)])

        if all.mode != "RGB" and self.input_shape[0] == 3:
            all = to_rgb(all)
        if all.mode == "RGB" and self.input_shape[0] == 1:
            all = all.convert('L')
        w, h = all.size

        if 'train' in self.mode:
            img_real = all.crop((0, 0, w / 3, h))
            img_b = all.crop((w / 3, 0, 2 * w / 3, h))
            mask = all.crop((2 * w / 3, 0, w, h))
            if random.random() < 0.25:
                img_real, img_b, mask = filp(img_real, img_b, mask)
            elif 0.25 <= random.random() < 0.5:
                img_real, img_b, mask = rotate(img_real, img_b, mask)
            elif 0.5 <= random.random() < 0.75:
                img_real, img_b, mask = crop(img_real, img_b, mask, [self.input_shape[1], self.input_shape[2]])

            img_real = self.transform(img_real)
            img_b = self.transform(img_b)
            mask = self.transform(mask)

            return {'R': img_real, 'B': img_b, 'M': mask}
        else:
            img_b = all.crop((0, 0, w / 2, h))
            mask = all.crop((w / 2, 0, w, h))
            img_b = self.transform(img_b)
            mask = self.transform(mask)

            return {'B': img_b, 'M': mask}

    def __len__(self):
        return len(self.files)
