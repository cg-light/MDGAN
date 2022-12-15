# -*- coding: utf-8 -*-
"""
@ Project Name: GAN_exp
@ Author: Jing
@ TIME: 10:58/24/02/2022
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Gradient_Net(nn.Module):
    def __init__(self, channels, cuda):
        super(Gradient_Net, self).__init__()
        single_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        single_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
        kernel_x = []
        kernel_y = []
        for i in range(channels):
            kernel_x.append(single_x)
            kernel_y.append(single_y)

        if cuda:
            kernel_x = torch.cuda.FloatTensor(kernel_x).unsqueeze(0)
            kernel_y = torch.cuda.FloatTensor(kernel_y).unsqueeze(0)
        else:
            kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0)
            kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0)

        kernel_x.requires_grad = False
        kernel_y.requires_grad = False
        self.weight_x = kernel_x
        self.weight_y = kernel_y

    def forward(self, x):
        grad_x = F.conv2d(x, self.weight_x, stride=1)
        grad_y = F.conv2d(x, self.weight_y, stride=1)
        gradient = torch.abs(grad_x) + torch.abs(grad_y)
        return gradient


def norm(x, beta, gama):
    mu = torch.mean(x, dim=1, keepdim=True)
    seta = torch.mean((x ** 2 - mu ** 2), dim=1, keepdim=True).sqrt()
    out = gama * (x - mu) / seta + beta
    return out


##############################
#           U-NET
##############################

class BRM(nn.Module):
    """attention block"""
    def __init__(self, in_size, outc, ks, latent_dim=6, norm=False):
        super(BRM, self).__init__()
        channels, in_h, in_w = in_size
        self.out_c, self.h, self.w, self.c = outc, in_h, in_w, channels

        self.down_sample = nn.AvgPool2d(kernel_size=ks, stride=ks)
        layers = [nn.Conv2d(channels, outc, 3, 1, 1)]
        if norm:
            layers.append(nn.InstanceNorm2d(outc))
        layers.append(nn.Sigmoid())
        self.m = nn.Sequential(*layers)
        self.bg_down = nn.Upsample(scale_factor=1 / ks)
        if channels != outc:
            # self.pool = nn.AdaptiveAvgPool2d(1)
            self.use1x1 = nn.Conv2d(channels, outc, 1, 1, 0)
        # nn.Conv2d(outc, outc, 3, 1, 1))

    def forward(self, bg, mask, x):
        mask = self.down_sample(mask)
        atten_m = self.m(mask)
        bg = self.bg_down(bg)
        if self.c != atten_m.size(1):
            bg = self.use1x1(bg)
        return atten_m * x + (1 - atten_m) * bg, atten_m


class maskDown(nn.Module):
    def __init__(self, img_shape, inc, outc, ks, latent_dim=8, normalize=True, mask=True, noise=False, kernel_size=3,
                 dropout=0.1):
        super(maskDown, self).__init__()
        self.mask = mask
        self.c = img_shape[0]
        layers = [nn.Conv2d(inc, outc, kernel_size, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(outc, 0.8))
        # self.act = nn.LeakyReLU(0.2, inplace=True)
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.model = nn.Sequential(*layers,
                                   nn.Dropout(dropout))
        if mask:
            self.m = BRM(img_shape, outc, ks)
        if noise:
            self.insert_noise = nn.Linear(self.c, outc)
        else:
            self.insert_noise = None

    def forward(self, bg, x, z, mask):
        feature = self.model(x)
        if self.mask:
            feature, _ = self.m(bg, mask, feature)
            if self.insert_noise is not None:
                noise = torch.rand(feature.shape[0], feature.shape[2], feature.shape[3], self.c).cuda()
                noise = self.insert_noise(noise).permute(0, 3, 1, 2)
                feature = feature + noise
        return feature


class maskUup(nn.Module):
    def __init__(self, img_shape, inc, outc, ks, mask=True, noise=False):
        super(maskUup, self).__init__()
        self.mask = mask
        self.c = img_shape[0]
        self.model = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(inc, outc, 3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(outc, 0.8),
            nn.LeakyReLU(0.2, inplace=True))
        # self.act = nn.LeakyReLU(0.2, inplace=True)
        if mask:
            self.m = BRM(img_shape, outc, ks)
        if noise:
            self.insert_noise = nn.Linear(self.c, outc)
        else:
            self.insert_noise = None

    def forward(self, bg, x, skip_input, z, mask):
        feature = self.model(x)
        if self.mask:
            feature, _ = self.m(bg, mask, feature)
            if self.insert_noise is not None:
                noise = torch.rand(feature.shape[0], feature.shape[2], feature.shape[3], self.c).cuda()
                noise = self.insert_noise(noise).permute(0, 3, 1, 2)
                feature = feature + noise
        return torch.cat((feature, skip_input), 1)


class UGenerator(nn.Module):
    def __init__(self, img_shape, latent_dim, m=True):
        super(UGenerator, self).__init__()
        channels, h, w = img_shape
        self.latent_dim, self.c, self.h, self.w = latent_dim, channels, h, w

        self.fc = nn.Linear(latent_dim, channels * self.h * self.w)  # 输入噪声映射为图片大小

        self.down1 = maskDown([channels, h // 2, w // 2], channels, 64, 2, normalize=False)
        self.down2 = maskDown([channels, h // 4, w // 4], 64, 128, 4, mask=m)
        self.down3 = maskDown([channels, h // 8, w // 8], 128, 256, 8, mask=m)
        self.down4 = maskDown([channels, h // 16, w // 16], 256, 256, 16, mask=m)
        self.down5 = maskDown([channels, h // 32, w // 32], 256, 512, 32, mask=m)
        self.down6 = maskDown([channels, h // 64, w // 64], 512, 512, 64, normalize=False, mask=False)
        self.up1 = maskUup([channels, h // 32, w // 32], 512, 512, 32, mask=False)
        self.up2 = maskUup([channels, h // 16, w // 16], 1024, 256, 16, mask=m)
        self.up3 = maskUup([channels, h // 8, w // 8], 512, 256, 8, mask=m)
        self.up4 = maskUup([channels, h // 4, w // 4], 512, 128, 4, mask=m)
        self.up5 = maskUup([channels, h // 2, w // 2], 256, 64, 2, mask=m)
        if m:
            self.final_m = BRM([channels, h, w], channels, 1, norm=False)
        else:
            self.final_m = None

        self.final = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(128, channels, 3, stride=1, padding=1),
                                   nn.Tanh())  #

    def forward(self, x, z, mask):
        z_x = self.fc(z).view(z.size(0), self.c, self.h, self.w)
        bg = x
        x = z_x * mask + x
        d1 = self.down1(bg, x, z, mask)
        d2 = self.down2(bg, d1, z, mask)
        d3 = self.down3(bg, d2, z, mask)
        d4 = self.down4(bg, d3, z, mask)
        d5 = self.down5(bg, d4, z, mask)
        d6 = self.down6(bg, d5, z, mask)
        u1 = self.up1(bg, d6, d5, z, mask)
        u2 = self.up2(bg, u1, d4, z, mask)
        u3 = self.up3(bg, u2, d3, z, mask)
        u4 = self.up4(bg, u3, d2, z, mask)
        u5 = self.up5(bg, u4, d1, z, mask)

        final_feature = self.final(u5)
        if self.final_m is not None:
            out, final_m = self.final_m(bg, mask, final_feature)
        else:
            out = final_feature
            final_m = (mask + 1) / 2
        return final_feature, out, final_m#,


class final_atten(nn.Module):
    def __init__(self, img_shape, ksize):
        super(final_atten, self).__init__()
        channels, h, w = img_shape

        self.circle_m = BRM([channels, h, w], channels, 1, norm=False)
        self.e_m = BRM([channels, h, w], channels, 1, norm=False)
        if ksize > 1:
            self.dilate_erode = nn.MaxPool2d(kernel_size=ksize, stride=1, padding=int((ksize - 1) // 2))
        else:
            self.dilate_erode = None

    def forward(self, z, mask, x):
        if self.dilate_erode is not None:
            mask_e = -self.dilate_erode(-mask)
        else:
            mask_e = mask
        mask_c = mask - mask_e
        circle_m = self.circle_m(z, mask_c, x)
        e_m = self.e_m(z, mask_e, x)
        return circle_m + e_m


class DMask(nn.Module):
    def __init__(self, in_size, outc, ks, norm=True):
        super(DMask, self).__init__()
        channels, in_h, in_w = in_size
        self.h = in_h

        self.down_sample = nn.AvgPool2d(kernel_size=ks, stride=ks)
        layers = [nn.Conv2d(channels, outc, 3, 1, 1)]
        if norm:
            layers.append(nn.InstanceNorm2d(outc))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.m = nn.Sequential(*layers)
        # nn.Conv2d(outc, outc, 3, 1, 1))

    def forward(self, mask):
        feature = self.m(self.down_sample(mask))
        return feature


class Dblock(nn.Module):
    def __init__(self, in_size, outc, ks, normalize=True, mask=False, mask_norm=True):
        super(Dblock, self).__init__()
        self.mask = mask
        channels, inc, h, w = in_size
        layers = [nn.Conv2d(inc, outc, 4, 2, 1)]
        if normalize:
            layers.append(nn.InstanceNorm2d(outc, 0.8))
        layers.append(nn.ReLU())
        if mask:
            self.m = DMask([channels, h, w], outc, ks, mask_norm)
        self.models = nn.Sequential(*layers)

    def forward(self, x, mask):
        feature = self.models(x)
        if self.mask:
            output = torch.cat(((self.m(mask) * feature), feature), 1)
            return output
        else:
            return feature


class Discriminator(nn.Module):
    def __init__(self, in_size):
        super(Discriminator, self).__init__()
        channels, h, w = in_size

        self.models = nn.ModuleList([
            Dblock([channels, channels, h, w], 32, 2, mask=True, mask_norm=False),
            Dblock([channels, 64, h, w], 128, 4, mask=True),
            Dblock([channels, 256, h, w], 256, 8, mask=True),
            Dblock([channels, 512, h, w], 256, 16, mask=True),
            Dblock([channels, 512, h, w], 512, 32),
        ])  # 4*4
        self.outlayer = nn.Conv2d(512, 1, 3, 2, 1)

        # self.downsample = nn.AvgPool2d()

    def compute_loss(self, x, mask, gt):
        loss = sum([torch.mean((out - gt) ** 2) for out in self.forward(x, mask)])
        return loss

    def forward(self, x, mask):
        output = None
        # defect = x * mask
        # x = torch.cat((mask, x), 1)
        for model in self.models:
            output = model(x, mask)
            x = output
        output = self.outlayer(x)
        return output