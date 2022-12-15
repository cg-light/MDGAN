# -*- coding: utf-8 -*-
"""
@ Project Name: GAN_exp
@ Author: Jing
@ TIME: 10:57/24/02/2022
"""
import itertools
import sys
import time
import cv2
import torch
import numpy as np
import argparse
import datetime
import os

from torch import autograd
from torchvision.utils import save_image
import torch.nn.functional as FF
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader, DistributedSampler

from pytorch_ssim import pytorch_ssim
from models import *
from datasets2 import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, default="../../paras/MDGAN-2", help="saved path")
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=401, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="grid", help="name of the dataset")
parser.add_argument("--defect_name", type=str, default="glue", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=20, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0004, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--res_num", type=int, default=4)
parser.add_argument("--latent_dim", type=int, default=8)
parser.add_argument("--noise", type=str, default=False)
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--lambda_r", type=float, default=5, help="weight of generated background and real background")
parser.add_argument("--lambda_grad", type=float, default=10, help="weight of generated background and real background")
parser.add_argument("--lambda_gp", type=float, default=10, help="weight of reconstruct defect image and real image")
parser.add_argument("--lambda_diver", type=float, default=15, help="weight of reconstruct defect image and real image")
parser.add_argument("--lambda_5", type=float,  default=1)
parser.add_argument("--checkpoint_interval", type=int, default=2, help="interval between model checkpoints")
parser.add_argument("--sample_interval", type=int, default=1000, help="interval between test checkpoints")
opt = parser.parse_args()

print(opt)
file_path = os.path.join(opt.filename, opt.dataset_name, opt.defect_name)
os.makedirs(file_path, exist_ok=True)
f = open(file_path + '//loss %f .txt' % opt.epoch, 'w')

img_path = "%s/images/%s/%s" % (opt.filename, opt.dataset_name, opt.defect_name)
os.makedirs(img_path, exist_ok=True)
model_path = "%s/saved_models/%s/%s" % (opt.filename, opt.dataset_name, opt.defect_name)
os.makedirs(model_path, exist_ok=True)

img_shape = [opt.channels, opt.img_height, opt.img_width]
L1 = torch.nn.L1Loss()
L2 = torch.nn.MSELoss()

G = UGenerator(img_shape, opt.latent_dim, True)
D = Discriminator(img_shape)
cuda = True if torch.cuda.is_available() else False
grad = Gradient_Net(opt.channels, cuda)
div = Gradient_Net(1, cuda)

if cuda:
    G = G.cuda()
    D = D.cuda()
    grad = grad.cuda()
    div = div.cuda()
    L1.cuda()
    L2.cuda()

if opt.epoch != 0:
    G.load_state_dict(torch.load(model_path + "/G_%d.pth" % opt.epoch))
    D.load_state_dict((torch.load(model_path + "/D_%d.pth" % opt.epoch)))
else:
    G.apply(weights_init_norm)
    D.apply(weights_init_norm)
optimizer_G = torch.optim.Adam(G.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(D.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

print('Loading Datasets...')
dataset = ImageDataset("../../paras/cycleGAN/test/%s" % opt.dataset_name, img_shape, mode='train%s' % opt.defect_name, unaligned=True)
dataloader = DataLoader(
    dataset,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu
)
print(len(dataloader))
dataset = ImageDataset("../../datasets/%s" % opt.dataset_name, img_shape, mode='glue_test', unaligned=False)
val_loader = DataLoader(
    dataset,
    batch_size=8,
    # sampler=DistributedSampler(dataset),
    shuffle=True,
    num_workers=opt.n_cpu
)
print(len(val_loader))


def sample_imgs(batches_done):
    G.eval()
    img_samples = None
    imgs = next(iter(val_loader))
    for bg, mask in zip(imgs['B'], imgs['M']):
        img = bg.repeat(1, 1, 1, 1).type(Tensor)
        mask_in = mask.repeat(1, 1, 1, 1).type(Tensor)
        sample_z = Tensor(np.random.normal(0, 1, (1, opt.latent_dim)))
        _, fake_imgs, _ = G(img, sample_z, mask_in)
        fake_imgs = torch.cat([x for x in fake_imgs.data.cpu()], -2)
        # print(fake_imgs.shape)
        mask[mask == 0] = -1
        img_sample = torch.cat((bg, mask, fake_imgs), -2)
        # print(img_sample.max(), img_sample.min(), gt.min(), gt.max(), fake_imgs.min(), fake_imgs.max())
        img_sample = img_sample.view(1, *img_sample.shape)
        img_samples = img_sample if img_samples is None else torch.cat((img_samples, img_sample), -1)
    save_image(img_samples, img_path + '/' + '%s.png' % batches_done, nrow=8,
               normalize=True)
    G.train()


def compute_cosssim(x1, x2):
    x1 = x1.reshape(1, -1)
    x2 = x2.reshape(1, -1)
    cos = FF.cosine_similarity(x1, x2, dim=1)
    return cos


def loss_r_and_d():
    def_re = recon_img * mask_single
    def_img = fake_img * mask_single
    # retain background of source image and reconstruct source image
    re_bg = recon_img * (1 - mask_single)
    fake_bg = fake_img * (1 - mask_single)
    real_bg = bg * (1 - mask_single)
    recon_5 = L1(fake_5 * (1 - mask_single), real_bg) + L1(re_5 * (1 - mask_single), real_bg)

    loss_r = opt.lambda_r * (2 * L1(fake_bg, real_bg) + 2 * L1(re_bg, real_bg) + 10 * L1(def_re, gt * mask_single) + opt.lambda_5 * recon_5)
    loss_d = opt.lambda_diver * L1(def_re, def_img)  #
    return loss_r, loss_d


def grad_loss(img_gt, img2, img3, mask_):
    grad_mask = grad(mask_)
    grad_mask[grad_mask != 0] = 1
    grad_gt = grad(img_gt) * grad_mask
    grad2 = grad(img2) * grad_mask
    grad3 = grad(img3) * grad_mask
    return L1(grad_gt, grad2) + L1(grad_gt, grad3)


def compute_loss(x, gt):
    loss = sum([torch.mean((out - gt) ** 2) for out in x])
    return loss


def atten_loss():
    d_mask_n = 1 - mask
    zero = torch.ones_like(d_mask_n)
    loss_m = L1(d_mask_n * atten_recon, zero) + L1(d_mask_n * atten_fake, zero)# + L1(me_recon, e_mask) + L1(me_fake, e_mask
    return loss_m


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D.compute_loss(interpolates, mask, valid)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        # grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


fake = 0
valid = 1
print('Training...')
prev_time = time.time()
if __name__ == "__main__":
    j = 0
    for epoch in range(opt.epoch, opt.n_epochs):
        for i, batch in enumerate(dataloader):
            j += 1
            gt = batch['R'].type(Tensor)
            bg = batch['B'].type(Tensor)
            mask = batch['M'].type(Tensor)
            mask_single = torch.mean(mask, dim=1, keepdim=True)
            mask_single[mask_single != 1] = 0

            #  Training Encoder
            optimizer_G.zero_grad()

            z1 = Tensor(np.random.normal(0, 1, (gt.size(0), opt.latent_dim)))
            re_5, recon_img, atten_recon = G(bg, z1, mask)
            def_re = recon_img * mask_single

            z2 = Tensor(np.random.normal(0, 1, (gt.size(0), opt.latent_dim)))
            fake_5, fake_img, atten_fake = G(bg, z2, mask)
            def_img = fake_img * mask_single

            loss_m = atten_loss()

            loss_grad = grad_loss(gt, recon_img, fake_img, mask)
            loss_recon, loss_diver = loss_r_and_d()
            loss_adv = D.compute_loss(recon_img, mask, valid) + D.compute_loss(fake_img, mask, valid)

            loss_G = loss_adv + loss_recon - loss_diver + opt.lambda_grad * loss_grad + opt.lambda_m * loss_m #
            loss_G.backward()
            optimizer_G.step()

            # Training Discriminator

            optimizer_D.zero_grad()
            loss_D = 2 * D.compute_loss(gt, mask, valid) + D.compute_loss(fake_img.detach(), mask, fake) + \
                     D.compute_loss(recon_img.detach(), mask, fake) + opt.lambda_gp * compute_gradient_penalty(D,
                                                                                                               gt.detach(),
                                                                                                               fake_img.detach())
            loss_D.backward()
            optimizer_D.step()

            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            with torch.no_grad():
                if batches_done % opt.sample_interval == 0:
                    sample_imgs(batches_done)

        # Save model checkpoints
        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0 and epoch > 200:
            torch.save(G.state_dict(), model_path + "/G_%d.pth" % epoch)
            torch.save(D.state_dict(), model_path + "/D_%d.pth" % epoch)

        loss_item = "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, recon: %f, grad: %f, diver: %f, m: %f] ETA: %s\n" \
                    % (
                        epoch,
                        opt.n_epochs,
                        i,
                        len(dataloader),
                        loss_D.item(),
                        loss_adv.item(),
                        loss_recon.item(),
                        opt.lambda_grad * loss_grad.item(),
                        loss_diver.item(),
                        loss_m.item(),
                        time_left,
                    )
        f.write(loss_item)
        sys.stdout.write(loss_item)
f.close()
