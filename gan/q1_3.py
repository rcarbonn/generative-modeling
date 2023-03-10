import argparse
import os
from utils import get_args

import torch

from networks import Discriminator, Generator
import torch.nn.functional as F
from train import train_model


def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp, interp, lamb
):
    """
    TODO 1.3.1: Implement GAN loss for discriminator.
    Do not use discrim_interp, interp, lamb. They are placeholders for Q1.5.
    """
    criterion = torch.nn.BCEWithLogitsLoss()
    # batch_sz_real = discrim_real.shape[0]
    # batch_sz_fake = discrim_fake.shape[0]
    # sig_real = F.sigmoid(discrim_real)
    # sig_fake = 1 - F.sigmoid(discrim_fake)
    # loss_real = -torch.log(sig_real).mean()
    # loss_fake = -torch.log(sig_fake).mean()
    loss_real = criterion(discrim_real, torch.ones_like(discrim_real))
    loss_fake = criterion(discrim_fake, torch.zeros_like(discrim_fake))
    loss = loss_real + loss_fake

    return loss


def compute_generator_loss(discrim_fake):
    """
    TODO 1.3.1: Implement GAN loss for generator.
    """
    # batch_sz_fake = discrim_fake
    # sig_fake = 1-F.sigmoid(discrim_fake)
    criterion = torch.nn.BCEWithLogitsLoss()
    loss = criterion(discrim_fake, torch.ones_like(discrim_fake))
    # loss = (-1*torch.log(sig_fake)/batch_sz_fake).sum()
    # loss = -1*torch.log(sig_fake).mean()

    return loss


if __name__ == "__main__":
    args = get_args()
    gen = Generator().cuda()
    disc = Discriminator().cuda()
    prefix = "data_gan/"
    os.makedirs(prefix, exist_ok=True)

    # TODO 1.3.2: Run this line of code.
    train_model(
        gen,
        disc,
        num_iterations=int(3e4),
        batch_size=256,
        prefix=prefix,
        gen_loss_fn=compute_generator_loss,
        disc_loss_fn=compute_discriminator_loss,
        log_period=1000,
        amp_enabled=not args.disable_amp,
    )
