import os

import torch
from utils import get_args

from networks import Discriminator, Generator
from train import train_model


def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp, interp, lamb
):
    """
    TODO 1.5.1: Implement WGAN-GP loss for discriminator.
    loss = E[D(fake_data)] - E[D(real_data)] + lambda * E[(|| grad wrt interpolated_data (D(interpolated_data))|| - 1)^2]
    """
    sig_fake = torch.sigmoid(discrim_fake) + 1e-6
    sig_real = torch.sigmoid(discrim_real) + 1e-6
    loss1 = torch.log(sig_fake)
    loss2 = torch.log(sig_real)
    interp.grad.zero_()
    discrim_interp.backward(torch.ones_like(interp))
    grad_interp = interp.grad
    loss3 = torch.square(grad_interp.norm() - 1)
    loss = loss1 - loss2 + lamb*loss3
    return loss.mean()


def compute_generator_loss(discrim_fake):
    """
    TODO 1.5.1: Implement WGAN-GP loss for generator.
    loss = - E[D(fake_data)]
    """
    sig_fake = torch.sigmoid(discrim_fake) + 1e-6
    loss = - torch.log(sig_fake).mean()
    return loss


if __name__ == "__main__":
    args = get_args()
    gen = Generator().cuda()
    disc = Discriminator().cuda()
    prefix = "data_wgan_gp/"
    os.makedirs(prefix, exist_ok=True)

    # TODO 1.5.2: Run this line of code.
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
