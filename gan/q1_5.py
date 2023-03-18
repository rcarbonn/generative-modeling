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
    # interp.grad.zero_()
    # interp.retain_grad()
    # discrim_interp.backward(torch.ones_like(discrim_fake))
    # grad_interp = interp.grad
    # grad_interp = grad_interp.view(grad_interp.shape[0], -1)
    # loss3 = torch.square(grad_interp.norm(2,dim=1)-1)
    # interp.grad.zero_()

    gradp = torch.autograd.grad(
        outputs = discrim_interp,
        inputs = interp,
        grad_outputs = torch.autograd.Variable(torch.Tensor(discrim_real.shape[0],1).fill_(1.0), requires_grad=True).cuda(),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradp = gradp.view(gradp.size(0),-1)
    grad_pen = ((gradp.norm(2,dim=1)-1)**2).mean()
    # loss = torch.mean(discrim_fake) - torch.mean(discrim_real) + lamb * loss3
    loss = discrim_fake.mean() - discrim_real.mean() + lamb * grad_pen
    return loss


def compute_generator_loss(discrim_fake):
    """
    TODO 1.5.1: Implement WGAN-GP loss for generator.
    loss = - E[D(fake_data)]
    """
    loss = -torch.mean(discrim_fake)
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
