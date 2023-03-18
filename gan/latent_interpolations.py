from utils import interpolate_latent_space
import torch
import os

# wgan-gp
wgan_gen_path = os.path.join('.', 'data_wgan_gp', 'generator.pt')

# lsgan
lsgan_gen_path = os.path.join('.', 'data_ls_gan', 'generator.pt')

# gan
gan_gen_path = os.path.join('.', 'data_gan', 'generator.pt')

gen = torch.jit.load(gan_gen_path)
interpolate_latent_space(gen, "./gan_interp.png")