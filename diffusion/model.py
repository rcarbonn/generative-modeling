import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import (
    cosine_beta_schedule,
    default,
    extract,
    unnormalize_to_zero_to_one,
)
from einops import rearrange, reduce

class DiffusionModel(nn.Module):
    def __init__(
        self,
        model,
        timesteps = 1000,
        sampling_timesteps = None,
        ddim_sampling_eta = 1.,
    ):
        super(DiffusionModel, self).__init__()

        self.model = model
        self.channels = self.model.channels
        self.device = torch.cuda.current_device()

        self.betas = cosine_beta_schedule(timesteps).to(self.device)
        self.num_timesteps = self.betas.shape[0]

        alphas = 1. - self.betas
        # TODO 3.1: compute the cumulative products for current and previous timesteps
        self.alphas_cumprod = torch.cumprod(alphas, 0)
        self.alphas_cumprod_prev =  self.alphas_cumprod / alphas

        # TODO 3.1: pre-compute values needed for forward process
        # This is the coefficient of x_t when predicting x_0
        self.x_0_pred_coef_1 = 1.0 / torch.sqrt(self.alphas_cumprod)
        # This is the coefficient of pred_noise when predicting x_0
        self.x_0_pred_coef_2 = -1.0 * (torch.sqrt(1-self.alphas_cumprod)/torch.sqrt(self.alphas_cumprod))

        # TODO 3.1: compute the coefficients for the mean
        # This is coefficient of x_0 in the DDPM section
        self.posterior_mean_coef1 = torch.sqrt(self.alphas_cumprod_prev) * self.betas / (1-self.alphas_cumprod)
        # This is coefficient of x_t in the DDPM section
        self.posterior_mean_coef2 = torch.sqrt(alphas) * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)

        # TODO 3.1: compute posterior variance
        # calculations for posterior q(x_{t-1} | x_t, x_0) in DDPM
        self.posterior_variance = (1 - self.alphas_cumprod_prev) * self.betas / (1 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(
            self.posterior_variance.clamp(min =1e-20))

        # sampling related parameters
        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

    def get_posterior_parameters(self, x_0, x_t, t):
        # TODO 3.1: Compute the posterior mean and variance for x_{t-1}
        # using the coefficients, x_t, and x_0
        # hint: can use extract function from utils.py
        posterior_mean = extract(self.posterior_mean_coef2, t, x_t.shape) * x_t + extract(self.posterior_mean_coef1, t, x_0.shape) * x_0
        posterior_variance = torch.sqrt(extract(self.posterior_variance, t, x_t.shape))
        posterior_log_variance_clipped = torch.sqrt(extract(self.posterior_log_variance_clipped, t, x_t.shape))
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x_t, t):
        # TODO 3.1: given a noised image x_t, predict x_0 and the additive noise
        # to predict the additive noise, use the denoising model.
        # Hint: You can use extract function from utils.py.
        # clamp x_0 to [-1, 1]
        pred_noise = self.model(x_t, t)
        x_0 = extract(self.x_0_pred_coef_1, t, x_t.shape) * x_t + extract(self.x_0_pred_coef_2, t, x_t.shape) * pred_noise
        x_0 = torch.clamp(x_0, -1.0, 1.0)
        return (pred_noise, x_0)

    @torch.no_grad()
    def predict_denoised_at_prev_timestep(self, x, t: int):
        # TODO 3.1: given x at timestep t, predict the denoised image at x_{t-1}.
        # also return the predicted starting image.
        # Hint: To do this, you will need a predicted x_0. Which function can do this for you?
        pred_noise, x_0 = self.model_predictions(x, t)
        posterior_mean, posterior_variance, _ = self.get_posterior_parameters(x_0, x, t)
        pred_img = posterior_mean + posterior_variance * torch.randn_like(x)
        return pred_img, x_0

    @torch.no_grad()
    def sample_ddpm(self, shape, z):
        img = z
        for t in tqdm(range(self.num_timesteps-1, 0, -1)):
            batched_times = torch.full((img.shape[0],), t, device=self.device, dtype=torch.long)
            img, _ = self.predict_denoised_at_prev_timestep(img, batched_times)
        img = unnormalize_to_zero_to_one(img)
        return img

    def sample_times(self, total_timesteps, sampling_timesteps):
        # TODO 3.2: Generate a list of times to sample from.
        step_size = total_timesteps // sampling_timesteps
        # sampled_times = list(range(total_timesteps, 0, -1*step_size+1))
        sampled_times = list(range(0, total_timesteps, step_size))[::-1]
        return sampled_times

    def get_time_pairs(self, times):
        # TODO 3.2: Generate a list of adjacent time pairs to sample from.
        return list(zip(times[:-1], times[1:]))

    def ddim_step(self, batch, device, tau_i, tau_isub1, img, model_predictions, alphas_cumprod, eta):
        # TODO 3.2: Compute the output image for a single step of the DDIM sampling process.

        # predict x_0 and the additive noise for tau_i

        # extract \alpha_{\tau_{i - 1}} and \alpha_{\tau_{i}}

        # compute \sigma_{\tau_{i}}

        # compute the coefficient of \epsilon_{\tau_{i}}

        # sample from q(x_{\tau_{i - 1}} | x_{\tau_t}, x_0)
        # HINT: use the reparameterization trick
        t = torch.full((batch,), tau_i, device=device, dtype=torch.long)
        t1 = torch.full((batch,), tau_isub1, device=device, dtype=torch.long)
        pred_noise, x_0 = model_predictions(img, t)
        alpha_tau_i1 = extract(alphas_cumprod, t1, img.shape)
        alpha_tau_i = extract(alphas_cumprod, t, img.shape)
        sigma_tau_i = torch.sqrt(eta * (1 - alpha_tau_i1)/(1 - alpha_tau_i) * self.betas[tau_isub1])
        coeff_eps_tau_i = torch.sqrt(1-alpha_tau_i1 - torch.square(sigma_tau_i))

        mean_tau_i = torch.sqrt(alpha_tau_i1) * x_0 + coeff_eps_tau_i * pred_noise
        img = mean_tau_i + sigma_tau_i * torch.randn_like(img)

        return img, x_0

    def sample_ddim(self, shape, z):
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

        times = self.sample_times(total_timesteps, sampling_timesteps)
        time_pairs = self.get_time_pairs(times)

        img = z
        for tau_i, tau_isub1 in tqdm(time_pairs, desc='sampling loop time step'):
            img, _ = self.ddim_step(batch, device, tau_i, tau_isub1, img, self.model_predictions, self.alphas_cumprod, eta)

        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample(self, shape):
        sample_fn = self.sample_ddpm if not self.is_ddim_sampling else self.sample_ddim
        z = torch.randn(shape, device = self.betas.device)
        return sample_fn(shape, z)

    @torch.no_grad()
    def sample_given_z(self, z, shape):
        #TODO 3.3: fill out based on the sample function above
        sample_fn = self.sample_ddpm if not self.is_ddim_sampling else self.sample_ddim
        return sample_fn(shape, z)
