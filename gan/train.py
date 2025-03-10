from glob import glob
import os
import torch
from tqdm import tqdm
from utils import get_fid, interpolate_latent_space, save_plot
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from torchvision.datasets import VisionDataset


def build_transforms():
    # TODO 1.2: Add two transforms:
    # 1. Convert input image to tensor.
    # 2. Rescale input image to be between -1 and 1.
    # NOTE: don't do anything fancy for 2, hint: the input image is between 0 and 1.
    ds_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    return ds_transforms


def get_optimizers_and_schedulers(gen, disc):
    # TODO 1.2 Get optimizers and learning rate schedulers.
    # 1. Construct the optimizers for the discriminator and generator.
    # Both should use the Adam optimizer with learning rate of .0002 and Beta1 = 0, Beta2 = 0.9.
    # 2. Construct the learning rate schedulers for the generator and discriminator.
    # The learning rate for the discriminator should be decayed to 0 over 500K iterations.
    # The learning rate for the generator should be decayed to 0 over 100K iterations.
    optim_discriminator = torch.optim.Adam(disc.parameters(), lr=0.0002, betas=(0.0, 0.9))
    optim_generator = torch.optim.Adam(gen.parameters(), lr=0.0002, betas=(0.0, 0.9))

    # Try constant LR (doesnt work)
    # scheduler_discriminator = torch.optim.lr_scheduler.ConstantLR(optim_discriminator, factor=0.9, total_iters=500000)
    # scheduler_generator = torch.optim.lr_scheduler.ConstantLR(optim_generator, factor=0.9, total_iters=100000)

    # Try Step LR (doesnt work)
    # scheduler_discriminator = torch.optim.lr_scheduler.StepLR(optim_discriminator, step_size=1, gamma=0.9999)
    # scheduler_generator = torch.optim.lr_scheduler.StepLR(optim_generator, step_size=1, gamma=0.9995)

    lr_disc = lambda iter: 1 - (iter/500000)
    lr_gen = lambda iter: 1 - (iter/100000)
    scheduler_discriminator = torch.optim.lr_scheduler.LambdaLR(optim_discriminator, lr_lambda=lr_disc)
    scheduler_generator = torch.optim.lr_scheduler.LambdaLR(optim_generator, lr_lambda=lr_gen)

    
    return (
        optim_discriminator,
        scheduler_discriminator,
        optim_generator,
        scheduler_generator,
    )


class Dataset(VisionDataset):
    def __init__(self, root, transform=None):
        super(Dataset, self).__init__(root)
        self.file_names = glob(os.path.join(self.root, "*.jpg"), recursive=True)
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.file_names[index])
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.file_names)


def train_model(
    gen,
    disc,
    num_iterations,
    batch_size,
    lamb=10,
    prefix=None,
    gen_loss_fn=None,
    disc_loss_fn=None,
    log_period=10000,
    amp_enabled=True,
):
    torch.backends.cudnn.benchmark = True # speed up training
    ds_transforms = build_transforms()
    train_loader = torch.utils.data.DataLoader(
        Dataset(root="../datasets/CUB_200_2011_32", transform=ds_transforms),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=False,
    )

    (
        optim_discriminator,
        scheduler_discriminator,
        optim_generator,
        scheduler_generator,
    ) = get_optimizers_and_schedulers(gen, disc)

    scaler = torch.cuda.amp.GradScaler()

    iters = 0
    fids_list = []
    iters_list = []
    pbar = tqdm(total = num_iterations)
    while iters < num_iterations:
        for train_batch in train_loader:
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                train_batch = train_batch.cuda()
                ############################ UPDATE DISCRIMINATOR ######################################
                # TODO 1.2: compute generator, discriminator and interpolated outputs
                # 1. Compute generator output -> the number of samples must match the batch size.
                # 2. Compute discriminator output on the train batch.
                # 3. Compute the discriminator output on the generated data.
                gen_out = gen(train_batch.shape[0])
                gen_out_isolate = gen_out.detach()
                disc_gen = disc(gen_out_isolate)
                disc_train = disc(train_batch)
                eps = torch.rand(train_batch.shape[0],1,1,1).cuda()

                # TODO: 1.5 Compute the interpolated batch and run the discriminator on it.
                gen_out2 = gen(train_batch.shape[0])
                interp = eps*train_batch + (1-eps)*gen_out2
                discrim_interp = disc(interp)
                discriminator_loss = disc_loss_fn(disc_train, disc_gen, discrim_interp.requires_grad_(), interp, lamb)
                print("Disc Loss: ", discriminator_loss.item())


            optim_discriminator.zero_grad(set_to_none=True)
            # discriminator_loss.backward()
            # optim_discriminator.step()
            scaler.scale(discriminator_loss).backward()
            scaler.step(optim_discriminator)
            scheduler_discriminator.step()

            if iters % 5 == 0:
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    # TODO 1.2: compute generator and discriminator output on generated data.
                    # no need to recompute gen data
                    gen_out = gen(train_batch.shape[0])
                    disc_gen = disc(gen_out)
                    generator_loss = gen_loss_fn(disc_gen)
                    print("Gen Loss: ", generator_loss.item())
                optim_generator.zero_grad(set_to_none=True)
                # generator_loss.backward()
                # optim_generator.step()
                scaler.scale(generator_loss).backward()
                scaler.step(optim_generator)
                scheduler_generator.step()

            if iters % log_period == 0 and iters != 0:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=amp_enabled):
                        # TODO 1.2: Generate samples using the generator, make sure they lie in the range [0, 1].
                        gen_samples = gen(100)

                        # do an inverse norm, add 1 and divide by 2
                        inv_norm = transforms.Normalize([-1.0,-1.0, -1.0], [2.0, 2.0, 2.0])
                        generated_samples = inv_norm(gen_samples)

                        # gen_samples_min,_ = gen_samples.view(100, 3, -1).min(dim=2)
                        # generated_samples = gen_samples - gen_samples_min.unsqueeze(-1).unsqueeze(-1)
                        # generated_samples_max,_  = generated_samples.view(100, 3, -1).max(dim=2)
                        # generated_samples = generated_samples / generated_samples_max.unsqueeze(-1).unsqueeze(-1)
                        
                    save_image(
                        generated_samples.data.float(),
                        prefix + "samples_{}.png".format(iters),
                        nrow=10,
                    )
                    if int(os.environ.get('PYTORCH_JIT', 1)):
                        torch.jit.save(torch.jit.script(gen), prefix + "/generator.pt")
                        torch.jit.save(torch.jit.script(disc), prefix + "/discriminator.pt")
                    else:
                        torch.save(gen, prefix + "/generator.pt")
                        torch.save(disc, prefix + "/discriminator.pt")
                    fid = get_fid(
                        gen,
                        dataset_name="cub",
                        dataset_resolution=32,
                        z_dimension=128,
                        batch_size=256,
                        num_gen=10_000,
                    )
                    print(f"Iteration {iters} FID: {fid}")
                    fids_list.append(fid)
                    iters_list.append(iters)

                    save_plot(
                        iters_list,
                        fids_list,
                        xlabel="Iterations",
                        ylabel="FID",
                        title="FID vs Iterations",
                        filename=prefix + "fid_vs_iterations",
                    )
                    interpolate_latent_space(
                        gen, prefix + "interpolations_{}.png".format(iters)
                    )
            scaler.update()
            iters += 1
            pbar.update(1)
    fid = get_fid(
        gen,
        dataset_name="cub",
        dataset_resolution=32,
        z_dimension=128,
        batch_size=256,
        num_gen=50_000,
    )
    print(f"Final FID (Full 50K): {fid}")
