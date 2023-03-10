import torch
import torch.nn as nn
import torch.nn.functional as F


class UpSampleConv2D(torch.jit.ScriptModule):

    def __init__(
        self,
        input_channels,
        kernel_size=(3,3),
        n_filters=128,
        upscale_factor=2,
        padding=(0,0)
    ):
        super(UpSampleConv2D, self).__init__()
        self.conv = nn.Conv2d(in_channels=input_channels, out_channels=n_filters, kernel_size=kernel_size, stride=(1,1), padding=padding)
        self.upscale_factor = upscale_factor


    @torch.jit.script_method
    def forward(self, x):
        # TODO 1.1: Implement nearest neighbor upsampling
        # 1. Repeat x channel wise upscale_factor^2 times
        # 2. Use pixel shuffle (https://pytorch.org/docs/master/generated/torch.nn.PixelShuffle.html#torch.nn.PixelShuffle)
        # to form a (batch x channel x height*upscale_factor x width*upscale_factor) output
        # 3. Apply convolution and return output
        x = x.repeat(1, self.upscale_factor*self.upscale_factor, 1, 1)
        x = F.pixel_shuffle(x, self.upscale_factor)
        x = self.conv(x)
        return x


class DownSampleConv2D(torch.jit.ScriptModule):

    def __init__(
        self,
        input_channels,
        kernel_size=(3,3),
        n_filters=128, 
        downscale_ratio=2, 
        padding=(0,0)
    ):
        super(DownSampleConv2D, self).__init__()
        self.conv = nn.Conv2d(in_channels=input_channels, out_channels=n_filters, kernel_size=kernel_size, padding=padding)
        self.downscale_ratio = downscale_ratio


    @torch.jit.script_method
    def forward(self, x):
        # TODO 1.1: Implement spatial mean pooling
        # 1. Use pixel unshuffle (https://pytorch.org/docs/master/generated/torch.nn.PixelUnshuffle.html#torch.nn.PixelUnshuffle)
        # to form a (batch x channel * downscale_factor^2 x height x width) output
        # 2. Then split channel wise into (downscale_factor^2xbatch x channel x height x width) images
        # 3. Average across dimension 0, apply convolution and return output
        x = F.pixel_unshuffle(x, self.downscale_ratio)
        assert x.shape[1]%self.downscale_ratio==0
        x = x.split(x.shape[1]//(self.downscale_ratio**2), 1)
        x = torch.stack(x)
        x = x.mean(0)
        x = self.conv(x)
        return x


class ResBlockUp(torch.jit.ScriptModule):

    def __init__(self, input_channels, kernel_size=(3,3), n_filters=128):
        super(ResBlockUp, self).__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm2d(num_features=input_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=input_channels, out_channels=n_filters, kernel_size=kernel_size, stride=(1,1), padding=(1,1), bias=False),
            nn.BatchNorm2d(num_features=n_filters, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            UpSampleConv2D(input_channels=n_filters, kernel_size=(3,3), n_filters=n_filters, upscale_factor=2, padding=(1,1))
        )
        self.upsample_residual = UpSampleConv2D(input_channels=input_channels, kernel_size=(1,1), n_filters=n_filters)

    @torch.jit.script_method
    def forward(self, x):
        # TODO 1.1: Forward through the layers and implement a residual connection.
        # Make sure to upsample the residual before adding it to the layer output.
        x_layers = self.layers(x)
        x_residual = self.upsample_residual(x)
        return x_layers + x_residual


class ResBlockDown(torch.jit.ScriptModule):

    def __init__(self, input_channels, kernel_size=(3,3), n_filters=128):
        super(ResBlockDown, self).__init__()
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=input_channels, out_channels=n_filters, kernel_size=kernel_size, stride=(1,1), padding=(1,1)),
            nn.ReLU(),
            DownSampleConv2D(input_channels=n_filters, kernel_size=kernel_size, n_filters=n_filters, downscale_ratio=2, padding=(1,1))
        )
        self.downsample_residual = DownSampleConv2D(input_channels=input_channels, kernel_size=(1,1), n_filters=n_filters)

    @torch.jit.script_method
    def forward(self, x):
        # TODO 1.1: Forward through self.layers and implement a residual connection.
        # Make sure to downsample the residual before adding it to the layer output.
        x_layers = self.layers(x)
        x_residual = self.downsample_residual(x)
        return x_layers + x_residual


class ResBlock(torch.jit.ScriptModule):

    def __init__(self, input_channels, kernel_size=(3,3), n_filters=128):
        super(ResBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=input_channels, out_channels=n_filters, kernel_size=kernel_size, padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=kernel_size, padding=(1,1))
        )

    @torch.jit.script_method
    def forward(self, x):
        # TODO 1.1: Forward the conv layers. Don't forget the residual connection!
        residual = x
        x_layers = self.layers(x)
        return x_layers + residual


class Generator(torch.jit.ScriptModule):

    def __init__(self, starting_image_size=4):
        super(Generator, self).__init__()

        self.starting_image_size = starting_image_size
        self.dense = nn.Linear(in_features=128, out_features=2048, bias=True)
        self.layers = nn.Sequential(
            ResBlockUp(input_channels=128, kernel_size=(3,3), n_filters=128),
            ResBlockUp(input_channels=128, kernel_size=(3,3), n_filters=128),
            ResBlockUp(input_channels=128, kernel_size=(3,3), n_filters=128),
            nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=3, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.Tanh()
        )

    @torch.jit.script_method
    def forward_given_samples(self, z):
        # TODO 1.1: forward the generator assuming a set of samples z have been passed in.
        # Don't forget to re-shape the output of the dense layer into an image with the appropriate size!
        x = self.dense(z)
        # TODO: check the order of reshape
        x = x.view(x.shape[0], -1, self.starting_image_size, self.starting_image_size)
        x = self.layers(x)
        return x

    @torch.jit.script_method
    def forward(self, n_samples: int = 1024):
        # TODO 1.1: Generate n_samples latents and forward through the network.
        samples = torch.randn(n_samples, 128)
        return self.forward_given_samples(samples)


class Discriminator(torch.jit.ScriptModule):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.layers = nn.Sequential(
            ResBlockDown(input_channels=3, kernel_size=(3,3), n_filters=128),
            ResBlockDown(input_channels=128, kernel_size=(3,3), n_filters=128),
            ResBlock(input_channels=128, kernel_size=(3,3), n_filters=128),
            ResBlock(input_channels=128, kernel_size=(3,3), n_filters=128),
            nn.ReLU()
        )
        self.dense = nn.Linear(in_features=128, out_features=1, bias=True)

    @torch.jit.script_method
    def forward(self, x):
        # TODO 1.1: Forward the discriminator assuming a batch of images have been passed in.
        # Make sure to sum across the image dimensions after passing x through self.layers.
        x = self.layers(x)
        x = x.flatten(start_dim=2)
        x = x.sum(dim=2)
        return self.dense(x)


if __name__ == "__main__":
    gen = Generator()
    disc = Discriminator()
    print("#"*100)
    print(gen)
    print("#"*100)
    print(disc)