import torch, os
osp = os.path
nn = torch.nn
F = nn.functional

from inrnet import inn

root = osp.expanduser("~/code/diffcoord/temp")

def translate_cyclegan(domain):
    D_A, D_B, G_A2B, G_B2A = load_pretrained_models(domain)

    return

def translate_D(D, image_shape):
    return inn.conversion.translate_discrete_model(nn.Sequential(
        D.main[:8], nn.Conv2d(256, 512, 3, padding=1),
        D.main[9:], nn.AdaptiveAvgPool2d(1), nn.Flatten()), image_shape)
def translate_G(G, image_shape):
    return inn.conversion.translate_discrete_model(nn.Sequential(
        G.main, nn.AdaptiveAvgPool2d(1), nn.Flatten()), image_shape)

def load_pretrained_models(domain):
    sd = torch.load(f"{root}/{domain}/netD_A.pth")
    D_A = Discriminator()
    D_A.load_state_dict(sd)
    sd = torch.load(f"{root}/{domain}/netD_B.pth")
    D_B = Discriminator()
    D_B.load_state_dict(sd)
    sd = torch.load(f"{root}/{domain}/netG_A2B.pth")
    G_A2B = Generator()
    G_A2B.load_state_dict(sd)
    sd = torch.load(f"{root}/{domain}/netG_B2A.pth")
    G_B2A = Generator()
    G_B2A.load_state_dict(sd)
    return D_A, D_B, G_A2B, G_B2A

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, stride=2, padding=1),
        )

    def forward(self, x):
        x = self.main(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = torch.flatten(x, 1)
        return x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Initial convolution block
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            # Downsampling
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),

            # Residual blocks
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),

            # Upsampling
            nn.Conv2d(256, 128, 3, padding=1),
            nn.Upsample(scale_factor=2),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.Upsample(scale_factor=2),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            # Output layer
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()

        self.res = nn.Sequential(nn.ReflectionPad2d(1),
                                 nn.Conv2d(in_channels, in_channels, 3),
                                 nn.InstanceNorm2d(in_channels),
                                 nn.ReLU(inplace=True),
                                 nn.ReflectionPad2d(1),
                                 nn.Conv2d(in_channels, in_channels, 3),
                                 nn.InstanceNorm2d(in_channels))

    def forward(self, x):
        return x + self.res(x)