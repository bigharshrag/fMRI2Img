from torch import nn

# Encoder Code
class Encoder(nn.Module):
    def __init__(self, ngpu, input_dim=64):
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        # todo figure out weight fillers
        self.s1 = nn.Sequential(
            nn.Conv2d(input_dim, 96, 11, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75),
            nn.Conv2d(96, 256, 5, padding=2, group=2),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75),
            nn.Conv2d(256, 384, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, padding=1, group=2),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, padding=1, group=2),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
        )

        
    def forward(self, input):
        s1 = self.s1(input)
        # TODO figure out if recon_loss needs to be in the model itself
        generated = s1
        return generated

# Generator Code
class Generator(nn.Module):
    def __init__(self, ngpu, input_dim=64):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.s1 = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Linear(4096, 4096),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Linear(4096, 4096)
        )

        # Need to figure out for these layers msra stuff, input sizes
        self.s2 = nn.Sequential(
            nn.ConvTranspose2d(4096, 256, 4, padding=1, stride=2),
            nn.LeakyReLU(negative_slope=0.3),
            nn.ConvTranspose2d(256, 512, 3, padding=1, stride=1),
            nn.LeakyReLU(negative_slope=0.3),
            nn.ConvTranspose2d(512, 256, 4, padding=1, stride=2),
            nn.LeakyReLU(negative_slope=0.3),
            nn.ConvTranspose2d(256, 256, 3, padding=1, stride=1),
            nn.LeakyReLU(negative_slope=0.3),
            nn.ConvTranspose2d(128, 128, 4, padding=1, stride=2),
            nn.LeakyReLU(negative_slope=0.3),
            nn.ConvTranspose2d(128, 128, 3, padding=1, stride=1),
            nn.LeakyReLU(negative_slope=0.3),
            nn.ConvTranspose2d(64, 64, 4, padding=1, stride=2),
            nn.LeakyReLU(negative_slope=0.3),
            nn.ConvTranspose2d(64, 32, 4, padding=1, stride=2),
            nn.LeakyReLU(negative_slope=0.3),
            nn.ConvTranspose2d(32, 3, 4, padding=1, stride=2),
            nn.LeakyReLU(negative_slope=0.3)
        )

    def forward(self, input):
        s1 = self.s1(input)
        s1 = s1.view(64, 456, 4, 4)
        s2 = self.s2(s1)

        # TODO figure out crop
        # TODO figure out if recon_loss needs to be in the model itself
        generated = s2
        return generated


class Discriminator(nn.Module):
    def __init__(self, ngpu, input_dim=64):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.s1 = nn.Sequential(
            nn.Conv2d(input_dim, 32, 7, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 128, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(11, stride=11)
        )

        self.s2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 2)
        )

        self.softmax = nn.Softmax()

    def forward(self, input):
        s1 = self.s1(input)
        s1 = s1.view(64, 256)
        s2 = self.s2(s1)

        output = self.softmax(s2)
        return output