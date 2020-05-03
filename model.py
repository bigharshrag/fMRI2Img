from torch import nn
import torchvision.models as models

# Encoder Code
class Encoder(nn.Module):
    def __init__(self, ngpu):
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        original_alexnet = models.alexnet(pretrained=True)
        self.features = nn.Sequential(*list(original_alexnet.children())[:-1])
        
    def forward(self, input):
        return self.features(input)

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
            nn.ConvTranspose2d(256, 256, 4, padding=1, stride=2, bias=False),
            # nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.3, inplace=True),
            
            nn.ConvTranspose2d(256, 512, 3, padding=1, stride=1, bias=False),
            # nn.ConvTranspose2d(256, 512, 4, padding=1, stride=2, bias=False),
            # nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.3, inplace=True),

            nn.ConvTranspose2d(512, 256, 4, padding=1, stride=2, bias=False),
            # nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.3, inplace=True),
            
            nn.ConvTranspose2d(256, 256, 3, padding=1, stride=1, bias=False),
            # nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.3, inplace=True),
            
            nn.ConvTranspose2d(256, 128, 4, padding=1, stride=2, bias=False),
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.3, inplace=True),
            
            nn.ConvTranspose2d(128, 128, 3, padding=1, stride=1, bias=False),
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.3, inplace=True),
            
            nn.ConvTranspose2d(128, 64, 4, padding=1, stride=2, bias=False),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.3, inplace=True),
            
            nn.ConvTranspose2d(64, 32, 4, padding=1, stride=2, bias=False),
            # nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.3, inplace=True),
            
            nn.ConvTranspose2d(32, 3, 4, padding=1, stride=2, bias=False),
            # nn.LeakyReLU(negative_slope=0.3)
            nn.Tanh()
        )

    def forward(self, input):
        s1 = self.s1(input)
        s1 = s1.view(-1, 256, 4, 4)
        s2 = self.s2(s1)
        # s2 = s2[:, :, 16:240, 16:240]
        generated = s2
        return generated


class DiscriminatorOld(nn.Module):
    def __init__(self, ngpu, input_dim=64):
        super(DiscriminatorOld, self).__init__()
        self.ngpu = ngpu
        self.s1 = nn.Sequential(
            nn.Conv2d(input_dim, 32, 7, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2),
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
            nn.Linear(256, 1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        s1 = self.s1(input)
        s1 = s1.view(-1, 256)
        s2 = self.s2(s1)

        output = self.sigmoid(s2)
        return output

class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x

class Discriminator(nn.Module):
    def __init__(self, ngpu, nc=3, ndf=256):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.s1 = nn.Sequential(
            nn.Conv2d(nc, ndf, 7, stride=4, padding=2, bias=False),
            # PrintLayer(),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 7, stride=4, padding=2, bias=False),
            # PrintLayer(),
            # nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=False),
            # PrintLayer(),
            # nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, stride=2, padding=1, bias=False),
            # PrintLayer(),
            # nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, 4, stride=1, padding=0, bias=False),
            # PrintLayer(),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.s1(input)

class fMRIClassifier(nn.Module):
    def __init__(self, ngpu, input_dim=64, output_dim=2):
        super(fMRIClassifier, self).__init__()
        self.s1 = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(p=0.6),
            # nn.Linear(8192, 4096),
            # nn.BatchNorm1d(4096),
            # nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.6),
            # nn.Linear(1024, 512),
            # nn.BatchNorm1d(512),
            # nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(1024, output_dim)
        )

        # self.s2 = nn.Sequential(
        #     nn.Conv1d(1, 4, 4, stride=2),
        #     nn.ReLU(),
        #     nn.Conv1d(4, 2, 2, stride=2),
        #     nn.ReLU(),
        #     nn.Conv1d(2, 2, 2, stride=1),
        #     nn.ReLU(),
        #     nn.Conv1d(2, 1, 1, stride=1),
        #     nn.ReLU()
        # )

        # self.ff = nn.Linear(1022, output_dim)

        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, input):
        output =  self.s1(input)
        # s1 = s1.unsqueeze(1)
        # s2 = self.s2(s1)
        # s2 = s2.squeeze()
        # output = self.ff(s2)

        return output # self.softmax(output)

class convClassifier(nn.Module):
    def __init__(self, ngpu, img_shape, output_dim=2):
        super(convClassifier, self).__init__()
        self.s1 = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.ff = nn.Linear(4 * 64 * 64, output_dim)
        
    def forward(self, input):
        conv =  self.s1(input)
        conv = conv.view(conv.shape[0], -1)
        output = self.ff(conv)

        return output
