import argparse

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--exp_name', type=str, default='exp1', help='Name of the experiment')
        self.parser.add_argument('--dataroot', type=str, default='data', help='Root directory for dataset')
        self.parser.add_argument('--workers', default=8, type=int, help='Number of workers for dataloader')
        self.parser.add_argument('--batch_size', default=16, type=int, help='Batch size during training')
        self.parser.add_argument('--ngpu', default=1, type=int, help='Size of feature maps in discriminator')

        self.parser.add_argument('--image_size', default=64, type=int, help='Spatial size of training images. All images will be resized to this size using a transformer.')
        self.parser.add_argument('--nc', default=3, type=int, help='Number of channels in the training images. For color images this is 3')
        self.parser.add_argument('--nz', default=100, type=int, help='Size of z latent vector (i.e. size of generator input')
        self.parser.add_argument('--ngf', default=64, type=int, help='Size of feature maps in generator')
        self.parser.add_argument('--ndf', default=64, type=int, help='Size of feature maps in discriminator')

        self.parser.add_argument('--num_epochs', default=10, type=int, help='Number of training epochs')
        self.parser.add_argument('--g_lr', default=0.00005, type=float, help='Learning rate for optimizers')
        self.parser.add_argument('--d_lr', default=0.00005, type=float, help='Learning rate for optimizers')
        self.parser.add_argument('--beta1', default=0.9, type=float, help='Beta1 hyperparam for Adam optimizers')
        
        self.parser.add_argument('--lambda_1', default=0, type=float, help='Beta1 hyperparam for Adam optimizers')
        self.parser.add_argument('--lambda_2', default=0, type=float, help='Beta1 hyperparam for Adam optimizers')
        self.parser.add_argument('--lambda_3', default=1, type=float, help='Beta1 hyperparam for Adam optimizers')
            
        self.parser.add_argument('--snapshot_interval', default=5)

    def parse(self):
        self.opt = self.parser.parse_args()
        return self.opt