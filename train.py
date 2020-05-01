from model import Encoder, Generator, Discriminator
import time
import torch
from torch import nn
from torch import optim
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from constants import BaseOptions
from data.fMRIimgdataset import fMRIImgDataset
import time

args = BaseOptions().parse()
batch_size = args.batch_size
ngpu = args.ngpu

dataset = fMRIImgDataset(args, subject='sub-01')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=args.workers)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
encoder = Encoder(ngpu).to(device)

# Create the generator
tfmri, _ = dataset.__getitem__(0)
generator = Generator(ngpu, tfmri.shape[0]).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    generator = nn.DataParallel(generator, list(range(ngpu)))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
generator.apply(weights_init)

# Print the model
# print(generator)

# Create the Discriminator
discriminator = Discriminator(ngpu, 3).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    discriminator = nn.DataParallel(discriminator, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
discriminator.apply(weights_init)

# Print the model
# print(discriminator)

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(args.ngf, args.nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1
generated_label = 0

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(discriminator.parameters(), lr=args.d_lr, betas=(args.beta1, 0.999))
optimizerG = optim.Adam(generator.parameters(), lr=args.g_lr, betas=(args.beta1, 0.999))

# Training Loop
train_discr = True
train_gen = True

# bce_loss = nn.BCEWithLogitsLoss()
bce_loss = nn.BCELoss()
mse_loss = nn.MSELoss(reduction='mean')

writer = SummaryWriter('runs/' + args.exp_name)

total_steps = 0
# Training main loop
start = time.time()
for it in range(args.num_epochs):
    for i, data in enumerate(dataloader):
        data_fmri, data_image = data[0].to(device), data[1].to(device)
        curr_sz = data_image.size(0)

        total_steps += curr_sz
        # Feed the data (images) to the encoder and run it        
        # encoded_real_image = encoder(data_image)

        # Feed the data to the generator and run it
        generator_output = generator(data_fmri)
        # print(generator_output.shape)
        # generator_output_crop = generator_output[:, :, 16:240, 16:240].detach()
        # generator_image_loss = mse_loss(generator_output_crop, data_image)

        # Feed the generated image into the encoder
        # encoded_generated_image = encoder(generator_output_crop)
        # generator_encoded_loss = mse_loss(encoded_generated_image, encoded_real_image)

        discriminator.zero_grad()
        # Run the discriminator on real image
        real_classification = discriminator(data_image).view(-1)
        real_labels = torch.full((curr_sz,), real_label, device=device)
        disc_loss_real = bce_loss(real_classification, real_labels)
        if train_discr:
            disc_loss_real.backward()

        # Run the discriminator on generated data
        generated_classification = discriminator(generator_output.detach()).view(-1)
        generated_labels = torch.full((curr_sz,), generated_label, device=device)
        disc_loss_gen = bce_loss(generated_classification, generated_labels)
        if train_discr:
            disc_loss_gen.backward()

            optimizerD.step()

        generator.zero_grad()
        # Run the discriminator on generated data with opposite labels, to get the gradient for the generator
        generated_classification_for_generator = discriminator(generator_output).view(-1)
        real_labels_for_generator_loss = torch.full((curr_sz,), real_label, device=device)
        generator_disc_loss = bce_loss(generated_classification_for_generator, real_labels_for_generator_loss)

        if train_gen:
            # gen_loss = args.lambda_1 * generator_image_loss + args.lambda_2 * generator_encoded_loss + args.lambda_3 * generator_disc_loss
            gen_loss = generator_disc_loss
            gen_loss.backward()
    
            optimizerG.step()

        # Save snapshot
        if it % args.snapshot_interval == 0:
            # torch.save(generator.state_dict(), 'generator_{0}.pth'.format(it))
            # torch.save(discriminator.state_dict(), 'discriminator_{0}.pth'.format(it))
            torch.save(generator.state_dict(), 'generator.pth')
            torch.save(discriminator.state_dict(), 'discriminator.pth')

        # Switch optimizing discriminator and generator, so that neither of them overfits too much
        
        # discr_loss_ratio = (disc_loss_real.item() + disc_loss_gen.item()) / generator_disc_loss.item()
        
        # if discr_loss_ratio < 1e-1 and train_discr:
        #     train_discr = False
        #     train_gen = True
        #     print('<<< real_loss=%e, fake_loss=%e, fake_loss_for_generator=%e, train_discr=%d, train_gen=%d >>>' % (disc_loss_real, disc_loss_gen, generator_disc_loss.item(), train_discr, train_gen))
        # if discr_loss_ratio > 5e-1 and not train_discr:
        #     train_discr = True
        #     train_gen = True
        #     print(' <<< real_loss=%e, fake_loss=%e, fake_loss_for_generator=%e, train_discr=%d, train_gen=%d >>>' % (disc_loss_real, disc_loss_gen, generator_disc_loss.item(), train_discr, train_gen))
        # if discr_loss_ratio > 1e1 and train_gen:
        #     train_gen = False
        #     train_discr = True
        #     print('<<< real_loss=%e, fake_loss=%e, fake_loss_for_generator=%e, train_discr=%d, train_gen=%d >>>' % (disc_loss_real, disc_loss_gen, generator_disc_loss.item(), train_discr, train_gen))

        # Display info
        print('========================================')
        print('')
        print('[%s] Iteration %d: %f seconds' % (time.strftime('%c'), it, time.time()-start))
        # print('  generator image loss: %e' % (generator_image_loss))
        # print('  generator feat loss: %e' % (generator_encoded_loss))
        print('  discr real loss: %e' % (disc_loss_real.item()))
        print('  discr generated loss: %e' % (disc_loss_gen.item()))
        print('  generator loss from discriminator: %e' % (generator_disc_loss.item()))
        # writer.add_scalar('data/generator_image_loss', args.lambda_1 * generator_image_loss, total_steps)
        # writer.add_scalar('data/generator_feat_loss', args.lambda_2 * generator_encoded_loss, total_steps)
        writer.add_scalar('data/discr_real_loss', disc_loss_real.item(), total_steps)
        writer.add_scalar('data/discr_generated_loss', disc_loss_gen.item(), total_steps)
        writer.add_scalar('data/generator_loss', args.lambda_3 * generator_disc_loss.item(), total_steps)
        # writer.add_scalar('data/discr_loss_ratio', discr_loss_ratio, total_steps)
        writer.add_scalars('loss/', {
            # 'generator': args.lambda_1 * generator_image_loss + args.lambda_2 * generator_encoded_loss + args.lambda_3 * generator_disc_loss.item(),
            'generator': generator_disc_loss.item(),
            'discriminator': disc_loss_real + disc_loss_gen.item(),
        }, total_steps)
        start = time.time()


# TODO rebuild this plot
# plt.figure(figsize=(10,5))
# plt.title("Generator and Discriminator Loss During Training")
# plt.plot(G_losses,label="G")
# plt.plot(D_losses,label="D")
# plt.xlabel("iterations")
# plt.ylabel("Loss")
# plt.legend()
# plt.show()