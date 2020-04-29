from model import Encoder, Generator, Discriminator
import torch
from torch import nn
from torch import optim
import torchvision.utils as vutils
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
generator = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    geenrator = nn.DataParallel(geenrator, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
# geenrator.apply(weights_init)

# Print the model
print(geenrator)

# Create the Discriminator
discriminator = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    discriminator = nn.DataParallel(discriminator, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
# discriminator.apply(weights_init)

# Print the model
print(discriminator)

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(args.ngf, args.nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1
generated_label = 0

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
optimizerG = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

# Training Loop
train_discr = True
train_gen = True

bce_loss = nn.BCELoss()
mse_loss = nn.MSELoss(reduction='sum')

# Training main loop
start = time.time()
for it in range(args.num_epochs):
    for data_fmri, data_image in dataloader:
        # Feed the data (images) to the encoder and run it        
        encoded_real_image = encoder.forward(data_image)

        # Feed the data to the generator and run it
        generator_output = generator.forward(data_fmri)
        generator_image_loss = mse_loss(generator_output, data_image)

        # Feed the generated image into the encoder
        encoded_generated_image = encoder.forward(generator_output)
        generator_encoded_loss = mse_loss(encoded_generated_image, encoded_real_image)

        discriminator.zero_grad()
        # Run the discriminator on real image
        real_classification = discriminator.forward(data_image)
        real_labels = torch.full((batch_size,), real_label, device=device)
        disc_loss_real = bce_loss(real_classification, real_labels)
        if train_discr:
            disc_loss_real.backward()

        # Run the discriminator on generated data
        generated_classification = discriminator.forward(generator_output.detach())
        generated_labels = torch.full((batch_size,), generated_label, device=device)
        disc_loss_gen = bce_loss(generated_classification, real_labels)
        if train_discr:
            disc_loss_gen.backward()

        optimizerD.step()

        generator.zero_grad()
        # Run the discriminator on generated data with opposite labels, to get the gradient for the generator
        generated_classification_for_generator = discriminator.forward(generator_output)
        real_labels_for_generator_loss = torch.full((batch_size,), real_label, device=device)
        generator_disc_loss = bce_loss(generated_classification_for_generator, real_labels_for_generator_loss)

        if train_gen:
            gen_loss = args.lambda_1 * generator_image_loss + args.lambda_2 * generator_encoded_loss + args.lambda_3 * generator_disc_loss
            gen_loss.backward()
    
        optimizerG.step()

        # Display info
        # if it % display_every == 0:
        #     print('========================================')
        #     print('')
        #     print('[%s] Iteration %d: %f seconds' % (time.strftime('%c'), it, time.time()-start))
        #     print('  recon loss: %e * %e = %f' % (recon_loss, recon_loss_weight, recon_loss*recon_loss_weight))
        #     print('  feat loss: %e * %e = %f' % (feat_recon_loss, feat_loss_weight, feat_recon_loss*feat_loss_weight))
        #     print('  discr real loss: %e * %e = %f' % (discr_real_loss, discr_loss_weight, discr_real_loss*discr_loss_weight))
        #     print('  discr fake loss: %e * %e = %f' % (discr_fake_loss, discr_loss_weight, discr_fake_loss*discr_loss_weight))
        #     print('  discr fake loss for generator: %e * %e = %f' % (discr_fake_for_generator_loss, discr_loss_weight, discr_fake_for_generator_loss*discr_loss_weight))
        #     start = time.time()

        # Save snapshot
        # if it % snapshot_every == 0:
        #     # TODO save models
        #     pass

        # Switch optimizing discriminator and generator, so that neither of them overfits too much
        
        discr_loss_ratio = (disc_loss_real + disc_loss_gen) / generator_disc_loss
        
        if discr_loss_ratio < 1e-1 and train_discr:
            train_discr = False
            train_gen = True
            print('<<< real_loss=%e, fake_loss=%e, fake_loss_for_generator=%e, train_discr=%d, train_gen=%d >>>' % (discr_real_loss, discr_fake_loss, discr_fake_for_generator_loss, train_discr, train_gen))
        if discr_loss_ratio > 5e-1 and not train_discr:
            train_discr = True
            train_gen = True
            print(' <<< real_loss=%e, fake_loss=%e, fake_loss_for_generator=%e, train_discr=%d, train_gen=%d >>>' % (discr_real_loss, discr_fake_loss, discr_fake_for_generator_loss, train_discr, train_gen))
        if discr_loss_ratio > 1e1 and train_gen:
            train_gen = False
            train_discr = True
            print('<<< real_loss=%e, fake_loss=%e, fake_loss_for_generator=%e, train_discr=%d, train_gen=%d >>>' % (discr_real_loss, discr_fake_loss, discr_fake_for_generator_loss, train_discr, train_gen))

# TODO rebuild this plot
# plt.figure(figsize=(10,5))
# plt.title("Generator and Discriminator Loss During Training")
# plt.plot(G_losses,label="G")
# plt.plot(D_losses,label="D")
# plt.xlabel("iterations")
# plt.ylabel("Loss")
# plt.legend()
# plt.show()