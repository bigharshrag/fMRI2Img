from model import Encoder, Generator, Discriminator
import torch
from torch import nn
from torch import optim
import torchvision.utils as vutils
from matplotlib import pyplot as plt
from constants import ngpu, nz, ngf, beta1, lr, batch_size, workers, num_epochs

dataset = [] # TODO @Rishabh

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

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
print(discriminator)# Create the generator

# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(ngf, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(geenrator.parameters(), lr=lr, betas=(beta1, 0.999))

# Training Loop
train_discr = True
train_gen = True

# Training main loop
start = time.time()
for it in range(num_epochs):
    for data_fmri, data_image in dataloader:
        # Feed the data (images) to the encoder and run it        
        encoded_real_image = encoder.forward(data_image)

        # Feed the data to the generator and run it
        generator_output = generator.forward(data_fmri)
        # TODO compute recon loss on generator_output vs data_image
        recon_loss = 0

        # Feed the generated image into the encoder
        encoded_generated_image = encoder.forward(generated_output)
        # TODO compute recon loss on encoded_generated_image vs encoded_real_image
        recon_loss = 0

        # Run the discriminator on real image
        real_classification = discriminator.forward(data_image)
        if train_discr:
            # TODO compute loss for classification, apply backward
            pass


        # Run the discriminator on generated data
        generated_classification = discriminator.forward(generator_output)
        if train_discr:
            # TODO compute loss for classification, apply backward
            pass

        # Run the discriminator on generated data with opposite labels, to get the gradient for the generator
        # TODO figure out what this means / what's going on here
        # discriminator.net.blobs['data'].data[...] = generated_img
        # discriminator.net.blobs['label'].data[...] = np.zeros((batch_size,1,1,1), dtype='float32')
        # discriminator.net.forward_simple()
        # discr_fake_for_generator_loss = np.copy(discriminator.net.blobs['discr_loss'].data)
        
        if train_gen:
            # TODO optimize generator by applying losses
            pass
            # generator.increment_iter()
            # generator.net.clear_param_diffs()
            # encoder.net.backward_simple()
            # discriminator.net.backward_simple()
            # generator.net.blobs['generated'].diff[...] = encoder.net.blobs['data'].diff + discriminator.net.blobs['data'].diff
            # generator.net.backward_simple()
            # generator.apply_update()

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
        
        discr_loss_ratio = (discr_real_loss + discr_fake_loss) / discr_fake_for_generator_loss
        
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