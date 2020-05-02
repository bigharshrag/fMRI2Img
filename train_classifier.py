from model import Classifier
import time
import torch
from torch import nn
from torch import optim
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from constants import BaseOptions
from data.fMRIimgdataset import fMRIImgClassifierDataset
import time

args = BaseOptions('classifier').parse()
batch_size = args.batch_size
ngpu = args.ngpu

dataset = fMRIImgClassifierDataset(args, subject='sub-01', split='train')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=args.workers)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Create the generator
tfmri, _ = dataset.__getitem__(0)
classifier = Classifier(ngpu, tfmri.shape[0], 150).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    classifier = nn.DataParallel(classifier, list(range(ngpu)))

# Setup Adam optimizers for both G and D
optimizer = optim.Adam(classifier.parameters(), lr=args.c_lr, betas=(args.beta1, 0.999))

ce_loss = nn.CrossEntropyLoss()

writer = SummaryWriter('runs/' + args.exp_name)

print('batches', len(dataset) / batch_size)

total_steps = 0
# Training main loop
start = time.time()
for it in range(args.num_epochs):
    correct = 0
    total = 0
    dataset.set_split('train')
    for i, data in enumerate(dataloader):
        data_fmri, labels = data[0].to(device), data[1].to(device)
        classifier.zero_grad()
        # import pdb; pdb.set_trace()
        # Feed the data to the generator and run it
        classifier_output = classifier.forward(data_fmri)
        predictions = torch.argmax(classifier_output, dim=1)
        for idx in range(len(predictions)):
            if predictions[idx] == labels[idx]:
                correct += 1
            total +=1 
        loss = ce_loss(classifier_output, labels)

        loss.backward()

        optimizer.step()

        writer.add_scalar('data/classifier_loss', loss, total_steps)

        # print('loss: ', loss, i)
    print("Epoch {0} Train Accuracy: {1} / {2} = {3}".format(it, correct, total, float(correct)/total))
    # Save snapshot
    if it % args.snapshot_interval == 0:
        torch.save(classifier.state_dict(), 'classifier.pth')

    if it % args.snapshot_interval == 0:
        correct = 0
        total = 0
        with torch.no_grad():
            dataset.set_split('val')
            for i, data in enumerate(dataloader):
                data_fmri, labels = data[0].to(device), data[1].to(device)
                classifier_output = classifier.forward(data_fmri)
                predictions = torch.argmax(classifier_output, dim=1)

                for idx in range(len(predictions)):
                    if predictions[idx] == labels[idx]:
                        correct += 1
                    total +=1 
        print("Epoch {0} Val Accuracy: {1} / {2} = {3}".format(it, correct, total, float(correct)/total))



# TODO rebuild this plot
# plt.figure(figsize=(10,5))
# plt.title("Generator and Discriminator Loss During Training")
# plt.plot(G_losses,label="G")
# plt.plot(D_losses,label="D")
# plt.xlabel("iterations")
# plt.ylabel("Loss")
# plt.legend()
# plt.show()