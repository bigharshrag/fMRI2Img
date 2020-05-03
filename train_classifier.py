from model import Classifier
import time
import torch
import numpy as np
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

val_dataset = fMRIImgClassifierDataset(args, subject='sub-01', split='test')
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=args.workers)


# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Create the generator
tfmri, _, _ = dataset.__getitem__(0)
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
TOP_K = [1, 2, 3, 4, 5]

# Training main loop
start = time.time()
for it in range(args.num_epochs):
    correct = np.zeros(len(TOP_K))
    total = np.zeros(len(TOP_K))
    classifier.train()
    for i, data in enumerate(dataloader):
        data_fmri, labels = data[0].to(device), data[1].to(device)
        classifier.zero_grad()
        # import pdb; pdb.set_trace()
        # Feed the data to the generator and run it
        classifier_output = classifier.forward(data_fmri)
        classifier_prob = nn.functional.softmax(classifier_output, dim=1)
        for k_idx in range(len(TOP_K)):
            predictions = torch.topk(classifier_prob, TOP_K[k_idx], dim=1).indices

            for idx in range(predictions.shape[0]):
                if labels[idx] in predictions[idx]:
                    correct[k_idx] += 1
                total[k_idx] +=1 

        loss = ce_loss(classifier_output, labels)

        loss.backward()

        optimizer.step()

        writer.add_scalar('data/classifier_loss', loss, total_steps)

        # print('loss: ', loss, i)
    print("Epoch {0}".format(it))
    for k_idx in range(len(TOP_K)):
        print("K: {0} Train Accuracy: {1} / {2} = {3}".format(TOP_K[k_idx], correct[k_idx], total[k_idx], float(correct[k_idx])/total[k_idx]))
    # Save snapshot
    if it % args.snapshot_interval == 0:
        torch.save(classifier.state_dict(), 'classifier.pth')

    if it % args.snapshot_interval == 0:
        correct = np.zeros(len(TOP_K))
        total = np.zeros(len(TOP_K))
        with torch.no_grad():
            classifier.eval()
            for i, data in enumerate(val_loader):
                data_fmri, labels = data[0].to(device), data[1].to(device)
                classifier_output = classifier.forward(data_fmri)
                classifier_prob = nn.functional.softmax(classifier_output, dim=1)

                for k_idx in range(len(TOP_K)):
                    predictions = torch.topk(classifier_prob, TOP_K[k_idx], dim=1).indices

                    for idx in range(predictions.shape[0]):
                        if labels[idx] in predictions[idx]:
                            correct[k_idx] += 1
                        total[k_idx] +=1 
            for k_idx in range(len(TOP_K)):
                print("K: {0} Val Accuracy: {1} / {2} = {3}".format(TOP_K[k_idx], correct[k_idx], total[k_idx], float(correct[k_idx])/total[k_idx]))



# TODO rebuild this plot
# plt.figure(figsize=(10,5))
# plt.title("Generator and Discriminator Loss During Training")
# plt.plot(G_losses,label="G")
# plt.plot(D_losses,label="D")
# plt.xlabel("iterations")
# plt.ylabel("Loss")
# plt.legend()
# plt.show()