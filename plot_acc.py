from matplotlib import pyplot as plt
import numpy as np

def get_acc(fname, n_epoch=25, k_val=3):
  f = open(fname, 'r')
  train_acc = []
  val_acc = []
  curr_epoch = -1
  for line in f.readlines():
    if line.startswith('Epoch'):
      curr_epoch = int(line.split(' ')[-1])

    if line.startswith('K: {0} Val'.format(k_val)):
      val_acc.append((curr_epoch, float(line.split(' ')[-1])))
    if line.startswith('K: {0} Train'.format(k_val)):
      train_acc.append((curr_epoch, float(line.split(' ')[-1])))
  
  return np.array(train_acc), np.array(val_acc)

sub_1_train, sub_1_val = get_acc('subject_1_fmri.log')
sub_2_train, sub_2_val = get_acc('subject_2_fmri.log')
sub_3_train, sub_3_val = get_acc('subject_3_fmri.log')

plt.title('FMRI-based image classification')
plt.plot(sub_1_train[:, 0], sub_1_train[:, 1], color='red', label='Subject 1 Training')
plt.plot(sub_2_train[:, 0], sub_2_train[:, 1], color='orange', label='Subject 2 Training')
plt.plot(sub_3_train[:, 0], sub_3_train[:, 1], color='yellow', label='Subject 3 Training')
plt.plot(sub_1_val[:, 0], sub_1_val[:, 1], color='blue', label='Subject 1 Testing')
plt.plot(sub_2_val[:, 0], sub_2_val[:, 1], color='cyan', label='Subject 2 Testing')
plt.plot(sub_3_val[:, 0], sub_3_val[:, 1], color='green', label='Subject 3 Testing')

plt.legend()
plt.show()