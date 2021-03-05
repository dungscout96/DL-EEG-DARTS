import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
import h5py

class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

def _get_EEG_data(args):
    f = h5py.File(f'{args.data}/child_mind_x_train_v2.mat', 'r')
    x_train = f['X_train']
    x_train = np.reshape(x_train, (-1, 1, 24, 256))
    print('X_train shape: ' + str(x_train.shape))
    f = h5py.File('child_mind_y_train_v2.mat', 'r')
    y_train = f['Y_train']
    print('Y_train shape: ' + str(y_train.shape))
    train_data = EEGDataset(x_train, y_train, True)

    f = h5py.File(f'{args.data}/child_mind_x_val_v2.mat', 'r')
    x_val = f['X_val']
    x_val = np.reshape(x_val, (-1, 1, 24, 256))
    print('X_val shape: ' + str(x_val.shape))
    f = h5py.File('child_mind_y_val_v2.mat', 'r')
    y_val = f['Y_val']
    print('Y_val shape: ' + str(y_val.shape))
    val_data = EEGDataset(x_val, y_val, True)

    return train_data, val_data


def _data_transforms_cifar10(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform


def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path):
  model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)

class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, train):
        super(EEGDataset).__init__()
        assert x.shape[0] == y.size
        self.x = x
        # temp_y = np.zeros((y.size, 2))
        # for i in range(y.size):
        #    temp_y[i, y[i]] = 1
        # self.y = temp_y
        self.y = [y[i][0] for i in range(y.size)]
        self.train = train

    def __getitem__(self, key):
        return (self.x[key], self.y[key])

    def __len__(self):
        return len(self.y)
