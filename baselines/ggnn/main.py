import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim

from utils.model import ClassPrediction, GGNN, BiGGNN, ContrastiveLoss
from utils.train import train_ggnn, train_biggnn
from utils.test import test_ggnn, test_biggnn
from utils.data.dataset import MonoLanguageProgramData, CrossLingualProgramData
from utils.data.dataloader import bAbIDataloader
from tensorboardX import SummaryWriter
import os

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--train_batch_size', type=int, default=32, help='input batch size')
parser.add_argument('--test_batch_size', type=int, default=32, help='input batch size')
parser.add_argument('--state_dim', type=int, default=5, help='GGNN hidden state size')
parser.add_argument('--n_steps', type=int, default=10, help='propogation steps number of GGNN')
parser.add_argument('--niter', type=int, default=150, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--verbal', type=bool, default=True, help='print training info or not')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--n_classes', type=int, default=104, help='manual seed')
parser.add_argument('--directory', default="program_data/github_cpp_babi_format_Sep-29-2018-0000006", help='encoded program data')
parser.add_argument('--right_directory', default="", help='right encoded program data')
parser.add_argument('--model_path', default="model", help='path to save the model')
parser.add_argument('--n_hidden', type=int, default=50, help='number of hidden layers')
parser.add_argument('--size_vocabulary', type=int, default=59, help='maximum number of node types')
parser.add_argument('--is_training_ggnn', type=bool, default=False, help='Training GGNN or BiGGNN')
parser.add_argument('--training', action="store_true",help='is training')
parser.add_argument('--testing', action="store_true",help='is testing')
parser.add_argument('--data_percentage', type=float, default=1.0 ,help='percentage of data use for training')
parser.add_argument('--loss', type=int, default=0 ,help='1 is contrastive loss, 0 is cross entropy loss')
parser.add_argument('--log_path', default="logs/ggnn" ,help='log path for tensorboard')
parser.add_argument('--epoch', type=int, default=-1 ,help='which epoch to start or test')

opt = parser.parse_args()
print(opt)
previous_runs = os.listdir(opt.log_path)
if len(previous_runs) == 0:
    run_number = 1
else:
    run_number = max([int(s.split('run-')[1]) for s in previous_runs]) + 1
writer = SummaryWriter("%s/run-%03d" % (opt.log_path, run_number))


if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

# This part is the implementation to illustrate Graph-Level output from program data
def main(opt):
    if opt.right_directory == "":
       train_dataset = MonoLanguageProgramData(opt.size_vocabulary, opt.directory, True, opt.n_classes, opt.data_percentage)
    else:
       train_dataset = CrossLingualProgramData(opt.size_vocabulary, opt.directory,opt.right_directory, True, opt.loss, opt.n_classes, opt.data_percentage)
    train_dataloader = bAbIDataloader(train_dataset, batch_size=opt.train_batch_size, shuffle=True, num_workers=2)

    if opt.right_directory == "":
       test_dataset = MonoLanguageProgramData(opt.size_vocabulary, opt.directory, False, opt.n_classes, opt.data_percentage)
    else:
       test_dataset = CrossLingualProgramData(opt.size_vocabulary, opt.directory,opt.right_directory, False, opt.loss, opt.n_classes, opt.data_percentage)
    test_dataloader = bAbIDataloader(test_dataset, batch_size=opt.test_batch_size, shuffle=True, num_workers=2)

    opt.annotation_dim = 1  # for bAbI
    opt.n_edge_types = train_dataset.n_edge_types
    opt.n_node = train_dataset.n_node

    epoch = -1
    epochs = os.listdir(opt.model_path)
    if opt.right_directory == "":
       sep = ':'
    else:
       sep = '#'
    if opt.epoch != -1 and os.path.exists("{}/{}{}".format(opt.model_path, sep, opt.epoch)):
       print("Using No. {} of the saved models...".format(opt.epoch))
       net = torch.load("{}/{}{}".format(opt.model_path, sep, opt.epoch))
       epoch = opt.epoch + 1
    else:
       if os.path.exists(opt.model_path):
         if len(epochs) > 0:
           for s in epochs:
              if sep in s:
                 epoch = max(epoch, int(s.split(sep)[1]))
           if epoch != -1:
              # find the last epoch
              for i in range(0, epoch):
                  if not os.path.exists("{}/{}{}".format(opt.model_path, sep, i)):
                     epoch = i - 1
                     break
              if epoch != -1:
                 print("Using No. {} of the saved models...".format(epoch))
                 net = torch.load("{}/{}{}".format(opt.model_path, sep, epoch))
                 epoch = epoch + 1
    if epoch == -1:
       if opt.right_directory == "":
          net = GGNN(opt)
       else:
          net = BiGGNN(opt)
       net.double()
       epoch = 0

    if opt.loss == 1:
        criterion = ContrastiveLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    if opt.cuda:
        net.cuda()
        criterion.cuda()
    optimizer = optim.Adam(net.parameters(), lr=opt.lr)

    if opt.training:
        for epoch in range(epoch, opt.niter):
          if opt.right_directory == "":
            train_ggnn(epoch, train_dataloader, net, criterion, optimizer, opt, writer)
          else:
            train_biggnn(epoch, train_dataloader, net, criterion, optimizer, opt, writer)
          if opt.right_directory == "":
             test_ggnn(test_dataloader, net, criterion, optimizer, opt)
          else:
             test_biggnn(test_dataloader, net, criterion, optimizer, opt)

        writer.close()

    if opt.testing:
       if opt.right_directory == "":
          test_ggnn(test_dataloader, net, criterion, optimizer, opt)
       else:
          test_biggnn(test_dataloader, net, criterion, optimizer, opt)

if __name__ == "__main__":
    main(opt)
