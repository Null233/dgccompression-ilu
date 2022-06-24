from __future__ import print_function

import argparse
from tarfile import CompressionError

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import datasets, transforms
import torchvision.datasets as datasets
import horovod.torch as hvd
import torch
from src.horovod.optimizer import DistributedOptimizer
from src.compression import *

import horovod.torch as hvd

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--compression', type=str, default='none',
                    help='Compression methods: none, terngrad, ternallreduce, \
                         powersgd, dgc, fp16, qsgd, randomk, signsgd, signum, topk')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Horovod: initialize library.
hvd.init()
torch.manual_seed(args.seed)

def log(s, nl=True):
    if hvd.rank() != 0:
        return
    print(s, end='\n' if nl else '')

if args.cuda:
    # Horovod: pin GPU to local rank.
    torch.cuda.set_device(hvd.local_rank())
    torch.cuda.manual_seed(args.seed)

# Horovod: limit # of CPU threads to be used per worker.
torch.set_num_threads(1)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_dataset = \
    datasets.MNIST('data-%d' % hvd.rank(), train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
# Horovod: use DistributedSampler to partition the training data.
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs)

test_dataset = \
    datasets.MNIST('data-%d' % hvd.rank(), train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))
# Horovod: use DistributedSampler to partition the test data.
test_sampler = torch.utils.data.distributed.DistributedSampler(
    test_dataset, num_replicas=hvd.size(), rank=hvd.rank())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                                          sampler=test_sampler, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


model = Net()

if args.cuda:
    # Move model to GPU.
    model.cuda()

# Horovod: scale learning rate by the number of GPUs.
optimizer = optim.SGD(model.parameters(), lr=args.lr * hvd.size(),
                      momentum=args.momentum)

# Horovod: (optional) compression algorithm.
if args.compression == 'dgc':
        log(f'\n==> initializing dgc compression')
        # configs.train.compression.memory = configs.train.compression.memory()
        compression = DGCCompressor(0.05)
        compression.memory.initialize(model.named_parameters())
        cpr_parameters = {}
        for name, param in model.named_parameters(): #nn.Module里面关于参数有两个很重要的属性named_parameters()和parameters()，前者给出网络层的名字和参数的迭代器，而后者仅仅是参数的迭代器。
            if param.dim() > 1:
                cpr_parameters[name] = param
        compression.initialize(cpr_parameters.items())
elif args.compression == 'topk':
    log(f'\n==> initializing topk compression')
    compression = topkCompressor(0.1)
elif args.compression == 'fp16':
    log(f'\n==> initializing fp16 compression')
    compression = fp16Compressor()
elif args.compression == 'powersgd':
    log(f'\n==> initializing powersgd compression')
    compression = powersgdCompressor(memory=True,compress_rank=4)
elif args.compression == 'sign':
    log(f'\n==> initializing signsgd compression')
    compression = SignSGDCompressor()
# elif args.compression == 'efsign':
#     log(f'\n==> initializing efsignsgd compression')
#     compression = EFSignSGDCompressor()
# elif args.compression == 'natural':
#     log(f'\n==> initializing naturalsgd compression')
#     compression = NaturalCompressor()
# elif args.compression == 'onebit':
#     log(f'\n==> initializing onebitsgd compression')
#     compression = OneBitCompressor()
elif args.compression == 'qsgd':
    log(f'\n==> initializing QSGDsgd compression')
    compression = QSGDCompressor(quantum_num=64)
elif args.compression == 'randomk':
    log(f'\n==> initializing randomksgd compression')
    compression = RandomKCompressor(compress_ratio=0.1)
elif args.compression == 'signum':
    log(f'\n==> initializing signnumsgd compression')
    compression = SignumCompresson(momentum=0.8)
elif args.compression == 'terngrad':
    log(f'\n==> initializing terngradsgd compression')
    compression = TernGradCompressor()
elif args.compression == 'ternallreduce':
    log(f'\n==> initializing ternallreduce compression')
    compression = TernAllreduceCompressor()
else:
    #compression = configs.train.compression()
    compression = NoneCompressor()
    log("Use hvd.none compression...")

# Horovod: broadcast parameters & optimizer state.
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

# Horovod: wrap optimizer with DistributedOptimizer.
optimizer = DistributedOptimizer(
    optimizer, named_parameters=model.named_parameters(), 
    compression=compression)

def train(epoch):
    model.train()
    # Horovod: set epoch to sampler for shuffling.
    train_sampler.set_epoch(epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            # Horovod: use train_sampler to determine the number of examples in
            # this worker's partition.
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_sampler), 100. * batch_idx / len(train_loader), loss.item()))


def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()


def test():
    model.eval()
    test_loss = 0.
    test_accuracy = 0.
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        test_accuracy += pred.eq(target.data.view_as(pred)).cpu().float().sum()

    # Horovod: use test_sampler to determine the number of examples in
    # this worker's partition.
    test_loss /= len(test_sampler)
    test_accuracy /= len(test_sampler)

    # Horovod: average metric values across workers.
    test_loss = metric_average(test_loss, 'avg_loss')
    test_accuracy = metric_average(test_accuracy, 'avg_accuracy')

    # Horovod: print output only on first rank.
    if hvd.rank() == 0:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
            test_loss, 100. * test_accuracy))


for epoch in range(1, args.epochs + 1):
    ps = {k:v for k, v in model.named_parameters()}
    print(ps.keys())
    train(epoch)
    test()
