import argparse
import timeit
import time
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import horovod.torch as hvd
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data.distributed
import torch.optim as optim
from src.horovod.optimizer import DistributedOptimizer as DGCDistributedOptimizer
from horovod.torch.optimizer import DistributedOptimizer as BasicDistributedOptimizer
from src.compression import *

# Benchmark settings
parser = argparse.ArgumentParser(description='PyTorch Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--model', type=str, default='resnet50',
                    help='model to benchmark')
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size')
parser.add_argument('--num-warmup-batches', type=int, default=10,
                    help='number of warm-up batches that don\'t count towards benchmark')
parser.add_argument('--num-batches-per-iter', type=int, default=10,
                    help='number of batches per benchmark iteration')
parser.add_argument('--num-iters', type=int, default=10,
                    help='number of benchmark iterations')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')
parser.add_argument('--optimizer', type=str, default='basic',
                    help='Optimizer to use in training: basic, scheduler, grc')
parser.add_argument('--compression', type=str, default='none',
                    help='Compression methods: none, terngrad, ternallreduce, \
                         powersgd, dgc, fp16, qsgd, randomk, signsgd, signum, topk')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
hvd.init()

def log(s, nl=True):
    if hvd.rank() != 0:
        return
    print(s, end='\n' if nl else '')

if args.cuda:
    # Horovod: pin GPU to local rank.
    torch.cuda.set_device(hvd.local_rank())

cudnn.benchmark = True

# Set up standard model.
model = getattr(models, args.model)()

# Set up groups
groups = [p for p in model.parameters() if p.numel() < 65536]
groups = [groups]
# By default, Adasum doesn't need scaling up learning rate.
lr_scaler = hvd.size() if not args.use_adasum else 1

if args.cuda:
    # Move model to GPU.
    model.cuda()
    # If using GPU Adasum allreduce, scale learning rate by local_size.
    if args.use_adasum and hvd.nccl_built():
        lr_scaler = hvd.local_size()

optimizer = optim.SGD(model.parameters(), lr=0.01 * lr_scaler)

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

groups = [list(model.parameters())]

# Horovod: wrap optimizer with DistributedOptimizer.
optimizer = DGCDistributedOptimizer(
    optimizer, named_parameters=model.named_parameters(), #groups=groups,
    compression=compression)

# Set up fixed fake data
data = torch.randn(args.batch_size, 3, 224, 224)
target = torch.LongTensor(args.batch_size).random_() % 1000
if args.cuda:
    data, target = data.cuda(), target.cuda()

def benchmark_step():
    optimizer.zero_grad()
    output = model(data)
    loss = F.cross_entropy(output, target)
    loss.backward()
    optimizer.step()

log('Model: %s' % args.model)
log('Batch size: %d' % args.batch_size)
device = 'GPU' if args.cuda else 'CPU'
log('Number of %ss: %d' % (device, hvd.size()))

# Warm-up
log('Running warmup...')
timeit.timeit(benchmark_step, number=args.num_warmup_batches)

# Benchmark
log('Running benchmark...')
img_secs = []
for x in range(args.num_iters):
    time = timeit.timeit(benchmark_step, number=args.num_batches_per_iter)
    img_sec = args.batch_size * args.num_batches_per_iter / time
    log('Iter #%d: %.1f img/sec per %s' % (x, img_sec, device))
    img_secs.append(img_sec)

# Results
img_sec_mean = np.mean(img_secs)
img_sec_conf = 1.96 * np.std(img_secs)
log('Img/sec per %s: %.1f +-%.1f' % (device, img_sec_mean, img_sec_conf))
log('Total img/sec on %d %s(s): %.1f +-%.1f' %
    (hvd.size(), device, hvd.size() * img_sec_mean, hvd.size() * img_sec_conf))
