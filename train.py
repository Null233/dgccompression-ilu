import argparse
import math
import os
import random
import shutil

import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import horovod.torch as hvd
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
from tqdm import tqdm
import time

from torchpack.mtpack.utils.config import Config, configs

import torch.optim as optim
from src.horovod.optimizer import DistributedOptimizer
from src.compression import *

from src.horovod.compression import Compression


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+') #参数在使用可以提供的个数，+ 表示可以有一个或多个参数
    parser.add_argument('--devices', default='gpu')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--suffix', default='')
    args, opts = parser.parse_known_args()

    ##################
    # Update configs #
    ##################

    printr(f'==> loading configs from {args.configs}')
    Config.update_from_modules(*args.configs)
    Config.update_from_arguments(*opts)

    if args.devices is not None and args.devices != 'cpu':
        configs.device = 'cuda'
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())
        cudnn.benchmark = True
    else:
        configs.device = 'cpu'

    if 'seed' in configs and configs.seed is not None:
        random.seed(configs.seed)
        np.random.seed(configs.seed)
        torch.manual_seed(configs.seed)
        if configs.device == 'cuda' and configs.get('deterministic', True):
            cudnn.deterministic = True
            cudnn.benchmark = False
    
    configs.train.num_batches_per_step = \
        configs.train.get('num_batches_per_step', 1)

    configs.train.save_path = get_save_path(*args.configs) \
                              + f'{args.suffix}.np{hvd.size()}'
    printr(f'[train.save_path] = {configs.train.save_path}')
    checkpoint_path = os.path.join(configs.train.save_path, 'checkpoints')
    configs.train.checkpoint_path = os.path.join(
        checkpoint_path, f'e{"{epoch}"}-r{hvd.rank()}.pth'
    )
    configs.train.latest_pth_path = os.path.join(
        checkpoint_path, f'latest-r{hvd.rank()}.pth'
    )
    configs.train.best_pth_path = os.path.join(
        checkpoint_path, f'best-r{hvd.rank()}.pth'
    )
    os.makedirs(checkpoint_path, exist_ok=True)

    if args.evaluate:
        configs.train.latest_pth_path = configs.train.best_pth_path

    printr(configs)

    #####################################################################
    # Initialize DataLoaders, Model, Criterion, LRScheduler & Optimizer #
    #####################################################################
    
    printr(f'\n==> creating model "{configs.model}"')
    model = configs.model()
    model = model.cuda()
    """for name,param in model.named_parameters():
        print(f"name:{name}, params:{param}")"""

    printr(f'\n==> creating dataset "{configs.dataset}"')
    dataset = load_imagenet(num_classes=configs.dataset.num_classes,image_size=configs.dataset.image_size)
    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(configs.data.num_threads_per_worker)
    loader_kwargs = {'num_workers': configs.data.num_threads_per_worker,
                     'pin_memory': True} if configs.device == 'cuda' else {}
    # When supported, use 'forkserver' to spawn dataloader workers
    # instead of 'fork' to prevent issues with Infiniband implementations
    # that are not fork-safe
    if (loader_kwargs.get('num_workers', 0) > 0 and
            hasattr(mp, '_supports_context') and
            mp._supports_context and
            'forkserver' in mp.get_all_start_methods()):
        loader_kwargs['multiprocessing_context'] = 'forkserver'
    printr(f'\n==> loading dataset "{loader_kwargs}""')
    samplers, loaders = {}, {}
   

    samplers['train'] = torch.utils.data.distributed.DistributedSampler(
        dataset[0], num_replicas=hvd.size(), rank=hvd.rank())
    loaders['train'] = torch.utils.data.DataLoader(
        dataset[0], batch_size=configs.train.batch_size * (
            configs.train.num_batches_per_step),
        sampler=samplers['train'],
        drop_last=(configs.train.num_batches_per_step > 1),
        **loader_kwargs
    )

    samplers['test'] = torch.utils.data.distributed.DistributedSampler(
        dataset[1], num_replicas=hvd.size(), rank=hvd.rank())
    loaders['test'] = torch.utils.data.DataLoader(
        dataset[0], batch_size=configs.train.batch_size,
        sampler=samplers['test'],
        drop_last=False,
        **loader_kwargs
    )




    criterion = configs.train.criterion().to(configs.device)
    # Horovod: scale learning rate by the number of GPUs.
    configs.train.base_lr = configs.train.optimizer.lr
    configs.train.optimizer.lr *= (configs.train.num_batches_per_step
                                   * hvd.size())
    printr(f'\n==> creating optimizer "{configs.train.optimizer}"')

    if not configs.train.dgc:
        optimizer = optim.SGD(model.parameters(), lr=configs.train.optimizer.lr,
                      momentum=configs.train.optimizer.momentum)
    elif configs.train.optimize_bn_separately:
        optimizer = configs.train.optimizer([
            dict(params=get_common_parameters(model)),
            dict(params=get_bn_parameters(model), weight_decay=0)
        ])
    else:
        optimizer = configs.train.optimizer(model.parameters())

    # Horovod: (optional) compression algorithm.
    printr(f'\n==> creating compression "{configs.train.compression}"')
    if configs.train.dgc:
        printr(f'\n==> initializing dgc compression')
        configs.train.compression.memory = configs.train.compression.memory()
        compression = configs.train.compression()
        compression.memory.initialize(model.named_parameters())
        print(model.named_parameters())
        cpr_parameters = {}
        for name, param in model.named_parameters(): #nn.Module里面关于参数有两个很重要的属性named_parameters()和parameters()，前者给出网络层的名字和参数的迭代器，而后者仅仅是参数的迭代器。
            if param.dim() > 1:
                cpr_parameters[name] = param
        compression.initialize(cpr_parameters.items())
    elif configs.train.topk:
        printr(f'\n==> initializing topk compression')
        compression = topkCompressor(0.1)
    elif configs.train.fp16:
        printr(f'\n==> initializing fp16 compression')
        compression = fp16Compressor()
    elif configs.train.powersgd:
        printr(f'\n==> initializing powersgd compression')
        compression = powersgdCompressor(memory=True,compress_rank=4)
    elif configs.train.sign:
        printr(f'\n==> initializing signsgd compression')
        compression = SignSGDCompressor()
    elif configs.train.efsign:
        printr(f'\n==> initializing efsignsgd compression')
        compression = EFSignSGDCompressor()
    elif configs.train.natural:
        printr(f'\n==> initializing naturalsgd compression')
        compression = NaturalCompressor()
    elif configs.train.onebit:
        printr(f'\n==> initializing onebitsgd compression')
        compression = OneBitCompressor()
    elif configs.train.qsgd:
        printr(f'\n==> initializing QSGDsgd compression')
        compression = QSGDCompressor(quantum_num=64)
    elif configs.train.randomk:
        printr(f'\n==> initializing randomksgd compression')
        compression = RandomKCompressor(compress_ratio=0.1)
    elif configs.train.signum:
        printr(f'\n==> initializing signnumsgd compression')
        compression = SignumCompresson(momentum=0.8)
    elif configs.train.terngrad:
        printr(f'\n==> initializing terngradsgd compression')
        compression = TernGradCompressor()
    else:
        #compression = configs.train.compression()
        compression = Compression.none
        print("Use hvd.none compression...")
##################################################################################################

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters(),
        compression=compression,
        backward_passes_per_step=configs.train.num_batches_per_step,
        op=hvd.Average
    )
##################################################################################################
    # resume from checkpoint
    last_epoch, best_metric = -1, None
    if os.path.exists(configs.train.latest_pth_path):
        printr(f'\n[resume_path] = {configs.train.latest_pth_path}')
        checkpoint = torch.load(configs.train.latest_pth_path)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint.pop('model'))
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint.pop('optimizer'))
        if configs.train.dgc and 'compression' in checkpoint:
            compression.memory.load_state_dict(checkpoint.pop('compression'))
        last_epoch = checkpoint.get('epoch', last_epoch)
        best_metric = checkpoint.get('meters', {}).get(
            f'{configs.train.metric}_best', best_metric)
        # Horovod: broadcast parameters.
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    else:
        printr('\n==> train from scratch')
        # Horovod: broadcast parameters & optimizer state.
        printr('\n==> broadcasting paramters and optimizer state')
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)    # 从根节点广播初始化的参数
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    num_steps_per_epoch = len(loaders['train'])
    if 'scheduler' in configs.train and configs.train.scheduler is not None:  # 动态调整学习率
        if configs.train.schedule_lr_per_epoch:
            last = max(last_epoch - configs.train.warmup_lr_epochs - 1, -1)
        else:
            last = max((last_epoch - configs.train.warmup_lr_epochs + 1)
                       * num_steps_per_epoch - 2, -1)
        scheduler = configs.train.scheduler(optimizer, last_epoch=last)
    else:
        scheduler = None

    
    ############
    # Training #
    ############
    print("before evaluate")
    meters = evaluate(model, device=configs.device, meters=configs.train.meters,
                      loader=loaders['test'], split='test')
    print("after evaluate")
    for k, meter in meters.items():
        printr(f'[{k}] = {meter:2f}')
    if args.evaluate or last_epoch >= configs.train.num_epochs:
        return

    if hvd.rank() == 0:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(configs.train.save_path)
    else:
        writer = None

    all_time=0
    for current_epoch in range(last_epoch + 1, configs.train.num_epochs):
        printr(f'\n==> training epoch {current_epoch}'
                f'/{configs.train.num_epochs}')

        if configs.train.dgc:
            compression.warmup_compress_ratio(current_epoch)
        
        start_train_time = time.time()
        train(model=model, loader=loaders['train'],     #调用函数
              device=configs.device, epoch=current_epoch, 
              sampler=samplers['train'], criterion=criterion,
              optimizer=optimizer, scheduler=scheduler,
              batch_size=configs.train.batch_size,
              num_batches_per_step=configs.train.num_batches_per_step,
              num_steps_per_epoch=num_steps_per_epoch,
              warmup_lr_epochs=configs.train.warmup_lr_epochs,
              schedule_lr_per_epoch=configs.train.schedule_lr_per_epoch,
              writer=writer, quiet=hvd.rank() != 0)
        end_train_time = time.time()
        diff = end_train_time-start_train_time
        print(f'this epoch\'s train spend:{str(diff)}s')
        all_time+=diff

        meters = dict()
        for split, loader in loaders.items():
            if split != 'train':
                meters.update(evaluate(model, loader=loader,
                                       device=configs.device,
                                       meters=configs.train.meters,
                                       split=split, quiet=hvd.rank() != 0))

        best = False
        if 'metric' in configs.train and configs.train.metric is not None:
            if best_metric is None or best_metric < meters[configs.train.metric]:
                best_metric, best = meters[configs.train.metric], True
            meters[configs.train.metric + '_best'] = best_metric

        if writer is not None:
            num_inputs = ((current_epoch + 1) * num_steps_per_epoch
                          * configs.train.num_batches_per_step
                          * configs.train.batch_size * hvd.size())
            print('')
            for k, meter in meters.items():
                print(f'[{k}] = {meter:2f}')
                writer.add_scalar(k, meter, num_inputs)

        checkpoint = {
            'epoch': current_epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'meters': meters,
            'compression': compression.memory.state_dict() \
                            if configs.train.dgc else None
        }

        # save checkpoint
        checkpoint_path = \
            configs.train.checkpoint_path.format(epoch=current_epoch)
        torch.save(checkpoint, checkpoint_path)
        shutil.copyfile(checkpoint_path, configs.train.latest_pth_path)
        if best:
            shutil.copyfile(checkpoint_path, configs.train.best_pth_path)
        if current_epoch >= 3:
            os.remove(
                configs.train.checkpoint_path.format(epoch=current_epoch - 3)
            )
        printr(f'[save_path] = {checkpoint_path}')
    print(f'spend all time:{all_time}s')


def train(model, loader, device, epoch, sampler, criterion, optimizer,
          scheduler, batch_size, num_batches_per_step, num_steps_per_epoch, warmup_lr_epochs, schedule_lr_per_epoch, writer=None, quiet=True):
    step_size = num_batches_per_step * batch_size
    num_inputs = epoch * num_steps_per_epoch * step_size * hvd.size()
    _r_num_batches_per_step = 1.0 / num_batches_per_step

    sampler.set_epoch(epoch)
    model.train()
    for step, (inputs, targets) in enumerate(tqdm(
            loader, desc='train', ncols=0, disable=quiet)):
        adjust_learning_rate(scheduler, epoch=epoch, step=step,
                             num_steps_per_epoch=num_steps_per_epoch,
                             warmup_lr_epochs=warmup_lr_epochs,
                             schedule_lr_per_epoch=schedule_lr_per_epoch)

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad()

        loss = torch.tensor([0.0])
        for b in range(0, step_size, batch_size):
            _inputs = inputs[b:b+batch_size]
            _targets = targets[b:b+batch_size]
            _outputs = model(_inputs)
            #_loss = criterion(_outputs[0], _targets)
            _loss = criterion(_outputs, _targets)
            _loss.mul_(_r_num_batches_per_step)
            _loss.backward()
            loss += _loss.item()
        
        optimizer.step()

        # write train loss log
        loss = hvd.allreduce(loss, name='loss').item()
        
        if writer is not None:
            num_inputs += step_size * hvd.size()
            writer.add_scalar('loss/train', loss, num_inputs)


def evaluate(model, loader, device, meters, split='test', quiet=True):
    _meters = {}
    for k, meter in meters.items():
        _meters[k.format(split)] = meter()
    meters = _meters

    model.eval()

    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc=split):
            inputs = inputs.to(device, non_blocking=True)  # to方法： Returns a Tensor with the specified device and (optional) dtype.例如，转换一个固定内存的CPU张量到CUDA张量。
            targets = targets.to(device, non_blocking=True)

            outputs = model(inputs)
            for meter in meters.values():
                meter.update(outputs, targets)

    for k, meter in meters.items():
        data = meter.data()
        for dk, d in data.items():
            data[dk] = \
                hvd.allreduce(torch.tensor([d]), name=dk, op=hvd.Sum).item()
        meter.set(data)
        meters[k] = meter.compute()
    return meters


# Horovod: using `lr = base_lr * hvd.size()` from the very beginning
# leads to worse final accuracy.
# Scale the learning rate `lr = base_lr` ---> `lr = base_lr * hvd.size()`
# during the first five epochs. See https://arxiv.org/abs/1706.02677.
def adjust_learning_rate(scheduler, epoch, step, num_steps_per_epoch,
                         warmup_lr_epochs=0, schedule_lr_per_epoch=False):
    if epoch < warmup_lr_epochs:
        size = hvd.size()
        epoch += step / num_steps_per_epoch
        factor = (epoch * (size - 1) / warmup_lr_epochs + 1) / size
        for param_group, base_lr in zip(scheduler.optimizer.param_groups,
                                        scheduler.base_lrs):
            param_group['lr'] = base_lr * factor
    elif schedule_lr_per_epoch and (step > 0 or epoch == 0):
        return
    elif epoch == warmup_lr_epochs and step == 0:
        for param_group, base_lr in zip(scheduler.optimizer.param_groups,
                                        scheduler.base_lrs):
            param_group['lr'] = base_lr
        return
    else:
        scheduler.step()

def get_bn_parameters(module):
    def get_members_fn(m):
        if isinstance(m, nn.BatchNorm2d):
            return m._parameters.items()
        else:
            return dict()
    gen = module._named_members(get_members_fn=get_members_fn)
    for _, elem in gen:
        yield elem


def get_common_parameters(module):
    def get_members_fn(m):
        if isinstance(m, nn.BatchNorm2d):
            return dict()
        else:
            for n, p in m._parameters.items():
                yield n, p

    gen = module._named_members(get_members_fn=get_members_fn)
    for _, elem in gen:
        yield elem


def get_save_path(*configs, prefix='runs'):
    memo = dict()
    for c in configs:
        cmemo = memo
        c = c.replace('configs/', '').replace('.py', '').split('/')
        for m in c:
            if m not in cmemo:
                cmemo[m] = dict()
            cmemo = cmemo[m]

    def get_str(m, p):
        n = len(m)
        if n > 1:
            p += '['
        for i, (k, v) in enumerate(m.items()):
            p += k
            if len(v) > 0:
                p += '.'
            p = get_str(v, p)
            if n > 1 and i < n - 1:
                p += '+'
        if n > 1:
            p += ']'
        return p

    return os.path.join(prefix, get_str(memo, ''))


def printr(*args, **kwargs):
    if hvd.rank() == 0:
        print(*args, **kwargs)


def load_imagenet(num_classes, image_size, val_ratio=None, extra_train_transforms=None):
    train_transforms_pre = [
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip()
    ]
    train_transforms_post = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    if extra_train_transforms is not None:
        if not isinstance(extra_train_transforms, list):
            extra_train_transforms = [extra_train_transforms]
        for ett in extra_train_transforms:
            if isinstance(ett, (transforms.LinearTransformation, transforms.Normalize, transforms.RandomErasing)):
                train_transforms_post.append(ett)
            else:
                train_transforms_pre.append(ett)
    train_transforms = transforms.Compose(train_transforms_pre + train_transforms_post)

    test_transforms = [
        transforms.Resize(int(image_size / 0.875)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    test_transforms = transforms.Compose(test_transforms)

    transforms1 = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    #train = datasets.ImageNet(root=root, split='train', download=False, transform=train_transforms)
    #test = datasets.ImageNet(root=root, split='val', download=False, transform=test_transforms)

    train = datasets.ImageFolder(root='/gs/home/lwang20/jzb_horovod_test/adversarial-patch-master/imagenetdata/val/',transform=train_transforms)
    test = datasets.ImageFolder(root='/gs/home/lwang20/jzb_horovod_test/adversarial-patch-master/imagenetdata/val_test/',transform=test_transforms)

    #train = datasets.ImageFolder(root='/gs/home/lwang20/jzb_horovod_test/adversarial-patch-master/imagenet/imagenet',transform=transforms1)
    #test = datasets.ImageFolder(root='/gs/home/lwang20/jzb_horovod_test/adversarial-patch-master/imagenet/imagenet_test/',transform=transforms1)
    return train,test

if __name__ == '__main__':
    hvd.init()
    main()
