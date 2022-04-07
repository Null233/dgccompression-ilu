import torch

import horovod.torch as hvd

__all__ = ['Memory', 'DGCSGDMemory', 'powersgdMemory']


# code modified from https://github.com/sands-lab/grace/blob/master/grace_dl/torch/memory/dgc.py
class Memory:
    @staticmethod
    def initialize(*args, **kwargs):
        pass

    @staticmethod
    def compensate(tensor, *args, **kwargs):
        return tensor
    
    @staticmethod
    def update(*args, **kwargs):
        pass

    @staticmethod
    def state_dict():
        return None
    
    @staticmethod
    def load_state_dict(state_dict):
        pass


class DGCSGDMemory(Memory):
    """ Memory for momentum correction in DGC for momentum SGD optimizer"""
    def __init__(self, momentum=0.9, nesterov=False,
                 gradient_clipping=None, momentum_masking=True):
        self.gradient_clipping = gradient_clipping
        self.momentum_masking = momentum_masking

        self.momentum = momentum
        self.nesterov = nesterov
        self.momentums = {}
        self.velocities = {}
    
    def initialize(self, named_parameters):
        if hvd.rank() == 0:
            print("=> initializing dgc sgd memory")
        for name, param in named_parameters:
            self.momentums[name] = torch.zeros_like(param.data)
            self.velocities[name] = torch.zeros_like(param.data)

    def compensate(self, grad, name, accumulate=True):
        """Update the velocities with the momentums."""
        if self.gradient_clipping is not None:
            grad = self.gradient_clipping(grad)
        mmt = self.momentums[name]
        if accumulate:
            vec = self.velocities[name]
            if self.nesterov:
                mmt.add_(grad).mul_(self.momentum)
                vec.add_(mmt).add_(grad)
            else:
                mmt.mul_(self.momentum).add_(grad)
                vec.add_(mmt)
            return vec
        else:
            if self.nesterov:
                mmt.add_(grad).mul_(self.momentum)
                return mmt.add(grad)
            else:
                mmt.mul_(self.momentum).add_(grad)
                return mmt.clone()  # TODO: save this clone

    def update(self, name, ctx):
        """Update the momentums."""
        indices = ctx[0]
        if self.momentum_masking:
            self.momentums[name].view(-1).index_fill_(0, indices, 0)
        self.velocities[name].view(-1).index_fill_(0, indices, 0)

    def state_dict(self):
        return dict(momentums=self.momentums, velocities=self.velocities)

    def load_state_dict(self, state_dict):
        momentums = state_dict['momentums']
        velocities = state_dict['velocities']
        for name in self.momentums.keys():
            if name in momentums:
                self.momentums[name] = momentums[name]
                self.velocities[name] = velocities[name]


class powersgdMemory(Memory):
    def __init__(self, q_memory, compress_rank=1):
        self.compress_rank = compress_rank
        self.q_memory = q_memory
        self.residuals = {}

    def compensate(self, tensor, name):
        """Update the tensor with the residuals."""
        if tensor.dim() == 1:
            return tensor

        if name in self.q_memory:
            tensor += self.residuals[name]

        shape = tensor.size()
        n = shape[0]
        m = 1
        for dim in shape[1:]:
            m = m * dim

        r = min(n, m, self.compress_rank)
        normal = torch.empty(m, r, dtype=tensor.dtype, layout=tensor.layout, device=tensor.device).normal_()
        self.q_memory[name] = normal

        return tensor

    def update(self, tensor, name, compressor, tensor_compressed, ctx):
        """Update the residuals."""
        if ctx is None:
            return

        self.residuals[name] = tensor - compressor.decompress(tensor_compressed, ctx)