import math
import random

import torch

import horovod.torch as hvd
from horovod.torch.mpi_ops import Average, size
from horovod.torch.mpi_ops import allreduce_async_
from horovod.torch.mpi_ops import allreduce_, allgather
from horovod.torch.mpi_ops import allgather_async as allgather_async_
from horovod.torch.mpi_ops import synchronize as synchronize_
# import cupy  # conda install -c conda-forge cupy=7.0.0=py37h0c141eb_2
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack

from src.memory import Memory
from src.memory import powersgdMemory

__all__ = ['DGCCompressor','topkCompressor','fp16Compressor','powersgdCompressor','SignSGDCompressor','EFSignSGDCompressor','OneBitCompressor','QSGDCompressor','RandomKCompressor','SignumCompresson','TernGradCompressor','TernAllreduceCompressor','NoneCompressor']


class DGCCompressor:
    def __init__(self, compress_ratio, memory=None,
                 sample_ratio=0.01, strided_sample=True,
                 compress_upper_bound=1.3, compress_lower_bound=0.8, max_adaptation_iters=10, resample=True,
                 fp16_values=False, int32_indices=False,
                 warmup_epochs=-1, warmup_coeff=None):
        self.world_size = hvd.size()
        self.op = Average
        self.fp16_values = fp16_values
        self.int32_indices = int32_indices

        self.base_compress_ratio = self.compress_ratio = \
            compress_ratio if compress_ratio <= 1.0 else 1.0 / compress_ratio
        self.memory = Memory if memory is None else memory
        self.warmup_epochs = warmup_epochs
        if self.warmup_epochs > 0:
            if warmup_coeff is None:
                self.warmup_coeff = self.base_compress_ratio \
                                    ** (1. / (self.warmup_epochs + 1))
            else:
                if isinstance(warmup_coeff, (tuple, list)):
                    assert len(warmup_coeff) >= self.warmup_epochs
                    for wc in warmup_coeff:
                        assert 0 < wc <= 1
                else:
                    assert 0 < warmup_coeff <= 1
                self.warmup_coeff = warmup_coeff
        else:
            self.warmup_coeff = 1

        self.sample_ratio = min(max(sample_ratio, 0.01), 1.0)
        self.strided_sample = strided_sample
        self.compress_upper_bound = compress_upper_bound
        self.compress_lower_bound = compress_lower_bound
        self.max_adaptation_iters = max_adaptation_iters
        self.resample = resample

        self.attributes = {}
    def add_attributes(self,numel,name,shape):
        if self.sample_ratio < 1.0:
            pct_numel = int(math.ceil(numel * self.sample_ratio))  # 与num_selects计算方法相同，数量*采样率=采样个数  2304*0.01=24 向上取整
            cpr_numel = int(math.ceil(2 / self.compress_ratio))  # 压缩率为0.001时 cpr_numel=2000
            if numel <= cpr_numel:  # 由于压缩率过高，梯度参数也不多，所以只能取1个梯度进行传输的情况
                if hvd.rank() == 0:
                    print(f'Warning: {name} with {numel} elements transmits 1 gradient element')
                sample_stride = 1
                num_samples = numel
            else:
                sample_stride = int(math.ceil(numel / max(pct_numel, cpr_numel) / 32)) * 32 + 1    # ceil((2304/2000)/32) *32+1 = 33
                num_samples = numel // sample_stride  # 2304//33 =  69
                while num_samples < max(pct_numel, cpr_numel):  # 69 < 2000
                    sample_stride = sample_stride - 8
                    num_samples = numel // sample_stride
        else:
            sample_stride = 1
            num_samples = numel
        top_k_samples = int(math.ceil(num_samples * self.compress_ratio))
        num_selects = int(math.ceil(numel * self.compress_ratio))
        self.attributes[name] = (numel, shape, num_selects, num_samples, top_k_samples, sample_stride)
        #self.attributes[name+str(1)] = (numel, shape, num_selects, num_samples, top_k_samples, sample_stride)
        #self.attributes[name+str(2)] = (numel, shape, num_selects, num_samples, top_k_samples, sample_stride)
        if hvd.rank() == 0:
            print(f'   {name:<25}: transmit {num_selects} / {numel} elements of shape {shape}\n'
                    f'   {" " * 25}  threshold {top_k_samples} / {num_samples} samples'
                    f' {f"at stride {sample_stride}" if self.strided_sample else "uniformly"}')
    
    def initialize(self, named_parameters):
        if hvd.rank() == 0:
            print("=> initializing dgc compressor")
        #print(named_parameters)
        for name, param in named_parameters:
            if torch.is_tensor(param):
                numel = param.numel()
                shape = list(param.size())
            else:
                assert isinstance(param, (list, tuple))
                numel, shape = param[0], param[1]
            self.add_attributes(numel,name,shape)
            
    
    def warmup_compress_ratio(self, epoch):
        if self.warmup_epochs > 0:
            if epoch < self.warmup_epochs:
                if isinstance(self.warmup_coeff, (tuple, list)):
                    compress_ratio = self.warmup_coeff[epoch]
                else:
                    compress_ratio = max(self.warmup_coeff ** (epoch + 1),
                                        self.base_compress_ratio)
            else:
                compress_ratio = self.base_compress_ratio
        else:
            compress_ratio = self.base_compress_ratio
        if compress_ratio != self.compress_ratio:
            if hvd.rank() == 0:
                print(f'update compress ratio: {compress_ratio}')
            self.compress_ratio = compress_ratio
            self.initialize(self.attributes.items())

    def _sparsify(self, tensor, name):
        #print("Use Compression._sparsify.")
        tensor = tensor.view(-1)
        numel, shape, num_selects, num_samples, top_k_samples, sample_stride = self.attributes[name]

        importance = tensor.abs()
        if numel == num_samples:
            samples = importance
        else:
            if self.strided_sample:
                sample_start = random.randint(0, sample_stride - 1)
                samples = importance[sample_start::sample_stride]
            else:
                samples = importance[torch.randint(0, numel, (num_samples, ), device=tensor.device)]

        threshold = torch.min(torch.topk(samples, top_k_samples, 0, largest=True, sorted=False)[0])
        mask = torch.ge(importance, threshold)
        indices = mask.nonzero().view(-1)
        num_indices = indices.numel()

        if numel > num_samples:
            # code modified from https://github.com/sands-lab/grace/blob/master/grace_dl/torch/compressor/dgc.py
            for _ in range(self.max_adaptation_iters):
                if num_indices > num_selects:
                    if num_indices > num_selects * self.compress_upper_bound:
                        if self.resample:
                            indices = indices[
                                torch.topk(importance[indices], num_selects,
                                           0, largest=True, sorted=False)[1]
                            ]
                            break
                        else:
                            threshold = threshold * self.compress_upper_bound
                    else:
                        break
                elif num_indices < self.compress_lower_bound * num_selects:
                    threshold = threshold * self.compress_lower_bound
                else:
                    break
                mask = torch.ge(importance, threshold)
                indices = mask.nonzero().view(-1)
                num_indices = indices.numel()
        indices = indices[:num_selects]
        diff = abs(indices.size().numel()-num_selects)
        if diff!=0:
            padding = torch.zeros(diff,device='cuda').long()
            indices = torch.cat((indices,padding),0)
        """if indices.size().numel()!=num_selects:
            print(f'tensor name:{name}, tensor size:{tensor.size()}, indices number is:{indices.size()}, and num_selects is:{num_selects}')"""
        values = tensor[indices]
        return values, indices, numel, shape, num_selects

    def compress(self, tensor, name):
        #print("Use DGC Compression.")
        if name not in self.attributes:
            #print(f'try to add attributes:{name}')
            self.add_attributes(tensor.numel(),name,tensor.shape)
        if self.compress_ratio < 1.0 and name in self.attributes:
            # compress
            #print(f'compressing...Name is in self.attributes:{name}')
            tensor_compensated = self.memory.compensate(
                tensor, name, accumulate=True)
            values, indices, numel, shape, num_selects = \
                self._sparsify(tensor_compensated, name)
            self.memory.update(name, (indices, ))
            indices = indices.view(-1, 1)
            values = values.view(-1, 1)

            ctx = (name, numel, shape, values.dtype, indices.dtype,
                   tensor.data.view(numel))
            if self.fp16_values and values.dtype.is_floating_point:
                values = values.type(torch.float16)
            if self.int32_indices and not indices.dtype.is_floating_point:
                indices = indices.type(torch.int32)
            #
            return (values, indices), ctx
        else:
            ctx = (name, None, None, tensor.dtype, None, None)
            if self.fp16_values and tensor.dtype.is_floating_point:
                tensor = tensor.type(torch.float16)
            #print(f'name is not in self.attributes:{name}')
            return tensor, ctx

    def decompress(self, tensor, ctx):
        name, numel, shape, vdtype, idtype, grad = ctx
        if name not in self.attributes:
            #print("something goes wrong...")
            self.add_attributes(numel,name,shape)
        if self.compress_ratio < 1.0 and name in self.attributes:
            # decompress
            assert isinstance(tensor, (list, tuple))
            values, indices = tensor
            values = values.view(-1)
            indices = indices.view(-1)
            if self.fp16_values and vdtype.is_floating_point:
                values = values.type(vdtype)
            if self.int32_indices and not idtype.is_floating_point:
                indices = indices.type(idtype)
            grad.zero_().index_put_([indices], values, accumulate=True)
            if self.op == Average:
                grad.mul_(1. / self.world_size)
            return grad.view(shape)
        else:
            if self.fp16_values and vdtype.is_floating_point:
                tensor = tensor.type(vdtype)
            return self.memory.compensate(tensor, name, accumulate=False)

    def communicate(self, tensor_compressed, name, op):
        self.op = op
        if self.compress_ratio < 1.0 and name in self.attributes:
            return [allgather_async_(t, name=f'{name}.t{e}')
                    for e, t in enumerate(tensor_compressed)]
        else:
            return allreduce_async_(tensor_compressed, name=name, op=op)

    def synchronize(self, handle):
        if isinstance(handle, (tuple, list)):
            return [synchronize_(h) for h in handle]
        else:
            return synchronize_(handle)

class topkCompressor:
    def __init__(self, compress_ratio):
        super().__init__()
        self.compress_ratio = compress_ratio

    def compress(self, tensor, name):
        #print(f"call topk, compress_ratio is:{self.compress_ratio}")
        tensors = sparsify(tensor, self.compress_ratio)
        ctx = tensor.numel(), tensor.size()
        return tensors, ctx

    def decompress(self, tensors, ctx):
        """Decompress by filling empty slots with zeros and reshape back using the original shape"""
        numel, shape = ctx
        tensor_decompressed = desparsify(tensors, numel)
        return tensor_decompressed.view(shape)


def sparsify(tensor, compress_ratio):
    tensor = tensor.flatten()
    k = max(1, int(tensor.numel() * compress_ratio))
    _, indices = torch.topk(tensor.abs(), k, sorted=False,)
    values = torch.gather(tensor, 0, indices)
    #print(f'tensor\'s numel is:{tensor.numel()}, slect elements:{indices.size()}, value nums is:{values.size()}')
    return values, indices


def desparsify(tensors, numel):
    values, indices = tensors
    tensor_decompressed = torch.zeros(numel, dtype=values.dtype, layout=values.layout, device=values.device)
    tensor_decompressed.scatter_(0, indices, values)
    return tensor_decompressed

class fp16Compressor:
    def __init__(self, average=True, tensors_size_are_same=True):
        self.average = average
        self.tensors_size_are_same = tensors_size_are_same
    def compress(self, tensor, name):
        """Downcasts the tensor to 16-bit."""
        dtype = tensor.dtype
        if dtype.is_floating_point:
            # Only allow compression from other floating point types
            tensor = tensor.type(torch.float16)
        return [tensor], dtype

    def decompress(self, tensors, dtype):
        """Upcasts the tensor to the initialization dtype."""
        tensor_decompressed, = tensors
        if dtype.is_floating_point:
            tensor_decompressed = tensor_decompressed.type(dtype)
        return tensor_decompressed


@torch.jit.script
def orthogonalize(matrix,eps=torch.tensor(1e-8)):
    n, m = matrix.shape
    for i in range(m):
        # Normalize the i'th column
        col = matrix[:, i: i + 1]
        col /= torch.sqrt(torch.sum(col ** 2))+ eps
        # Project it on the rest and remove it
        if i + 1 < m:
            rest = matrix[:, i + 1:]
            # rest -= torch.matmul(col.t(), rest) * col
            rest -= torch.sum(col * rest, dim=0) * col
class powersgdCompressor:
    def __init__(self, average=True, tensors_size_are_same=True, memory=False, compress_rank=1):
        print("powersgd init...")
        self.average = average
        self.tensors_size_are_same = tensors_size_are_same
        self.q_memory = {}
        self.compress_rank = compress_rank

        self.memory = powersgdMemory(q_memory=self.q_memory,compress_rank=compress_rank) if memory else Memory
        self.name_count=0
    def compress(self, tensor, name):
        if tensor.dim() == 1:
            return tensor, None

        device = tensor.device
        shape = tensor.size()
        #######################################
        # compensate
        #if(isinstance(self.memory,powersgdMemory)):
        tensor = self.memory.compensate(tensor,name)
        #######################################
        matrix = tensor.view([shape[0], -1])
        #print(f'matrix init status is:{matrix}')
        n, m = matrix.size()
        r = min(n, m, self.compress_rank)
        
        if name in self.q_memory:
            q = self.q_memory[name]
        else:
            q = torch.empty(m, r, dtype=matrix.dtype, layout=matrix.layout, device=matrix.device).normal_()
            # q, _ = torch.qr(q)
            #orthogonalize(q)
        #q = self.q_memory[name]
        #orthogonalize(q)

        p = torch.mm(matrix, q)
        #print(f'p before allreduce_ is:{p}')
        p = allreduce_(tensor=p,name=name+str(self.name_count))
        #print(f'p after allreduce_ is:{p}')
        self.name_count+=1
        # p, _ = torch.qr(p)
        orthogonalize(p)
        #print(f'p after orthogonalize is:{p}')
        q = torch.mm(matrix.t(), p)
        #print(f'q before allreduce_ is:{q}')
        q = allreduce_(tensor=q,name=name+str(self.name_count))
        self.name_count+=1
        ctx = p, q, shape
        self.q_memory[name] = q

        #######################################
        # update
        #if(isinstance(self.memory,powersgdMemory)):
        tensor = self.memory.update(tensor,name,self,[],ctx)
        #######################################

        return torch.zeros([1], device=device), ctx

    def decompress(self, tensors, ctx):
        if ctx is None:
            tensor = tensors
            return tensor
        p, q, tensor_shape = ctx
        new_tensor = torch.mm(p, q.t())
        tensor_decompressed = new_tensor.view(tensor_shape)
        return tensor_decompressed


class SignSGDCompressor:

    def __init__(self):
        self.average = False
        self.tensors_size_are_same = True

    def compress(self, tensor, name):
        shape = tensor.size()
        tensor = tensor.flatten()

        tensor_compressed = tensor >= 0
        return [tensor_compressed.type(torch.uint8)], shape

    def decompress(self, tensors, shape):
        """Decoding the signs to float format """
        sign_encode, = tensors
        sign_decode = sign_encode.type(torch.float32) * 2 - 1
        tensor_decompressed = sign_decode.view(shape)
        return tensor_decompressed

    def aggregate(self, tensors):
        """Aggregate a list of tensors."""
        agged_tensor = sum(tensors)
        sign = agged_tensor >= 0
        agged_tensor = sign * 2.0 - 1.0
        return agged_tensor

class EFSignSGDCompressor:

    def __init__(self):
        self.average = False
        self.tensors_size_are_same = True
        #self.learning_rate = lr

    def compress(self, tensor, name):
        """Encoding and compressing the signs """
        shape = tensor.size()
        tensor = tensor.flatten()

        sign_encode = tensor >= 0
        mean = tensor.abs().mean()
        tensor_compressed = mean, sign_encode.type(torch.uint8)

        return tensor_compressed, shape

    def decompress(self, tensor_compressed, shape):
        """Decoding the signs to float format """
        mean, sign_encode = tensor_compressed
        sign_decode = sign_encode.type(torch.float32) * 2 - 1
        sign_decode = mean * sign_decode
        tensor_decompressed = sign_decode.view(shape)
        return tensor_decompressed

    def aggregate(self, tensors):
        """Aggregate a list of tensors."""
        return sum(tensors) / self.learning_rate

"""class NaturalCompressor:

    def compress(self, tensor, name):
        shape = tensor.size()
        tensor_flatten = tensor.flatten()
        cupy_tensor = cupy.fromDlpack(to_dlpack(tensor_flatten))
        tensor_cast = cupy_tensor.view(cupy.int32)
        sign = tensor_cast & cupy.int32(0b10000000000000000000000000000000)
        exp = tensor_cast & cupy.int32(0b01111111100000000000000000000000)
        mantissa = tensor_cast & cupy.int32(0b00000000011111111111111111111111)
        exp_add_one = mantissa > cupy.random.randint(low=0, high=0b00000000011111111111111111111111,
                                                     size=cupy_tensor.shape,
                                                     dtype=cupy.int32)
        exponent = cupy.where(exp_add_one, exp + 0b00000000100000000000000000000000, exp)
        exp_shift = cupy.clip(exponent, a_min=0b00001001000000000000000000000000, a_max=0b01001000100000000000000000000000)
        exps = cupy.right_shift(exp_shift, 23)
        exps = cupy.bitwise_or(cupy.right_shift(sign, 24), exps - 18)
        tensor_compressed = exps.astype(cupy.uint8)
        return [from_dlpack(tensor_compressed.toDlpack())], shape


    def decompress(self, tensor_compressed, shape):
        tensor_compressed, = tensor_compressed
        cupy_tensor = cupy.fromDlpack(to_dlpack(tensor_compressed))
        sign = cupy_tensor > 127
        exps = cupy.bitwise_and(cupy_tensor, 0b01111111)
        floats = cupy.left_shift((exps + 18).astype(cupy.int32), 23).view(cupy.float32)
        tensor_decompressed = cupy.where(sign, -floats, floats)
        tensor_decompressed = cupy.multiply((exps >= 1).astype(cupy.float32), tensor_decompressed)
        return from_dlpack(tensor_decompressed.toDlpack()).view(shape)"""

class OneBitCompressor:

    def compress(self, tensor, name):
        shape = tensor.size()
        tensor = tensor.flatten()
        numel = tensor.numel()

        mask0 = tensor < 0
        sum0 = torch.sum(tensor[mask0])
        num0 = torch.sum(mask0).float()
        mean0 = sum0 / num0 if num0 > 0 else sum0

        mask1 = ~mask0
        sum1 = torch.sum(tensor[mask1])
        num1 = numel - num0
        mean1 = sum1 / num1 if num1 > 0 else sum1

        tensor_compressed = mask0.type(torch.uint8), mean0, mean1

        return tensor_compressed, shape

    def decompress(self, tensor_compressed, shape):
        mask0, mean0, mean1 = tensor_compressed
        mask0= mask0.bool()
        tensor_decompressed = mask0 * mean0 + ~mask0 * mean1
        tensor_decompressed = tensor_decompressed.view(shape)
        return tensor_decompressed


class QSGDCompressor:

    def __init__(self, quantum_num):
        self.quantum_num = quantum_num

    def compress(self, tensor, name):
        shape = tensor.size()
        tensor = tensor.flatten()

        norm = tensor.norm()
        norm = norm.flatten()
        abs_gradient = tensor.abs()

        level_float = self.quantum_num / norm * abs_gradient
        previous_level = level_float.floor()
        prob = torch.empty_like(tensor).uniform_()
        is_next_level = (prob < (level_float - previous_level)).type(torch.float32)
        new_level = (previous_level + is_next_level)

        sign = tensor.sign()
        tensor_compressed = (new_level * sign).type(torch.int16)
        tensor_compressed = tensor_compressed.type(torch.int8 if self.quantum_num < 128 else torch.half)
        tensor_compressed = tensor_compressed, norm

        return tensor_compressed, shape

    def decompress(self, tensor_compressed, shape):
        tensor_compressed, norm = tensor_compressed

        decode_output = tensor_compressed.type(torch.float32)
        tensor_decompressed = norm / self.quantum_num * decode_output
        tensor_decompressed = tensor_decompressed.view(shape)
        return tensor_decompressed


def randomKSparsify(tensor, compress_ratio):
    tensor = tensor.flatten()
    numel = tensor.numel()
    k = max(1, int(numel * compress_ratio))
    indices = torch.randperm(numel, device=tensor.device)[:k]
    values = tensor[indices]
    return indices, values


class RandomKCompressor:
    """Python libraries Based Compress by performing sparsification (i.e., sending a ratio of the actual tensor size."""

    def __init__(self, compress_ratio, average = True):
        self.average = average
        self.global_step = 0
        self.compress_ratio = compress_ratio

    def compress(self, tensor, name):
        """Use Python Random libraries RNG to compress by generating a list of indices to be transmitted."""

        h = sum(bytes(name, encoding='utf8'), self.global_step)
        self.global_step += 1
        torch.manual_seed(h)
        indices, values = randomKSparsify(tensor, self.compress_ratio)

        ctx = indices, tensor.numel(), tensor.size()
        return [values], ctx

    def decompress(self, tensors, ctx):
        """Decompress by filling empty slots with zeros and reshape back using the original shape"""
        indices, numel, shape = ctx
        values, = tensors
        tensor_decompressed = torch.zeros(numel, dtype=values.dtype, layout=values.layout, device=values.device)
        tensor_decompressed.scatter_(0, indices, values)
        return tensor_decompressed.view(shape)

class SignumCompresson:

    def __init__(self, momentum):
        self.average = False
        self.momentum = momentum
        self.momentums = {}

    def compress(self, tensor, name):
        """Encoding and compressing the signs """
        shape = tensor.size()
        tensor = tensor.flatten()

        # update tensor by momentum
        if name in self.momentums:
            tensor = (1.0 - self.momentum) * tensor + self.momentum * self.momentums[name]
        self.momentums[name] = tensor
        tensor_compressed = tensor >= 0
        return [tensor_compressed.type(torch.uint8)], shape

    def decompress(self, tensors, shape):
        sign_encode, = tensors
        """Decoding the signs to float format """
        sign_decode = sign_encode.type(torch.float32) * 2 - 1
        tensor_decompressed = sign_decode.view(shape)
        return tensor_decompressed

    def aggregate(self, tensors):
        """Aggregate a list of tensors."""
        agged_tensor = sum(tensors)
        agged_tensor = agged_tensor >= 0
        agged_tensor = agged_tensor * 2.0 - 1.0
        return agged_tensor

class TernGradCompressor:

    def compress(self, tensor, name):
        shape = tensor.size()
        tensor = tensor.flatten()

        std = (tensor - torch.mean(tensor)) ** 2
        std = torch.sqrt(torch.mean(std))
        c = 2.5 * std.item()
        gradient = torch.clamp(tensor, -c, c)
        abs_gradient = gradient.abs()
        scalar = abs_gradient.max()

        sign_gradient = gradient.sign() * scalar
        rnd_sample = torch.empty_like(tensor).uniform_(0, scalar.item())
        sign_gradient[rnd_sample >= abs_gradient] = 0
        new_sign = sign_gradient.sign()  # -1, 0, 1

        tensor_compressed = new_sign.type(torch.int8), scalar.flatten()

        return tensor_compressed, shape

    def decompress(self, tensor_compressed, shape):
        tensor_compressed, scalar = tensor_compressed
        sign = tensor_compressed.type(torch.float32)
        tensor_decompressed = sign * scalar
        return tensor_decompressed.view(shape)

def tensor_clamp(tensor):
    c = 2.5 * torch.std(tensor).item()
    tensor_ = torch.clamp(tensor, -c, c)
    return tensor_


class TernAllreduceCompressor():
    """Quantize all floating point to ternary from."""

    def __init__(self, tensor_size_threshold=65536, compress_rate=4):
        self.compress_rate = compress_rate
        self.mul_factor = pow(2, 32//compress_rate)
        self.shift_factors = [pow(self.mul_factor, i)
                              for i in range(compress_rate)]
        self.tensor_size_threshold = tensor_size_threshold
        self.layer_params = {}
        self.index_el = 0
        self.enable_async = False
        self._size = size()

    def get_max_scaler(self, tensor, name):
        scaler = tensor.abs().max().view(1)
        scaler_name = f'{name}.scaler'
        if self.enable_async:
            handle = allgather_async_(scaler, scaler_name)
        else:
            scaler = allgather(scaler, scaler_name).max().item()
            handle = None
        return scaler, handle

    def stochastical_binarize_tensor(self, tensor, scaler):
        zeros = torch.zeros_like(tensor)
        abs_tensor = torch.abs(tensor)
        sign_tensor = torch.sign(tensor)
        rnd_sample = torch.rand_like(tensor) * scaler
        where_cond = torch.less(rnd_sample, abs_tensor)
        sign_tensor = torch.where(where_cond, sign_tensor, zeros)
        return sign_tensor

    def ternary_decoder(self, encoded_data, scaler, shape):
        """Decoding the signs to float format """
        numel = torch.prod(torch.tensor(shape))
        index_original = torch.arange(0, numel, device=encoded_data.device)
        splits = [torch.div(encoded_data, shift_factor, rounding_mode='floor') %
            self.mul_factor for shift_factor in self.shift_factors]
        decoded_summed_data = torch.gather(
            torch.cat(splits, 0), 0, index_original).view(shape)
        decoded_summed_data = decoded_summed_data.sub_(
            size()).type(torch.float)
        return decoded_summed_data * scaler / size()

    def ternary_encoder(self, tensor, scaler):
        tensor = self.stochastical_binarize_tensor(tensor, scaler)
        sum_all = 0
        e = torch.sign(tensor).type(torch.int) + 1
        redundant_size = self.compress_rate - \
            e.size(dim=0) % self.compress_rate
        e = torch.cat(
            (e, torch.zeros(redundant_size, dtype=torch.int, device=tensor.device)), 0)
        for split, shift_factor in zip(torch.chunk(e, self.compress_rate), self.shift_factors):
            sum_all += split * shift_factor
        return sum_all

    def compress(self, tensor, name):
        shape = tensor.shape
        ctx = tensor.dtype
        is_compressed = False
        tensor_compressed = tensor
        unified_scaler = 0
        self.layer_params[name] = self.layer_params.get(name, self.index_el)
        self.index_el += 1
        handle = None
        if tensor.numel() > self.tensor_size_threshold:
            tensor_compressed = tensor_clamp(tensor_compressed.flatten())
            unified_scaler, handle = self.get_max_scaler(tensor_compressed, name)
            tensor_compressed = self.ternary_encoder(
                tensor_compressed, unified_scaler)
            is_compressed = True
        return tensor_compressed * self._size, (ctx, shape, unified_scaler, is_compressed, handle)

    def decompress(self, tensors, ctx):
        tensor_decompressed = tensors
        dtype, shape, scaler, is_compressed, handle = ctx
        if is_compressed:
            if handle is not None:
                scaler = synchronize_(handle).max().item()
            tensor_decompressed = self.ternary_decoder(
                tensor_decompressed, scaler, shape)
        return tensor_decompressed

class NoneCompressor():
    """Default no-op compression."""
    @staticmethod
    def compress(tensor, name=None):
        """Returns the tensor unmodified."""
        return tensor, name

    @staticmethod
    def decompress(tensor, ctx):
        """Returns the tensor unmodified."""
        return tensor
