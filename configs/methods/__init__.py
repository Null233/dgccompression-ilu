from torchpack.mtpack.utils.config import Config, configs

from src.compression import DGCCompressor
from src.memory import DGCSGDMemory
from src.optim import DGCSGD


configs.train.dgc = False
configs.train.compression = Config(DGCCompressor)
configs.train.compression.compress_ratio = 0.05
configs.train.compression.sample_ratio = 0.01
configs.train.compression.strided_sample = True
configs.train.compression.compress_upper_bound = 1.3
configs.train.compression.compress_lower_bound = 0.8
configs.train.compression.max_adaptation_iters = 10
configs.train.compression.resample = True

configs.train.topk = False

configs.train.fp16 = False

configs.train.powersgd = False

# uint8 doesn't support
configs.train.sign = False

# uint8 doesn't support
configs.train.efsign = False

configs.train.natural = False

# uint8 doesn't support
configs.train.onebit = False

configs.train.qsgd = False

configs.train.randomk = True

# uint8 doesn't support
configs.train.signum = False

configs.train.terngrad = False

configs.train.ternallreduce =  False


old_optimizer = configs.train.optimizer
configs.train.optimizer = Config(DGCSGD)
for k, v in old_optimizer.items():
    configs.train.optimizer[k] = v

configs.train.compression.memory = Config(DGCSGDMemory)
configs.train.compression.memory.momentum = configs.train.optimizer.momentum
