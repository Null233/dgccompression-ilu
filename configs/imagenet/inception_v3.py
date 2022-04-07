from torchvision.models import inception_v3

from torchpack.mtpack.utils.config import Config, configs

# model
configs.model = Config(inception_v3)
configs.model.num_classes = configs.dataset.num_classes
