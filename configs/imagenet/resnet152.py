from torchvision.models import resnet152

from torchpack.mtpack.utils.config import Config, configs

# model
configs.model = Config(resnet152)
configs.model.num_classes = configs.dataset.num_classes
