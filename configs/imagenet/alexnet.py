from torchvision.models import AlexNet

from torchpack.mtpack.utils.config import Config, configs

# model
configs.model = Config(AlexNet)
configs.model.num_classes = configs.dataset.num_classes