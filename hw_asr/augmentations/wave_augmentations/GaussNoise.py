import torch_audiomentations
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase
from torch import distributions

class GaussNoise(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self.aug = distributions.Normal(*args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        x = x + self.aug.sample(x.shape)
        return x.squeeze(1)
