import torch
from torch import distributions as dist

# From https://github.com/pytorch/pytorch/issues/11412
class LogUniform(dist.TransformedDistribution):
    def __init__(self, lower_bound, upper_bound):
        super(LogUniform, self).__init__(
            dist.Uniform(
                lower_bound.log(),
                upper_bound.log()
            ),
            dist.ExpTransform()
        )