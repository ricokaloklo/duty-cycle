import torch
from torch import distributions as dist

class JointIndependentDistribution:
    def __init__(self, *distributions):
        self.distributions = distributions

    def log_prob(self, x):
        return sum(d.log_prob(x_i) for d, x_i in zip(self.distributions, x))
    
    def sample(self, sample_shape=torch.Size()):
        return torch.stack([d.sample(sample_shape) for d in self.distributions], dim=-1)

# From https://github.com/pytorch/pytorch/issues/11412
class LogUniform(dist.TransformedDistribution):
    def __init__(self, low, high):
        if type(low) is not torch.Tensor:
            low = torch.tensor(low)
        if type(high) is not torch.Tensor:
            high = torch.tensor(high)

        super(LogUniform, self).__init__(
            dist.Uniform(
                low.log(),
                high.log()
            ),
            dist.ExpTransform()
        )