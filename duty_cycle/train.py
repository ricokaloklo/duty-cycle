import numpy as np
import torch
from sbi.utils.user_input_checks_utils import (
    MultipleIndependent,
)
from torch.distributions import Uniform
from duty_cycle.utils import LogUniform
from duty_cycle.infer import SimulationBasedInference
from duty_cycle.simulate import SigmoidDropOffVLMC

# Set up the prior
prior_mean_cont_up_time = LogUniform(torch.Tensor([0.1]), torch.Tensor([0.9]))
prior_std_cont_up_time = LogUniform(torch.Tensor([0.01]), torch.Tensor([0.5]))
prior_mean_cont_down_time = LogUniform(torch.Tensor([0.1]), torch.Tensor([0.9]))
prior_std_cont_down_time = LogUniform(torch.Tensor([0.01]), torch.Tensor([0.5]))
prior_k_cont_up = Uniform(torch.Tensor([1]), torch.Tensor([10]))
prior_k_cont_down = Uniform(torch.Tensor([1]), torch.Tensor([10]))

prior_for_sbi = MultipleIndependent(
    dists=[
        prior_mean_cont_up_time,
        prior_std_cont_up_time,
        prior_mean_cont_down_time,
        prior_std_cont_down_time,
        prior_k_cont_up,
        prior_k_cont_down,
    ],
)

inference = SimulationBasedInference(
    SigmoidDropOffVLMC(N=1000),
    prior_for_sbi,
    density_estimator="kde",
    ngridpoint=50,
    grid_range=(1e-2, 1),
    grid_spacing="log",
    nsample=5000,
)

inference.train(method="SNPE", nsimulation=10000, ncore=128)
inference.save_to_file("model.pkl")
