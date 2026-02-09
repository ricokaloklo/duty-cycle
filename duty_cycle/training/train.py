import numpy as np
import torch
from sbi.utils.user_input_checks_utils import (
    MultipleIndependent,
)
from torch.distributions import Uniform
from duty_cycle.utils import LogUniform
from duty_cycle.infer import SimulationBasedInference, SummaryStatisticInference
from duty_cycle.simulate import WeibullVLMC

model_name = "test-WeibullVLMC-EXP1"

# Set up the prior
"""
For WeibullVLMC, we have the following parameters:
['scale_up',
 'shape_up',
 'initial_deadtime_up',
 'scale_down',
 'shape_down',
 'initial_deadtime_down']
"""
prior_for_sbi = MultipleIndependent(
    dists=[
        LogUniform(torch.tensor([5e-3]), torch.tensor([1.0])),  # scale_up
        LogUniform(torch.tensor([1/2.5]), torch.tensor([2.5])),  # shape_up
        Uniform(torch.tensor([0.0]), torch.tensor([1e-2])),      # initial_deadtime_up
        LogUniform(torch.tensor([5e-3]), torch.tensor([1.0])), # scale_down
        LogUniform(torch.tensor([1/2.5]), torch.tensor([2.5])),  # shape_down
        Uniform(torch.tensor([0.0]), torch.tensor([1e-2])),      # initial_deadtime_down
    ],
)

inference = SummaryStatisticInference(
    WeibullVLMC(nmax=1000, dt=1.0/1000), # For now, set dt = 1/nmax such that the total duration is always 1 time unit
    prior_for_sbi,
    density_estimator="kde",
    ngridpoint=50,
    grid_range=(1e-3, 1),
    grid_spacing="log",
    nsample=10000,
)

inference.train(method="SNPE", nsimulation=25000, ncore=128)
inference.save_to_file(f"{model_name}.pkl")
