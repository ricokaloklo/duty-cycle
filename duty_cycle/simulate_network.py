import numpy as np

from . import _UP, _DOWN, _UNDEF
from .simulate import Simulator

class NetworkSimulator(Simulator):
    def initialize_network(self, components, sep_char='_'):
        """
        Initialize a network of components for simulation.

        Parameters
        ----------
        components : dict
            A dictionary where keys are component names and values are simulator types.
        sep_char : str, optional
            A separator character used in naming components, by default '_'.
        """
        self.components = {name: simulator(dt=self.dt, nmax=self.nmax) for name, simulator in components.items()}
        self.sep_char = sep_char

    def initialize_disturbances(self, disturbances):
        """
        Initialize external disturbances affecting the network.

        Parameters
        ----------
        disturbances : list
            A list of ExternalDisturbance objects.
        """
        self.disturbances = disturbances
        # Timeseries flags tracking the state of each disturbance over time
        # _UP means the disturbance is triggered, _DOWN means it is not triggered
        self.disturbances_flags = {
            n : np.ones(self.nmax, dtype=int)*_UNDEF
            for n in range(len(disturbances))
        }

    def simulate_duty_cycle(self, simulation_params):
        """
        Simulate the duty cycle of the network over time.
        
        Parameters
        ----------
        simulation_params : dict
            A dictionary of parameters for the simulation.
        
        Returns
        -------
        output : dict
            A dictionary containing the simulation results.
        """
        return NotImplementedError

class IndependentUpDownSegmentsNetworkSimulator(NetworkSimulator):
    pass
