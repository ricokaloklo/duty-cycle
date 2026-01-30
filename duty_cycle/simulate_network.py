import numpy as np

from . import _UP, _DOWN, _UNDEF
from .simulate import Simulator

class NetworkSimulator(Simulator):
    def register_params(self, simulator_dict):
        for name, component in simulator_dict.items():
            for param_name, param_label in zip(component.param_names, component.param_labels):
                full_param_name = f"{param_name}{self.sep_char}{name}"
                self.param_names.append(full_param_name)
                full_param_label = param_label[:-1] + f"^{{\\rm {name}}}" + param_label[-1]
                self.param_labels.append(full_param_label)


    def initialize_network(self, components, sep_char='^'):
        """
        Initialize a network of components for simulation.

        Parameters
        ----------
        components : dict
            A dictionary where keys are component names and values are simulator types.
        sep_char : str, optional
            A separator character used in naming components, by default '^'.
        """
        self.components = {name: simulator(dt=self.dt, nmax=self.nmax) for name, simulator in components.items()}
        self.sep_char = sep_char
        self.register_params(self.components)

    def initialize_disturbances(self, disturbances):
        """
        Initialize external disturbances affecting the network.

        Parameters
        ----------
        disturbances : dict
            A dictionary where keys are disturbance names and values are disturbance simulator types.
        """
        self.disturbances = {name: simulator(dt=self.dt, nmax=self.nmax) for name, simulator in disturbances.items()}
        self.register_params(self.disturbances)

    def unpack_params(self, simulation_params, use_torch=True):
        # Loop over components and disturbances to unpack parameters
        pass

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
