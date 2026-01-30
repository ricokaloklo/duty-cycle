import numpy as np

from . import _UP, _DOWN, _UNDEF
from .simulate import Simulator

class NetworkSimulator(Simulator):
    components: dict = {}
    disturbances: dict = {}
    sep_char: str = '^'

    @staticmethod
    def _get_full_param_name(param_name, component_name, sep_char):
        return f"{param_name}{sep_char}{component_name}"
    
    @staticmethod
    def _get_full_param_label(param_label, component_name, sep_char):
        return param_label[:-1] + f"^{{\\rm {component_name}}}" + param_label[-1]

    def register_params(self, simulator_dict):
        for name, component in simulator_dict.items():
            for param_name, param_label in zip(component.param_names, component.param_labels):
                self.param_names.append(self._get_full_param_name(param_name, name, self.sep_char))
                self.param_labels.append(self._get_full_param_label(param_label, name, self.sep_char))

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
        if type(simulation_params) is list:
            simulation_params = np.array(simulation_params)

        # Loop over components and disturbances to unpack parameters
        for name, component in list(self.components.items()) + list(self.disturbances.items()):
            indices_to_extract = []
            for _, param_name in enumerate(component.param_names):
                full_param_name = self._get_full_param_name(param_name, name, self.sep_char)
                if full_param_name in self.param_names:
                    index = self.param_names.index(full_param_name)
                    indices_to_extract.append(index)
            component.params = component.unpack_params(simulation_params[indices_to_extract], use_torch=use_torch)

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
