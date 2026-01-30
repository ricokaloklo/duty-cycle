import numpy as np

from . import _UP, _DOWN, _UNDEF
from .simulate import Simulator

class NetworkSimulator(Simulator):
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

    def __init__(self, **params):
        super(NetworkSimulator, self).__init__(**params)
        self.param_names = []
        self.param_labels = []
        self.components = {}
        self.disturbances = {}

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
        """
        Unpack simulation parameters for all components and disturbances in the network.

        Parameters
        ----------
        simulation_params : array-like
            An array or list of simulation parameters.
        use_torch : bool, optional
            Whether to keep the parameters as torch tensors if they are provided as such. By default True.

        Returns
        -------
        None
        """
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
        Simulate the duty cycle of the network.
        
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
    def _simulate_step_for_disturbances(
            self,
            output,
            dt,
            idx,
    ):
        pass

    def simulate_duty_cycle(self, simulation_params, initial_state_list=_UP, idx_lastup_list=0, cont_up_time_list=None, cont_down_time_list=None):
        """
        Simulate the duty cycle of the network, with components having independent up/down segments.

        Parameters
        ----------
        simulation_params : array-like
            An array or list of simulation parameters.
        initial_state_list : int or list of int, optional
            A list of initial states for each component, by default _UP.
        idx_lastup_list : int or list of int, optional
            A list of indices of the last up state for each component, by default 0.
        cont_up_time_list : float, None, list of float or list of None, optional
            A list of continuous up times for each component, by default None.
        cont_down_time_list : float, None, list of float or list of None, optional
            A list of continuous down times for each component, by default None.

        Returns
        -------
        output : dict
            A dictionary containing the simulation results.
        """
        self.unpack_params(simulation_params, use_torch=True)

        # Fill kwargs_dict for each component
        kwargs_dict = {name: {} for name in self.components.keys()}
        for name in self.components.keys():
            if isinstance(initial_state_list, list):
                kwargs_dict[name]['initial_state'] = initial_state_list[list(self.components.keys()).index(name)]
            else:
                kwargs_dict[name]['initial_state'] = initial_state_list

            if isinstance(idx_lastup_list, list):
                kwargs_dict[name]['idx_lastup'] = idx_lastup_list[list(self.components.keys()).index(name)]
            else:
                kwargs_dict[name]['idx_lastup'] = idx_lastup_list

            if isinstance(cont_up_time_list, list):
                kwargs_dict[name]['cont_up_time'] = cont_up_time_list[list(self.components.keys()).index(name)]
            else:
                kwargs_dict[name]['cont_up_time'] = cont_up_time_list

            if isinstance(cont_down_time_list, list):
                kwargs_dict[name]['cont_down_time'] = cont_down_time_list[list(self.components.keys()).index(name)]
            else:
                kwargs_dict[name]['cont_down_time'] = cont_down_time_list

        output = {}

        return kwargs_dict
