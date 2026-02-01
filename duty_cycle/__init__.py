_UP = 1 # Detector is in observation-intent/science mode
_DOWN = 0 # Detector is NOT in observation-intent/science mode

# NOTE This state is not strictly necessary, but can be useful for initialization
_UNDEF = -1 # Detector state is undefined

from . import simulate
from . import simulate_external
from . import simulate_network
from . import utils
from . import infer