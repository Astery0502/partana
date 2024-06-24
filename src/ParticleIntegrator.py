import os
import re
import doctest
from collections import namedtuple
from typing import Union, List, Tuple, Callable, Iterable

import h5py
import numpy as np
import pandas as pd

# The integrator format copies from amrvac/particle module
# it is built by cython mainly

