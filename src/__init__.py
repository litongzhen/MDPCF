import torch

torch.set_num_threads(1)
try:
    from icecream import install

    install()
except ImportError:
    pass

from . import env  # noqa
from .data import *  # noqa
from .deep import *  # noqa
from .env import *  # noqa
from .metrics import *  # noqa
from .util import *  # noqa
