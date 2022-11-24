import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

import os

class POD():
    def __init__(self, ntrain: int, ):
        self.nx = None
        self.nu = None
        self.u_all = None
        self.ntrain =ntrain


