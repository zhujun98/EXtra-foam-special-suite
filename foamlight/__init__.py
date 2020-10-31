"""
Distributed under the terms of the MIT License.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import os
from foamgraph import mkQApp


__version__ = "0.1.0dev"


# root path for storing config and log files
ROOT_PATH = os.path.join(os.path.expanduser("~"), ".foamlight")
if not os.path.isdir(ROOT_PATH):
    os.makedirs(ROOT_PATH)
