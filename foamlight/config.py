"""
Distributed under the terms of the MIT License.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import collections

import numpy as np


class Config(collections.abc.Mapping):

    __data = {
        "TOPIC": "",
        "MAX_N_PULSES_PER_TRAIN": 2700,
        "EXTENSION_PORT": 5555,  # FIXME
        "USE_KARABO_GATE_CLIENT": False,
        "DEFAULT_CLIENT_PORT": 45454,
        "CLIENT_TIME_OUT": 0.1,  # second
        # initial (width, height) of an application window
        "APP_WINDOW_SIZE":  (1680, 1080),
        # interval in milliseconds for polling new data and update the plot
        "APP_PLOT_UPDATE_TIMER": 10,
        # colors of for ROI bounding boxes 1 to 4
        "GUI_ROI_COLORS": ('b', 'r', 'g', 'o'),
    }

    def __getitem__(self, idx):
        """Overload."""
        return self.__data.__getitem__(idx)

    def __iter__(self):
        """Overload."""
        return self.__data.__iter__()

    def __len__(self):
        """overload."""
        return self.__data.__len__()

    def load(self, topic, **kwargs):
        self.__data.__setitem__("TOPIC", topic)

        for k, v in kwargs.items():
            if k in self:
                self.__data.__setitem__(k, v)
            else:
                raise KeyError

    @property
    def topics(self):
        return 'ACC', 'SPB', 'FXE', 'SCS', 'SQS', 'MID', 'HED'


config = Config()

_MAX_INT32 = np.iinfo(np.int32).max
_MIN_INT32 = np.iinfo(np.int32).min

_IMAGE_DTYPE = np.float32
_PIXEL_DTYPE = np.float32

# TODO: improve
_MAX_N_GOTTHARD_PULSES = 120

GOTTHARD_DEVICE = {
    "MID": "MID_EXP_DES/DET/GOTTHARD_RECEIVER:output",
    "SCS": "SCS_PAM_XOX/DET/GOTTHARD_RECEIVER1:daqOutput",
}
