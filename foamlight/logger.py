"""
Distributed under the terms of the MIT License.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import os
import logging
from logging.handlers import TimedRotatingFileHandler
import threading

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QPlainTextEdit

from . import ROOT_PATH


def create_logger(name):
    """Create the logger object for the whole API."""
    _logger = logging.getLogger(name)

    log_file = os.path.join(ROOT_PATH, name + ".log")
    fh = TimedRotatingFileHandler(log_file, when='midnight')

    fh.suffix = "%Y%m%d"
    formatter = logging.Formatter(
        '%(asctime)s - %(filename)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(filename)s - %(levelname)s - %(message)s'
    )

    ch.setFormatter(formatter)

    _logger.addHandler(fh)
    _logger.addHandler(ch)

    return _logger


logger = create_logger("foamlight")
logger.setLevel(logging.INFO)


class GuiLogger(logging.Handler):
    def __init__(self, parent):
        super().__init__(level=logging.INFO)
        self.widget = QPlainTextEdit(parent)

        formatter = logging.Formatter('%(levelname)s - %(message)s')
        self.setFormatter(formatter)

        logger_font = QFont()
        logger_font.setPointSize(11)
        self.widget.setFont(logger_font)

        self.widget.setReadOnly(True)
        self.widget.setMaximumBlockCount(500)

    def emit(self, record):
        # guard logger from other threads
        if threading.current_thread() is threading.main_thread():
            self.widget.appendPlainText(self.format(record))
