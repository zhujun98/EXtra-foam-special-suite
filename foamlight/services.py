"""
Distributed under the terms of the MIT License.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import argparse
import faulthandler

from foamgraph import mkQApp

from . import __version__
from .logger import logger
from .config import config
from .facade import FacadeController


def application():
    parser = argparse.ArgumentParser(prog="foamlight")
    parser.add_argument('-V', '--version', action='version',
                        version="%(prog)s " + __version__)
    parser.add_argument("topic", help="Name of the topic",
                        choices=config.topics,
                        type=lambda s: s.upper())
    parser.add_argument("--use-gate", action='store_true',
                        help="Use Karabo gate client (experimental feature)")
    parser.add_argument('--debug', action='store_true',
                        help="Run in debug mode")

    args = parser.parse_args()

    if args.debug:
        logger.setLevel("DEBUG")
    # No ideal whether it affects the performance. If it does, enable it only
    # in debug mode.
    faulthandler.enable(all_threads=False)

    topic = args.topic

    config.load(topic, USE_KARABO_GATE_CLIENT=args.use_gate)

    app = mkQApp()
    app.setStyleSheet(
        "QTabWidget::pane { border: 0; }"
    )

    controller = FacadeController()
    controller.showFacade(topic)

    app.exec_()


if __name__ == "__main__":

    application()
