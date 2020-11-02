"""
Distributed under the terms of the MIT License.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
from ..core import (
    _FoamLightApp, _BaseAnalysisCtrlWidgetS, create_app, QThreadWorker,
    QThreadKbClient
)


class DaqMonitorProcessor(QThreadWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class DaqMonitorCtrlWidget(_BaseAnalysisCtrlWidgetS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@create_app(DaqMonitorCtrlWidget,
            DaqMonitorProcessor,
            QThreadKbClient)
class DaqMonitor(_FoamLightApp):
    """DAQ monitor application."""

    icon = "daq_monitor.png"
    _title = "DAQ monitor"
    _long_title = "DAQ monitor"

    def __init__(self, topic):
        super().__init__(topic)

        self.initUI()
        self.initConnections()
        self.startWorker()

    def initUI(self):
        """Override."""
        pass

    def initConnections(self):
        """Override."""
        pass
