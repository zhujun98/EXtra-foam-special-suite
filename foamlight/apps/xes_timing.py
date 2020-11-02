"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file BSD_LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.

Modified from https://github.com/European-XFEL/EXtra-foam, version 1.0.0
Copyright (C) Jun Zhu
"""
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSplitter

from foamgraph import (
    FoamColor, ImageViewF, SmartStringLineEdit, TimedPlotWidgetF
)
from pyfoamalgo import SimplePairSequence

from ..core import (
    create_app, profiler, QThreadWorker, QThreadKbClient,
    _BaseAnalysisCtrlWidgetS, _FoamLightApp
)

_MAX_N_BINS = 999
_MAX_WINDOW = 180000  # 60 s * 10 train/s * 300 pulses/train


class XesTimingProcessor(QThreadWorker):
    """XES timing processor.

    Attributes:
        _output_channel (str): output channel name.
        _ppt (str): property name.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._output_channel = ''
        self._ppt = "data.image.data"  # 'data.adc'

        self._delay_device = ''
        self._delay_ppt = 'integerProperty'

        self._delay_scan = SimplePairSequence(max_len=_MAX_WINDOW)

    def onOutputChannelChanged(self, value: str):
        self._output_channel = value

    def onDelayDeviceChanged(self, value: str):
        self._delay_device = value

    def sources(self):
        """Override."""
        return [
            (self._output_channel, self._ppt, 1),
            (self._delay_device, self._delay_ppt, 0),
        ]

    @profiler("XES timing processor")
    def process(self, data):
        """Override."""
        data, meta = data["raw"], data["meta"]

        tid = self.getTrainId(meta)

        img = self.squeezeToImage(
            tid, self.getPropertyData(data, self._output_channel, self._ppt))
        if img is None:
            return

        delay = self.getPropertyData(
            data, self._delay_device, self._delay_ppt)

        self._delay_scan.append((tid, delay))

        self.log.info(f"Train {tid} processed")

        return {
            "displayed": img,
            "delay_scan": self._delay_scan.data(),
        }


class XesTimingCtrlWidget(_BaseAnalysisCtrlWidgetS):
    """XES timing control widget."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.output_ch_le = SmartStringLineEdit(
            "FXE_XAD_JF1M/DET/RECEIVER-2:daqOutput")
        self.delay_device_le = SmartStringLineEdit(
            "FXE_AUXT_LIC/DOOCS/PPODL")

        self.output_ch_le = SmartStringLineEdit(
            "camera1:output")
        self.delay_device_le = SmartStringLineEdit(
            "data_generator")

        self._non_reconfigurable_widgets = [
            self.output_ch_le,
            self.delay_device_le
        ]

        self.initUI()
        self.initConnections()

    def initUI(self):
        """Override."""
        layout = self.layout()

        layout.addRow("Output channel: ", self.output_ch_le)
        layout.addRow("Delay device: ", self.delay_device_le)

    def initConnections(self):
        """Override."""
        pass


class XesTimingView(ImageViewF):
    """XesTimingView class.

    Visualize the detector image.
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(has_roi=True, parent=parent)

    def updateF(self, data):
        """Override."""
        self.setImage(data['displayed'])


class XesTimingDelayScan(TimedPlotWidgetF):
    """XesTimingDelayScan class.

    Visualize path of laser delay scan.
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self.setTitle("Laser delay scan")
        self.setLabel('left', "Position (arb.)")
        self.setLabel('bottom', "Train ID")

        self._plot = self.plotCurve(pen=FoamColor.mkPen('b'))

    def refresh(self):
        """Override."""
        self._plot.setData(*self._data['delay_scan'])


@create_app(XesTimingCtrlWidget,
            XesTimingProcessor,
            QThreadKbClient)
class XesTiming(_FoamLightApp):
    """XES timing application."""

    icon = "xes_timing.png"
    _title = "XES timing"
    _long_title = "X-ray emission spectroscopy timing tool"

    def __init__(self, topic):
        super().__init__(topic)

        self._view = XesTimingView(parent=self)
        self._delay_scan = XesTimingDelayScan(parent=self)

        self.initUI()
        self.initConnections()
        self.startWorker()

    def initUI(self):
        """Override."""
        right_panel = QSplitter(Qt.Vertical)
        right_panel.addWidget(self._view)
        right_panel.addWidget(self._delay_scan)
        right_panel.setSizes([3 * self._TOTAL_H / 4, self._TOTAL_H / 4])

        cw = self.centralWidget()
        cw.addWidget(right_panel)
        cw.setSizes([self._TOTAL_W / 4, 3 * self._TOTAL_W / 4])

        self.resize(self._TOTAL_W, self._TOTAL_H)

    def initConnections(self):
        """Override."""
        self._ctrl_widget_st.output_ch_le.value_changed_sgn.connect(
            self._worker_st.onOutputChannelChanged)
        self._ctrl_widget_st.output_ch_le.returnPressed.emit()

        self._ctrl_widget_st.delay_device_le.value_changed_sgn.connect(
            self._worker_st.onDelayDeviceChanged)
        self._ctrl_widget_st.delay_device_le.returnPressed.emit()
