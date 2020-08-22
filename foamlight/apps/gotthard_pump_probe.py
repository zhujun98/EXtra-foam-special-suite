"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file BSD_LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.

Modified from https://github.com/European-XFEL/EXtra-foam, version 1.0.0
Copyright (C) Jun Zhu
"""
import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import QSplitter

from foamgraph import (
    FoamColor, ImageViewF, PlotWidgetF, SmartLineEdit, SmartSliceLineEdit,
    SmartStringLineEdit
)
from pyfoamalgo import MovingAverageArray

from ..config import _MAX_N_GOTTHARD_PULSES, _PIXEL_DTYPE, GOTTHARD_DEVICE
from ..core import (
    create_special, profiler, QThreadKbClient, QThreadWorker,
    _BaseAnalysisCtrlWidgetS, _FoamLightApp
)
from ..exceptions import ProcessingError


class GotthardPumpProbeProcessor(QThreadWorker):
    """Gotthard pump-probe analysis processor.

    Attributes:
        _output_channel (str): output channel name.
        _on_slicer (slice): a slicer used to slice on-pulses in a train.
        _off_slicer (slice): a slicer used to slice off-pulses in a train.
        _poi_index (int): index of the pulse of interest for pump-probe.
        _dark_slicer (slice): a slicer used to slice dark pulses in a train.
        _dark_poi_index (int): index of the pulse of interest for dark.
        _vfom_ma (numpy.ndarray): moving average of the vector figure-of-merit
            data. Shape=(pulses, pixels)
    """

    _vfom_ma = MovingAverageArray()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._output_channel = ""
        self._ppt = "data.adc"

        self._on_slicer = slice(None, None)
        self._off_slicer = slice(None, None)
        self._poi_index = 0

        self._dark_slicer = slice(None, None)
        self._dark_poi_index = 0

        del self._vfom_ma

    def onOutputChannelChanged(self, ch: str):
        self._output_channel = ch

    def onMaWindowChanged(self, value: str):
        self.__class__._vfom_ma.window = int(value)

    def onOnSlicerChanged(self, value: list):
        self._on_slicer = slice(*value)

    def onOffSlicerChanged(self, value: list):
        self._off_slicer = slice(*value)

    def onPoiIndexChanged(self, value: int):
        self._poi_index = value

    def onDarkSlicerChanged(self, value: list):
        self._dark_slicer = slice(*value)

    def onDarkPoiIndexChanged(self, value: int):
        self._dark_poi_index = value

    def sources(self):
        """Override."""
        return [
            (self._output_channel, self._ppt, 1),
        ]

    @profiler("Gotthard Processor (pump-probe)")
    def process(self, data):
        """Override."""
        data, meta = data["raw"], data["meta"]
        tid = self.getTrainId(meta)

        raw = self.getPropertyData(data, self._output_channel, self._ppt)

        # check data shape
        if raw.ndim != 2:
            raise ProcessingError(f"Gotthard data must be a 2D array: "
                                  f"actual {raw.ndim}D")

        raw = raw.astype(_PIXEL_DTYPE)

        # ------------
        # process data
        # ------------

        # Note: we do not check whether on/off/dark share a same pulse index

        # update the moving average of corrected data
        dark = raw[self._dark_slicer]
        # check dark POI index
        if self._dark_poi_index >= len(dark):
            raise ProcessingError(f"Dark POI index {self._dark_poi_index} out "
                                  f"of boundary [{0} - {len(dark) - 1}]")
        dark_mean = np.mean(dark, axis=0)
        corrected = raw - dark_mean

        # calculate figure-of-merit for the current train
        on, off = corrected[self._on_slicer], corrected[self._off_slicer]

        if len(on) != len(off):
            raise ProcessingError(f"Number of on and off pulses are different: "
                                  f"{len(on)} and {len(off)}")

        # check POI index
        if self._poi_index >= len(on):
            raise ProcessingError(f"POI index {self._poi_index} out of "
                                  f"boundary [{0} - {len(on) - 1}]")

        # TODO: switch among several VFOM definitions
        vfom = on - off
        vfom_mean = np.mean(vfom, axis=0)

        self._vfom_ma = vfom
        vfom_ma = self._vfom_ma
        vfom_ma_mean = np.mean(vfom_ma, axis=0)

        self.log.info(f"Train {tid} processed")

        return {
            # raw and corrected spectra
            "raw": raw,
            "corrected": corrected,
            # slicers
            "on_slicer": self._on_slicer,
            "off_slicer": self._off_slicer,
            "dark_slicer": self._dark_slicer,
            # pulses of interest
            "poi_index": self._poi_index,
            "dark_poi_index": self._dark_poi_index,
            # VFOM for the current train
            "vfom": vfom,
            # Moving averaged of vfom
            "vfom_ma": vfom_ma,
            # average of vfom over pulses
            "vfom_mean": vfom_mean,
            # average of vfom_ma over pulses
            "vfom_ma_mean": vfom_ma_mean,
        }

    def reset(self):
        """Override."""
        del self._vfom_ma


class GotthardPumpProbeCtrlWidget(_BaseAnalysisCtrlWidgetS):
    """Gotthard pump-probe analysis control widget."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.output_ch_le = SmartStringLineEdit(
            GOTTHARD_DEVICE.get(self.topic, "Gotthard:output"))

        self.ma_window_le = SmartLineEdit("1")
        validator = QIntValidator()
        validator.setBottom(1)
        self.ma_window_le.setValidator(validator)

        self.on_slicer_le = SmartSliceLineEdit("0:50:2")
        self.off_slicer_le = SmartSliceLineEdit("1:50:2")
        # Actual, POI index should be within on-pulse indices
        self.poi_index_le = SmartLineEdit("0")
        self.poi_index_le.setValidator(
            QIntValidator(0, _MAX_N_GOTTHARD_PULSES - 1))

        self.dark_slicer_le = SmartSliceLineEdit("100:120")
        self.dark_poi_index_le = SmartLineEdit("0")
        self.dark_poi_index_le.setValidator(
            QIntValidator(0, _MAX_N_GOTTHARD_PULSES - 1))

        self._non_reconfigurable_widgets = [
            self.output_ch_le
        ]

        self.initUI()
        self.initConnections()

    def initUI(self):
        """Override."""
        layout = self.layout()

        layout.addRow("Output channel: ", self.output_ch_le)
        layout.addRow("M.A. window: ", self.ma_window_le)
        layout.addRow("On-pulse slicer: ", self.on_slicer_le)
        layout.addRow("Off-pulse slicer: ", self.off_slicer_le)
        layout.addRow("Pump-probe P.O.I.: ", self.poi_index_le)
        layout.addRow("Dark-pulse slicer: ", self.dark_slicer_le)
        layout.addRow("Dark P.O.I.: ", self.dark_poi_index_le)

    def initConnections(self):
        """Override."""
        pass


class GotthardPumpProbeFomMeanPlot(PlotWidgetF):
    """GotthardPumpProbeFomMeanPlot class.

    Visualize averaged VFOM over a train as well as its moving average.
    """
    def __init__(self, *, parent=None):
        super().__init__(parent=parent)

        self.setTitle("Averaged FOM over train")
        self.setLabel('left', "ADU")
        self.setLabel('bottom', "Pixel")
        self.addLegend(offset=(10, 5))

        self._mean = self.plotCurve(name="Current", pen=FoamColor.mkPen("p"))
        self._mean_ma = self.plotCurve(name="Moving average",
                                       pen=FoamColor.mkPen("g"))

    def updateF(self, data):
        """Override."""
        vfom_mean, vfom_ma_mean = data['vfom_mean'], data['vfom_ma_mean']
        x = np.arange(len(vfom_mean))
        self._mean.setData(x, vfom_mean)
        self._mean_ma.setData(x, vfom_ma_mean)


class GotthardPumpProbeFomPulsePlot(PlotWidgetF):
    """GotthardPumpProbeFomPulsePlot class.

    Visualize VFOM of a single pump-probe pulse as well as its moving average.
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self._idx = 0

        self._updateTitle()
        self.setLabel('left', "ADU")
        self.setLabel('bottom', "Pixel")
        self.addLegend(offset=(10, 5))

        self._poi = self.plotCurve(name="Current", pen=FoamColor.mkPen("p"))
        self._poi_ma = self.plotCurve(name=f"Moving average",
                                      pen=FoamColor.mkPen("g"))

    def _updateTitle(self):
        self.setTitle(f"Pulse of interest (FOM): {self._idx}")

    def updateF(self, data):
        """Override."""
        idx = data['poi_index']
        if idx != self._idx:
            self._idx = idx
            self._updateTitle()

        vfom, vfom_ma = data['vfom'][idx], data['vfom_ma'][idx]
        x = np.arange(len(vfom))
        self._poi.setData(x, vfom)
        self._poi_ma.setData(x, vfom_ma)


class GotthardPumpProbeRawPulsePlot(PlotWidgetF):
    """GotthardPumpProbeRawPulsePlot class.

    Visualize raw data a pair of on/off pulses.
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self._idx = 0

        self._updateTitle()
        self.setLabel('left', "ADU")
        self.setLabel('bottom', "Pixel")
        self.addLegend(offset=(10, 5))

        self._on = self.plotCurve(name="On", pen=FoamColor.mkPen("r"))
        self._off = self.plotCurve(name="Off", pen=FoamColor.mkPen("b"))

    def _updateTitle(self):
        self.setTitle(f"Pulse of interest (raw): {self._idx}")

    def updateF(self, data):
        """Override."""
        idx = data['poi_index']
        if idx != self._idx:
            self._idx = idx
            self._updateTitle()

        on = data['raw'][data['on_slicer']][idx]
        off = data['raw'][data['off_slicer']][idx]
        x = np.arange(len(on))
        self._on.setData(x, on)
        self._off.setData(x, off)


class GotthardPumpProbeDarkPulsePlot(PlotWidgetF):
    """GotthardPumpProbeDarkPulsePlot class.

    Visualize raw data of a dark pulse.
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self._idx = 0

        self._updateTitle()
        self.setLabel('left', "ADU")
        self.setLabel('bottom', "Pixel")

        self._plot = self.plotCurve(pen=FoamColor.mkPen("k"))

    def _updateTitle(self):
        self.setTitle(f"Pulse of interest (dark): {self._idx}")

    def updateF(self, data):
        """Override."""
        idx = data['dark_poi_index']
        if idx != self._idx:
            self._idx = idx
            self._updateTitle()

        y = data['raw'][data['dark_slicer']][idx]
        x = np.arange(len(y))
        self._plot.setData(x, y)


class GotthardPumpProbeImageView(ImageViewF):
    """GotthardPumpProbeImageView class.

    Visualize the heatmap of pulse-resolved Gotthard data in a train
    after dark subtraction.
    """
    def __init__(self, *, parent=None):
        super().__init__(parent=parent)

        self.setAspectLocked(False)

        self.setTitle('ADU heatmap (corrected)')
        self.setLabel('left', "Pulse index")
        self.setLabel('bottom', "Pixel")

    def updateF(self, data):
        """Override."""
        self.setImage(data['corrected'])


@create_special(GotthardPumpProbeCtrlWidget,
                GotthardPumpProbeProcessor,
                QThreadKbClient)
class GotthardPumpProbe(_FoamLightApp):
    """Gotthard pump-probe application."""

    icon = "Gotthard_pump_probe.png"
    _title = "Gotthard (pump-probe)"
    _long_title = "Gotthard pump-probe analysis"

    def __init__(self, topic):
        super().__init__(topic, with_dark=False)

        self._fom_poi_plot = GotthardPumpProbeFomPulsePlot(parent=self)
        self._fom_mean_plot = GotthardPumpProbeFomMeanPlot(parent=self)
        self._heatmap = GotthardPumpProbeImageView(parent=self)
        self._raw_poi_plot = GotthardPumpProbeRawPulsePlot(parent=self)
        self._dark_poi_plot = GotthardPumpProbeDarkPulsePlot(parent=self)

        self.initUI()
        self.initConnections()
        self.startWorker()

    def initUI(self):
        """Override."""
        right_panel = QSplitter(Qt.Vertical)
        right_up_panel = QSplitter()
        right_up_panel.addWidget(self._fom_poi_plot)
        right_up_panel.addWidget(self._fom_mean_plot)

        right_down_panel = QSplitter(Qt.Horizontal)
        sub_right_down_panel = QSplitter(Qt.Vertical)
        sub_right_down_panel.addWidget(self._raw_poi_plot)
        sub_right_down_panel.addWidget(self._dark_poi_plot)
        right_down_panel.addWidget(self._heatmap)
        right_down_panel.addWidget(sub_right_down_panel)
        right_down_panel.setSizes([self._TOTAL_W / 3, self._TOTAL_W / 3])

        right_panel.addWidget(right_up_panel)
        right_panel.addWidget(right_down_panel)
        right_panel.setSizes([self._TOTAL_H / 2, self._TOTAL_H / 2])

        cw = self.centralWidget()
        cw.addWidget(right_panel)
        cw.setSizes([self._TOTAL_W / 3, 2 * self._TOTAL_W / 3])

        self.resize(self._TOTAL_W, self._TOTAL_H)

    def initConnections(self):
        """Override."""
        self._ctrl_widget_st.output_ch_le.value_changed_sgn.connect(
            self._worker_st.onOutputChannelChanged)
        self._ctrl_widget_st.output_ch_le.returnPressed.emit()

        self._ctrl_widget_st.on_slicer_le.value_changed_sgn.connect(
            self._worker_st.onOnSlicerChanged)
        self._ctrl_widget_st.on_slicer_le.returnPressed.emit()

        self._ctrl_widget_st.off_slicer_le.value_changed_sgn.connect(
            self._worker_st.onOffSlicerChanged)
        self._ctrl_widget_st.off_slicer_le.returnPressed.emit()

        self._ctrl_widget_st.poi_index_le.value_changed_sgn.connect(
            lambda x: self._worker_st.onPoiIndexChanged(int(x)))
        self._ctrl_widget_st.poi_index_le.returnPressed.emit()

        self._ctrl_widget_st.dark_slicer_le.value_changed_sgn.connect(
            self._worker_st.onDarkSlicerChanged)
        self._ctrl_widget_st.dark_slicer_le.returnPressed.emit()

        self._ctrl_widget_st.dark_poi_index_le.value_changed_sgn.connect(
            lambda x: self._worker_st.onDarkPoiIndexChanged(int(x)))
        self._ctrl_widget_st.dark_poi_index_le.returnPressed.emit()

        self._ctrl_widget_st.ma_window_le.value_changed_sgn.connect(
            self._worker_st.onMaWindowChanged)
        self._ctrl_widget_st.ma_window_le.returnPressed.emit()
