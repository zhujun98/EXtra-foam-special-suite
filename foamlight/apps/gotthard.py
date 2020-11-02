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
from PyQt5.QtGui import QDoubleValidator, QIntValidator
from PyQt5.QtWidgets import QCheckBox, QSplitter

from foamgraph import (
    FoamColor, HistWidgetF, ImageViewF, PlotWidgetF, SmartBoundaryLineEdit,
    SmartLineEdit, SmartSliceLineEdit, SmartStringLineEdit
)
from pyfoamalgo import hist_with_stats, MovingAverageArray

from ..config import (
    _MAX_INT32, _PIXEL_DTYPE, _MAX_N_GOTTHARD_PULSES, GOTTHARD_DEVICE
)
from ..core import (
    create_app, profiler, QThreadKbClient, QThreadWorker,
    _BaseAnalysisCtrlWidgetS, _FoamLightApp
)
from ..exceptions import ProcessingError

_MAX_N_BINS = 999
_DEFAULT_N_BINS = 10
_DEFAULT_BIN_RANGE = "-inf, inf"


class GotthardProcessor(QThreadWorker):
    """Gotthard analysis processor.

    Attributes:
        _output_channel (str): output channel name.
        _pulse_slicer (slice): a slicer used to slice pulses in a train.
        _poi_index (int): index of the pulse of interest after slicing.
        _scale (float): scale of the x axis. If 0, it means no scale will
            be applied and the unit of x-axis is pixel. While a positive
            value means converting pixel to eV by multiplying this value
            for the x axis.
        _offset (float): offset of the x axis when the value of scale is
            not zero.
        _bin_range (tuple): range of the ADU histogram.
        _n_bins (int): number of bins of the ADU histogram.
        _hist_over_ma (bool): True for calculating the histogram over the
            moving averaged data. Otherwise, it is calculated over the
            current train.
        _raw_ma (numpy.ndarray): moving average of the raw data.
            Shape=(pulses, pixels)
        _dark_ma (numpy.ndarray): moving average of the dark data.
            Shape=(pulses, pixels)
        _dark_mean_ma (numpy.ndarray): average of pulses in a train of the
            moving average of the dark data. It is used for dark subtraction.
            Shape=(pixels,)
    """

    _raw_ma = MovingAverageArray()
    _dark_ma = MovingAverageArray(_MAX_INT32)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._output_channel = ""
        self._ppt = "data.adc"

        self._pulse_slicer = slice(None, None)
        self._poi_index = 0

        self._scale = 0
        self._offset = 0

        self._bin_range = self.str2range(_DEFAULT_BIN_RANGE)
        self._n_bins = _DEFAULT_N_BINS
        self._hist_over_ma = False

        del self._raw_ma

        del self._dark_ma
        self._dark_mean_ma = None

    def onOutputChannelChanged(self, ch: str):
        self._output_channel = ch

    def onMaWindowChanged(self, value: str):
        self.__class__._raw_ma.window = int(value)

    def onScaleChanged(self, value: str):
        self._scale = float(value)

    def onOffsetChanged(self, value: str):
        self._offset = float(value)

    def onBinRangeChanged(self, value: tuple):
        self._bin_range = value

    def onNoBinsChanged(self, value: str):
        self._n_bins = int(value)

    def onHistOverMaChanged(self, state: bool):
        self._hist_over_ma = state

    def onPulseSlicerChanged(self, value: list):
        self._pulse_slicer = slice(*value)
        dark_ma = self._dark_ma
        if dark_ma is not None:
            self._dark_mean_ma = np.mean(dark_ma[self._pulse_slicer], axis=0)

    def onPoiIndexChanged(self, value: int):
        self._poi_index = value

    def onLoadDarkRun(self, dirpath):
        """Override."""
        run = self._loadRunDirectoryST(dirpath)
        if run is not None:
            try:
                arr = run.get_array(self._output_channel, self._ppt)
                shape = arr.shape
                if arr.ndim != 3:
                    self.log.error(f"Data must be a 3D array! "
                                   f"Actual shape: {shape}")
                    return

                self.log.info(f"Found dark data with shape {shape}")
                # FIXME: performance
                self._dark_ma = np.mean(
                    arr.values, axis=0, dtype=_PIXEL_DTYPE)
                self._dark_mean_ma = np.mean(
                    self._dark_ma[self._pulse_slicer],
                    axis=0, dtype=_PIXEL_DTYPE)
            except Exception as e:
                self.log.error(f"Unexpect exception when getting data array: "
                               f"{repr(e)}")

    def onRemoveDark(self):
        """Override."""
        del self._dark_ma
        self._dark_mean_ma = None

    def sources(self):
        """Override."""
        return [
            (self._output_channel, self._ppt, 1),
        ]

    @profiler("Gotthard Processor")
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

        # check POI index
        max_idx = raw[self._pulse_slicer].shape[0]
        if self._poi_index >= max_idx:
            raise ProcessingError(f"POI index {self._poi_index} out of "
                                  f"boundary [{0} - {max_idx - 1}]")

        # ------------
        # process data
        # ------------

        if self.recordingDark():
            # update the moving average of dark data
            self._dark_ma = raw

            self._dark_mean_ma = np.mean(
                self._dark_ma[self._pulse_slicer], axis=0)

            # During dark recording, no offset correcttion is applied and
            # only dark data and its statistics are displayed.
            spectrum = raw[self._pulse_slicer]
            spectrum_ma = self._dark_ma[self._pulse_slicer]
        else:
            # update the moving average of raw data
            self._raw_ma = raw

            if self.subtractDark() and self._dark_mean_ma is not None:
                spectrum = raw[self._pulse_slicer] - self._dark_mean_ma
                spectrum_ma = self._raw_ma[self._pulse_slicer] - self._dark_mean_ma
            else:
                spectrum = raw[self._pulse_slicer]
                spectrum_ma = self._raw_ma[self._pulse_slicer]

        spectrum_mean = np.mean(spectrum, axis=0)
        spectrum_ma_mean = np.mean(spectrum_ma, axis=0)

        if self._scale == 0:
            x = None
        else:
            x = np.arange(len(spectrum_mean)) * self._scale - self._offset

        self.log.info(f"Train {tid} processed")

        return {
            # x axis of the spectrum
            "x": x,
            # spectrum for the current train
            "spectrum": spectrum,
            # moving average of spectrum
            "spectrum_ma": spectrum_ma,
            # average of the spectrum for the current train over pulses
            "spectrum_mean": spectrum_mean,
            # moving average of spectrum_mean
            "spectrum_ma_mean": spectrum_ma_mean,
            # index of pulse of interest
            "poi_index": self._poi_index,
            # hist, bin_centers, mean, median, std
            "hist": hist_with_stats(
                self.getRoiData(spectrum_ma) if self._hist_over_ma else
                self.getRoiData(spectrum),
                self._bin_range, self._n_bins)
        }

    def reset(self):
        """Override."""
        del self._raw_ma
        del self._dark_ma


class GotthardCtrlWidget(_BaseAnalysisCtrlWidgetS):
    """Gotthard analysis control widget."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.output_ch_le = SmartStringLineEdit(
            GOTTHARD_DEVICE.get(self.topic, "Gotthard:output"))

        self.ma_window_le = SmartLineEdit("1")
        validator = QIntValidator()
        validator.setBottom(1)
        self.ma_window_le.setValidator(validator)

        self.pulse_slicer_le = SmartSliceLineEdit(":")

        self.poi_index_le = SmartLineEdit("0")
        self.poi_index_le.setValidator(
            QIntValidator(0, _MAX_N_GOTTHARD_PULSES - 1))

        self.bin_range_le = SmartBoundaryLineEdit(_DEFAULT_BIN_RANGE)
        self.n_bins_le = SmartLineEdit(str(_DEFAULT_N_BINS))
        self.n_bins_le.setValidator(QIntValidator(1, _MAX_N_BINS))
        self.hist_over_ma_cb = QCheckBox("Histogram over M.A. train")

        self.scale_le = SmartLineEdit("0")
        validator = QDoubleValidator()
        validator.setBottom(0)
        self.scale_le.setValidator(validator)
        self.offset_le = SmartLineEdit("0")
        self.offset_le.setValidator(QDoubleValidator())

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
        layout.addRow("Pulse slicer: ", self.pulse_slicer_le)
        layout.addRow("P.O.I. (sliced): ", self.poi_index_le)
        layout.addRow("Bin range: ", self.bin_range_le)
        layout.addRow("# of bins: ", self.n_bins_le)
        layout.addRow("Scale (eV/pixel): ", self.scale_le)
        layout.addRow("Offset (eV): ", self.offset_le)
        layout.addRow("", self.hist_over_ma_cb)

    def initConnections(self):
        """Override."""
        pass


class GotthardAvgPlot(PlotWidgetF):
    """GotthardAvgPlot class.

    Visualize signals of the averaged pulse over a train as well as its
    moving average.
    """
    def __init__(self, *, parent=None):
        super().__init__(parent=parent)

        self.setLabel('left', "ADU")
        self.setLabel('bottom', "Pixel")
        self.addLegend(offset=(5, 10))

        self.setTitle("Averaged spectra over pulses")
        self._mean = self.plotCurve(name="Current", pen=FoamColor.mkPen("p"))
        self._mean_ma = self.plotCurve(name="Moving average",
                                       pen=FoamColor.mkPen("g"))

    def updateF(self, data):
        """Override."""
        spectrum = data['spectrum_mean']
        spectrum_ma = data['spectrum_ma_mean']

        x = data["x"]
        if x is None:
            self.setLabel('bottom', "Pixel")
            x = np.arange(len(spectrum))
        else:
            self.setLabel('bottom', "eV")

        self._mean.setData(x, spectrum)
        self._mean_ma.setData(x, spectrum_ma)


class GotthardPulsePlot(PlotWidgetF):
    """GotthardPulsePlot class.

    Visualize signals of a single pulse as well as its moving average.
    """
    def __init__(self, *, parent=None):
        super().__init__(parent=parent)

        self._idx = 0

        self._updateTitle()
        self.setLabel('left', "ADU")
        self.setLabel('bottom', "Pixel")
        self.addLegend(offset=(5, 10))

        self._poi = self.plotCurve(name="Current", pen=FoamColor.mkPen("p"))
        self._poi_ma = self.plotCurve(name="Moving average",
                                      pen=FoamColor.mkPen("g"))

    def _updateTitle(self):
        self.setTitle(f"Pulse of interest: {self._idx}")

    def updateF(self, data):
        """Override."""
        idx = data['poi_index']
        if idx != self._idx:
            self._idx = idx
            self._updateTitle()

        spectrum = data['spectrum'][idx]
        spectrum_ma = data['spectrum_ma'][idx]

        x = data["x"]
        if x is None:
            self.setLabel('bottom', "Pixel")
            x = np.arange(len(spectrum))
        else:
            self.setLabel('bottom', "eV")

        self._poi.setData(x, spectrum)
        self._poi_ma.setData(x, spectrum_ma)


class GotthardImageView(ImageViewF):
    """GotthardImageView class.

    Visualize the heatmap of pulse-resolved Gotthard data in a train.
    """
    def __init__(self, *, parent=None):
        super().__init__(has_roi=True, roi_size=(100, 10), parent=parent)

        self.setAspectLocked(False)

        self.setTitle('ADU heatmap')
        self.setLabel('left', "Pulse index (sliced)")
        self.setLabel('bottom', "Pixel")

    def updateF(self, data):
        """Override."""
        self.setImage(data['spectrum'])


class GotthardHist(HistWidgetF):
    """GotthardHist class

    Visualize the ADU histogram in a train.
    """
    def __init__(self, *, parent=None):
        super().__init__(parent=parent)

        self.setLabel('left', 'Occurence')
        self.setLabel('bottom', 'ADU')

    def updateF(self, data):
        """Override."""
        hist, bin_centers, mean, median, std = data['hist']
        if bin_centers is None:
            self.reset()
        else:
            self._plot.setData(bin_centers, hist)
            self.updateTitle(mean, median, std)


@create_app(GotthardCtrlWidget,
            GotthardProcessor,
            QThreadKbClient)
class Gotthard(_FoamLightApp):
    """Gotthard application."""

    icon = "Gotthard.png"
    _title = "Gotthard"
    _long_title = "Gotthard analysis"

    def __init__(self, topic):
        super().__init__(topic)

        self._poi_plots = GotthardPulsePlot(parent=self)
        self._mean_plots = GotthardAvgPlot(parent=self)
        self._heatmap = GotthardImageView(parent=self)
        self._hist = GotthardHist(parent=self)

        self.initUI()
        self.initConnections()
        self.startWorker()

    def initUI(self):
        """Override."""
        middle_panel = QSplitter(Qt.Vertical)
        middle_panel.addWidget(self._poi_plots)
        middle_panel.addWidget(self._mean_plots)

        right_panel = QSplitter(Qt.Vertical)
        right_panel.addWidget(self._hist)
        right_panel.addWidget(self._heatmap)
        right_panel.setSizes([self._TOTAL_H / 2, self._TOTAL_H / 2])

        cw = self.centralWidget()
        cw.addWidget(middle_panel)
        cw.addWidget(right_panel)
        cw.setSizes([self._TOTAL_W / 3, self._TOTAL_W / 3, self._TOTAL_W / 3])

        self.resize(self._TOTAL_W, self._TOTAL_H)

    def initConnections(self):
        """Override."""
        self._ctrl_widget_st.output_ch_le.value_changed_sgn.connect(
            self._worker_st.onOutputChannelChanged)
        self._ctrl_widget_st.output_ch_le.returnPressed.emit()

        self._ctrl_widget_st.poi_index_le.value_changed_sgn.connect(
            lambda x: self._worker_st.onPoiIndexChanged(int(x)))
        self._ctrl_widget_st.poi_index_le.returnPressed.emit()

        self._ctrl_widget_st.pulse_slicer_le.value_changed_sgn.connect(
            self._worker_st.onPulseSlicerChanged)
        self._ctrl_widget_st.pulse_slicer_le.returnPressed.emit()

        self._ctrl_widget_st.ma_window_le.value_changed_sgn.connect(
            self._worker_st.onMaWindowChanged)
        self._ctrl_widget_st.ma_window_le.returnPressed.emit()

        self._ctrl_widget_st.scale_le.value_changed_sgn.connect(
            self._worker_st.onScaleChanged)
        self._ctrl_widget_st.scale_le.returnPressed.emit()

        self._ctrl_widget_st.offset_le.value_changed_sgn.connect(
            self._worker_st.onOffsetChanged)
        self._ctrl_widget_st.offset_le.returnPressed.emit()

        self._ctrl_widget_st.bin_range_le.value_changed_sgn.connect(
            self._worker_st.onBinRangeChanged)
        self._ctrl_widget_st.bin_range_le.returnPressed.emit()

        self._ctrl_widget_st.n_bins_le.value_changed_sgn.connect(
            self._worker_st.onNoBinsChanged)
        self._ctrl_widget_st.n_bins_le.returnPressed.emit()

        self._ctrl_widget_st.hist_over_ma_cb.toggled.connect(
            self._worker_st.onHistOverMaChanged)
        self._ctrl_widget_st.hist_over_ma_cb.toggled.emit(
            self._ctrl_widget_st.hist_over_ma_cb.isChecked())
