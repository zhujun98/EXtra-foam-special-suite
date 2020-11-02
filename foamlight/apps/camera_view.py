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
    HistWidgetF, ImageViewF, SmartBoundaryLineEdit, SmartLineEdit,
    SmartStringLineEdit
)
from pyfoamalgo import hist_with_stats, MovingAverageArray

from ..config import _IMAGE_DTYPE, _MAX_INT32
from ..core import (
    create_app, profiler, QThreadKbClient, QThreadWorker,
    _BaseAnalysisCtrlWidgetS, _FoamLightApp
)

_DEFAULT_N_BINS = 10
_DEFAULT_BIN_RANGE = "-inf, inf"
_MAX_N_BINS = 999
# a non-empty place holder
_DEFAULT_OUTPUT_CHANNEL = "camera:output"
# default is for Basler camera
_DEFAULT_PROPERTY = "data.image.data"


class CameraViewProcessor(QThreadWorker):
    """Camera view processor.

    Attributes:
        _output_channel (str): output channel name.
        _ppt (str): property name.
        _raw_ma (numpy.ndarray): moving average of the raw image data.
            Shape=(y, x)
        _dark_ma (numpy.ndarray): moving average of the dark data.
            Shape=(pulses, pixels)
        _bin_range (tuple): range of the ROI histogram.
        _n_bins (int): number of bins of the ROI histogram.
    """

    _raw_ma = MovingAverageArray()
    _dark_ma = MovingAverageArray(_MAX_INT32)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._output_channel = ''
        self._ppt = ''

        self.__class__._raw_ma.window = 1

        self._bin_range = self.str2range(_DEFAULT_BIN_RANGE)
        self._n_bins = _DEFAULT_N_BINS

        del self._dark_ma

    def onOutputChannelChanged(self, value: str):
        self._output_channel = value

    def onPropertyChanged(self, value: str):
        self._ppt = value

    def onMaWindowChanged(self, value: str):
        self.__class__._raw_ma.window = int(value)

    def onRemoveDark(self):
        """Override."""
        del self._dark_ma

    def onBinRangeChanged(self, value: tuple):
        self._bin_range = value

    def onNoBinsChanged(self, value: str):
        self._n_bins = int(value)

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
                self._dark_ma = np.mean(arr.values, axis=0, dtype=_IMAGE_DTYPE)
            except Exception as e:
                self.log.error(f"Unexpect exception when getting data array: "
                               f"{repr(e)}")

    def sources(self):
        """Override."""
        return [
            (self._output_channel, self._ppt, 1),
        ]

    @profiler("Camera view processor")
    def process(self, data):
        """Override."""
        data, meta = data["raw"], data["meta"]

        tid = self.getTrainId(meta)

        img = self.squeezeToImage(
            tid, self.getPropertyData(data, self._output_channel, self._ppt))
        if img is None:
            return

        if self.recordingDark():
            self._dark_ma = img
            displayed = self._dark_ma
        else:
            self._raw_ma = img
            displayed = self._raw_ma
            if self.subtractDark() and self._dark_ma is not None:
                # caveat: cannot subtract inplace
                displayed = displayed - self._dark_ma

        self.log.info(f"Train {tid} processed")

        return {
            "displayed": displayed,
            "roi_hist": hist_with_stats(self.getRoiData(displayed),
                                        self._bin_range, self._n_bins),
        }

    def reset(self):
        """Override."""
        del self._raw_ma
        del self._dark_ma


class CamViewCtrlWidget(_BaseAnalysisCtrlWidgetS):
    """Camera view control widget."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.output_ch_le = SmartStringLineEdit(_DEFAULT_OUTPUT_CHANNEL)
        self.property_le = SmartStringLineEdit(_DEFAULT_PROPERTY)

        self.ma_window_le = SmartLineEdit("1")
        validator = QIntValidator()
        validator.setBottom(1)
        self.ma_window_le.setValidator(validator)

        self.bin_range_le = SmartBoundaryLineEdit(_DEFAULT_BIN_RANGE)
        self.n_bins_le = SmartLineEdit(str(_DEFAULT_N_BINS))
        self.n_bins_le.setValidator(QIntValidator(1, _MAX_N_BINS))

        self._non_reconfigurable_widgets = [
            self.output_ch_le,
            self.property_le,
        ]

        self.initUI()
        self.initConnections()

    def initUI(self):
        """Override."""
        layout = self.layout()

        layout.addRow("Output channel: ", self.output_ch_le)
        layout.addRow("Property: ", self.property_le)
        layout.addRow("M.A. window: ", self.ma_window_le)
        layout.addRow("Bin range: ", self.bin_range_le)
        layout.addRow("# of bins: ", self.n_bins_le)

    def initConnections(self):
        """Override."""
        pass


class CameraViewImage(ImageViewF):
    """CameraViewImage class.

    Visualize the camera image.
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(has_roi=True, parent=parent)

    def updateF(self, data):
        """Override."""
        self.setImage(data['displayed'])


class CameraViewRoiHist(HistWidgetF):
    """CameraViewRoiHist class

    Visualize the ROI histogram.
    """
    def __init__(self, *, parent=None):
        super().__init__(parent=parent)

        self.setLabel('left', 'Occurence')
        self.setLabel('bottom', 'Pixel value')

    def updateF(self, data):
        """Override."""
        hist, bin_centers, mean, median, std = data['roi_hist']
        if bin_centers is None:
            self.reset()
        else:
            self._plot.setData(bin_centers, hist)
            self.updateTitle(mean, median, std)


@create_app(CamViewCtrlWidget,
            CameraViewProcessor,
            QThreadKbClient)
class CameraView(_FoamLightApp):
    """Camera view application."""

    icon = "camera_view.png"
    _title = "Camera view"
    _long_title = "Camera view"

    def __init__(self, topic):
        super().__init__(topic)

        self._view = CameraViewImage(parent=self)
        self._roi_hist = CameraViewRoiHist(parent=self)

        self.initUI()
        self.initConnections()
        self.startWorker()

    def initUI(self):
        """Override."""
        right_panel = QSplitter(Qt.Vertical)
        right_panel.addWidget(self._view)
        right_panel.addWidget(self._roi_hist)
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

        self._ctrl_widget_st.property_le.value_changed_sgn.connect(
            self._worker_st.onPropertyChanged)
        self._ctrl_widget_st.property_le.returnPressed.emit()

        self._ctrl_widget_st.ma_window_le.value_changed_sgn.connect(
            self._worker_st.onMaWindowChanged)
        self._ctrl_widget_st.ma_window_le.returnPressed.emit()

        self._ctrl_widget_st.bin_range_le.value_changed_sgn.connect(
            self._worker_st.onBinRangeChanged)
        self._ctrl_widget_st.bin_range_le.returnPressed.emit()

        self._ctrl_widget_st.n_bins_le.value_changed_sgn.connect(
            self._worker_st.onNoBinsChanged)
        self._ctrl_widget_st.n_bins_le.returnPressed.emit()
