"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file BSD_LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.

Modified from https://github.com/European-XFEL/EXtra-foam, version 1.0.0
Copyright (C) Jun Zhu
"""
import math

import numpy as np
from scipy import stats

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import QPushButton, QSplitter

from foamgraph import (
    FoamColor, ImageViewF, SmartBoundaryLineEdit, SmartLineEdit,
    SmartStringLineEdit, TimedImageViewF, TimedPlotWidgetF,
)
from pyfoamalgo import compute_spectrum_1d, nansum, SimpleSequence

from ..core import (
    create_app, profiler, QThreadWorker, QThreadFoamClient,
    _BaseAnalysisCtrlWidgetS, _FoamLightApp
)
from ..exceptions import ProcessingError

_MAX_N_BINS = 999
_DEFAULT_N_BINS = 20
_DEFAULT_BIN_RANGE = "-inf, inf"


class TrXasProcessor(QThreadWorker):
    """Time-resolved XAS processor.

    The implementation of tr-XAS processor is easier than bin processor
    since it cannot have empty device ID or property. Moreover, it does
    not include VFOM heatmap.

    Absorption ROI-i/ROI-j is defined as -log(sum(ROI-i)/sum(ROI-j)).

    Attributes:
        _device_id1 (str): device ID 1.
        _ppt1 (str): property of device 1.
        _device_id2 (str): device ID 2.
        _ppt2 (str): property of device 2.
        _slow1 (SimpleSequence): store train-resolved data of source 1.
        _slow2 (SimpleSequence): store train-resolved data of source 2.
        _a13 (SimpleSequence): store train-resolved absorption ROI1/ROI3.
        _a23 (SimpleSequence): store train-resolved absorption ROI2/ROI3.
        _a21 (SimpleSequence): store train-resolved absorption ROI2/ROI1.
        _edges1 (numpy.array): edges of bin 1. shape = (_n_bins1 + 1,)
        _counts1 (numpy.array): counts of bin 1. shape = (_n_bins1,)
        _a13_stats (numpy.array): 1D binning of absorption ROI1/ROI3 with
            respect to source 1.
        _a23_stats (numpy.array): 1D binning of absorption ROI2/ROI3 with
            respect to source 1.
        _a21_stats (numpy.array): 1D binning of absorption ROI2/ROI1 with
            respect to source 1.
        _edges2 (numpy.array): edges of bin 2. shape = (_n_bins2 + 1,)
        _a21_heat (numpy.array): 2D binning of absorption ROI2/ROI1.
            shape = (_n_bins2, _n_bins1)
        _a21_heat_count (numpy.array): counts of 2D binning of absorption
            ROI2/ROI1. shape = (_n_bins2, _n_bins1)
        _bin_range1 (tuple): bin 1 range requested.
        _actual_range1 (tuple): actual bin range used in bin 1.
        _n_bins1 (int): number of bins of bin 1.
        _bin_range2 (tuple): bin 2 range requested.
        _actual_range2 (tuple): actual bin range used in bin 2.
        _n_bins2 (int): number of bins of bin 2.
        _bin1d (bool): a flag indicates whether data need to be re-binned
            with respect to source 1.
        _bin2d (bool): a flag indicates whether data need to be re-binned
            with respect to both source 1 and source 2.
        _reset (bool): True for clearing all the existing data.
    """

    # 10 pulses/train * 60 seconds * 30 minutes = 18,000
    _MAX_POINTS = 100 * 60 * 60

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._device_id1 = ""
        self._ppt1 = ""
        self._device_id2 = ""
        self._ppt2 = ""

        self._slow1 = SimpleSequence(max_len=self._MAX_POINTS)
        self._slow2 = SimpleSequence(max_len=self._MAX_POINTS)
        self._a13 = SimpleSequence(max_len=self._MAX_POINTS)
        self._a23 = SimpleSequence(max_len=self._MAX_POINTS)
        self._a21 = SimpleSequence(max_len=self._MAX_POINTS)

        self._edges1 = None
        self._counts1 = None
        self._a13_stats = None
        self._a23_stats = None
        self._a21_stats = None

        self._edges2 = None
        self._a21_heat = None
        self._a21_heat_count = None

        self._bin_range1 = self.str2range(_DEFAULT_BIN_RANGE)
        self._actual_range1 = None
        self._auto_range1 = [True, True]
        self._n_bins1 = _DEFAULT_N_BINS
        self._bin_range2 = self.str2range(_DEFAULT_BIN_RANGE)
        self._actual_range2 = None
        self._auto_range2 = [True, True]
        self._n_bins2 = _DEFAULT_N_BINS

        self._bin1d = True
        self._bin2d = True

    def onDeviceId1Changed(self, value: str):
        self._device_id1 = value

    def onProperty1Changed(self, value: str):
        self._ppt1 = value

    def onDeviceId2Changed(self, value: str):
        self._device_id2 = value

    def onProperty2Changed(self, value: str):
        self._ppt2 = value

    def onNBins1Changed(self, value: str):
        n_bins = int(value)
        if n_bins != self._n_bins1:
            self._n_bins1 = n_bins
            self._bin1d = True
            self._bin2d = True

    def onBinRange1Changed(self, value: tuple):
        if value != self._bin_range1:
            self._bin_range1 = value
            self._auto_range1[:] = [math.isinf(v) for v in value]
            self._bin1d = True
            self._bin2d = True

    def onNBins2Changed(self, value: str):
        n_bins = int(value)
        if n_bins != self._n_bins2:
            self._n_bins2 = n_bins
            self._bin2d = True

    def onBinRange2Changed(self, value: tuple):
        if value != self._bin_range2:
            self._bin_range2 = value
            self._auto_range2[:] = [math.isinf(v) for v in value]

    def sources(self):
        """Override."""
        return [
            (self._device_id1, self._ppt1, 0),
            (self._device_id2, self._ppt2, 0),
        ]

    @profiler("tr-XAS Processor")
    def process(self, data):
        """Override."""
        processed = data["processed"]

        roi1, roi2, roi3 = None, None, None
        a13, a23, a21, s1, s2 = None, None, None, None, None
        try:
            roi1, roi2, roi3, a13, a23, a21, s1, s2 = \
                self._update_data_point(processed, data['raw'])
        except ProcessingError as e:
            self.log.error(repr(e))

        actual_range1 = self.get_actual_range(
            self._slow1.data(), self._bin_range1, self._auto_range1)
        if actual_range1 != self._actual_range1:
            self._actual_range1 = actual_range1
            self._bin1d = True
            self._bin2d = True

        if self._bin1d:
            self._new_1d_binning()
            self._bin1d = False
        else:
            if a21 is not None:
                self._update_1d_binning(a13, a23, a21, s1)

        actual_range2 = self.get_actual_range(
            self._slow2.data(), self._bin_range2, self._auto_range2)
        if actual_range2 != self._actual_range2:
            self._actual_range2 = actual_range2
            self._bin2d = True

        if self._bin2d:
            self._new_2d_binning()
            self._bin2d = False
        else:
            if a21 is not None:
                self._update_2d_binning(a21, s1, s2)

        self.log.info(f"Train {processed.tid} processed")

        return {
            "roi1": roi1,
            "roi2": roi2,
            "roi3": roi3,
            "centers1": self.edges2centers(self._edges1)[0],
            "counts1": self._counts1,
            "centers2": self.edges2centers(self._edges2)[0],
            "a13_stats": self._a13_stats,
            "a23_stats": self._a23_stats,
            "a21_stats": self._a21_stats,
            "a21_heat": self._a21_heat,
            "a21_heat_count": self._a21_heat_count
        }

    def _update_data_point(self, processed, raw):
        roi = processed.roi
        masked = processed.image.masked_mean

        # get three ROIs
        roi1 = roi.geom1.rect(masked)
        if roi1 is None:
            raise ProcessingError("ROI1 is not available!")
        roi2 = roi.geom2.rect(masked)
        if roi2 is None:
            raise ProcessingError("ROI2 is not available!")
        roi3 = roi.geom3.rect(masked)
        if roi3 is None:
            raise ProcessingError("ROI3 is not available!")

        # get sums of the three ROIs
        sum1 = nansum(roi1)
        if sum1 <= 0:
            raise ProcessingError("ROI1 sum <= 0!")
        sum2 = nansum(roi2)
        if sum2 <= 0:
            raise ProcessingError("ROI2 sum <= 0!")
        sum3 = nansum(roi3)
        if sum3 <= 0:
            raise ProcessingError("ROI3 sum <= 0!")

        # calculate absorptions
        a13 = -np.log(sum1 / sum3)
        a23 = -np.log(sum2 / sum3)
        a21 = -np.log(sum2 / sum1)

        # update historic data
        self._a13.append(a13)
        self._a23.append(a23)
        self._a21.append(a21)

        # fetch slow data
        s1 = self.getPropertyData(raw, self._device_id1, self._ppt1)
        self._slow1.append(s1)
        s2 = self.getPropertyData(raw, self._device_id2, self._ppt2)
        self._slow2.append(s2)

        return roi1, roi2, roi3, a13, a23, a21, s1, s2

    def _new_1d_binning(self):
        self._a13_stats, _, _ = compute_spectrum_1d(
            self._slow1.data(),
            self._a13.data(),
            n_bins=self._n_bins1,
            bin_range=self._actual_range1,
            edge2center=False,
            nan_to_num=True
        )

        self._a23_stats, _, _ = compute_spectrum_1d(
            self._slow1.data(),
            self._a23.data(),
            n_bins=self._n_bins1,
            bin_range=self._actual_range1,
            edge2center=False,
            nan_to_num=True
        )

        self._a21_stats, edges, counts = compute_spectrum_1d(
            self._slow1.data(),
            self._a21.data(),
            n_bins=self._n_bins1,
            bin_range=self._actual_range1,
            edge2center=False,
            nan_to_num=True
        )
        self._edges1 = edges
        self._counts1 = counts

    def _update_1d_binning(self, a13, a23, a21, delay):
        iloc_x = self.searchsorted(self._edges1, delay)
        if 0 <= iloc_x < self._n_bins1:
            self._counts1[iloc_x] += 1
            count = self._counts1[iloc_x]
            self._a13_stats[iloc_x] += (a13 - self._a13_stats[iloc_x]) / count
            self._a23_stats[iloc_x] += (a23 - self._a23_stats[iloc_x]) / count
            self._a21_stats[iloc_x] += (a21 - self._a21_stats[iloc_x]) / count

    def _new_2d_binning(self):
        # to have energy on x axis and delay on y axis
        # Note: the return array from 'stats.binned_statistic_2d' has a swap x and y
        # axis compared to conventional image data
        self._a21_heat, _, self._edges2, _ = \
            stats.binned_statistic_2d(self._slow1.data(),
                                      self._slow2.data(),
                                      self._a21.data(),
                                      'mean',
                                      [self._n_bins1, self._n_bins2],
                                      [self._actual_range1, self._actual_range2])
        np.nan_to_num(self._a21_heat, copy=False)

        self._a21_heat_count, _, _, _ = \
            stats.binned_statistic_2d(self._slow1.data(),
                                      self._slow2.data(),
                                      self._a21.data(),
                                      'count',
                                      [self._n_bins1, self._n_bins2],
                                      [self._actual_range1, self._actual_range2])
        np.nan_to_num(self._a21_heat_count, copy=False)

    def _update_2d_binning(self, a21, energy, delay):
        iloc_x = self.searchsorted(self._edges2, energy)
        iloc_y = self.searchsorted(self._edges1, delay)
        if 0 <= iloc_x < self._n_bins2 \
                and 0 <= iloc_y < self._n_bins1:
            self._a21_heat_count[iloc_y, iloc_x] += 1
            self._a21_heat[iloc_y, iloc_x] += \
                (a21 - self._a21_heat[iloc_y, iloc_x]) / \
                self._a21_heat_count[iloc_y, iloc_x]

    def reset(self):
        """Override."""
        self._slow1.reset()
        self._slow2.reset()
        self._a13.reset()
        self._a23.reset()
        self._a21.reset()

        self._edges1 = None
        self._counts1 = None
        self._a13_stats = None
        self._a23_stats = None
        self._a21_stats = None
        self._edges2 = None
        self._a21_heat = None
        self._a21_heat_count = None

        self._bin1d = True
        self._bin2d = True


class TrXasCtrlWidget(_BaseAnalysisCtrlWidgetS):
    """tr-XAS analysis control widget.

    tr-XAS stands for Time-resolved X-ray Absorption Spectroscopy.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.device_id1_le = SmartStringLineEdit(
            "SCS_ILH_LAS/MOTOR/LT3")
        self.ppt1_le = SmartStringLineEdit("AActualPosition")
        self.label1_le = SmartStringLineEdit("Delay (arb. u.)")

        self.device_id2_le = SmartStringLineEdit(
            "SA3_XTD10_MONO/MDL/PHOTON_ENERGY")
        self.ppt2_le = SmartStringLineEdit("actualEnergy")
        self.label2_le = SmartStringLineEdit("Energy (eV)")

        self.bin_range1_le = SmartBoundaryLineEdit(_DEFAULT_BIN_RANGE)
        self.n_bins1_le = SmartLineEdit(str(_DEFAULT_N_BINS))
        self.n_bins1_le.setValidator(QIntValidator(1, _MAX_N_BINS))

        self.bin_range2_le = SmartBoundaryLineEdit(_DEFAULT_BIN_RANGE)
        self.n_bins2_le = SmartLineEdit(str(_DEFAULT_N_BINS))
        self.n_bins2_le.setValidator(QIntValidator(1, _MAX_N_BINS))

        self.swap_btn = QPushButton("Swap devices")

        self._non_reconfigurable_widgets.extend([
            self.device_id1_le,
            self.ppt1_le,
            self.device_id2_le,
            self.ppt2_le,
            self.swap_btn
        ])

        self.initUI()
        self.initConnections()

    def initUI(self):
        """Override."""
        layout = self.layout()

        layout.addRow("Device ID 1: ", self.device_id1_le)
        layout.addRow("Property 1: ", self.ppt1_le)
        layout.addRow("Label 1: ", self.label1_le)
        layout.addRow("Bin range 1: ", self.bin_range1_le)
        layout.addRow("# of bins 1: ", self.n_bins1_le)
        layout.addRow("Device ID 2: ", self.device_id2_le)
        layout.addRow("Property 2: ", self.ppt2_le)
        layout.addRow("Label 2: ", self.label2_le)
        layout.addRow("Bin range 2: ", self.bin_range2_le)
        layout.addRow("# of bins 2: ", self.n_bins2_le)
        layout.addRow("", self.swap_btn)

    def initConnections(self):
        """Override."""
        self.swap_btn.clicked.connect(self._swapDataSources)

    def _swapDataSources(self):
        self._swapLineEditContent(self.device_id1_le, self.device_id2_le)
        self._swapLineEditContent(self.ppt1_le, self.ppt2_le)
        self._swapLineEditContent(self.label1_le, self.label2_le)
        self._swapLineEditContent(self.bin_range1_le, self.bin_range2_le)
        self._swapLineEditContent(self.n_bins1_le, self.n_bins2_le)

    def _swapLineEditContent(self, edit1, edit2):
        text1 = edit1.text()
        text2 = edit2.text()
        edit1.setText(text2)
        edit2.setText(text1)


class TrXasRoiImageView(ImageViewF):
    """TrXasRoiImageView class.

    Visualize ROIs.
    """
    def __init__(self, idx, **kwargs):
        """Initialization."""
        super().__init__(**kwargs)

        self._index = idx
        self.setTitle(f"ROI{idx}")

    def updateF(self, data):
        """Override."""
        self.setImage(data[f"roi{self._index}"])


class TrXasSpectraPlot(TimedPlotWidgetF):
    """TrXasSpectraPlot class.

    Visualize 1D binning of absorption(s).
    """
    def __init__(self, diff=False, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self._diff = diff

        self.setTitle("XAS")
        self.setLabel('left', "Absorption (arb. u.)")
        self.setLabel('right', "Count")
        self.addLegend(offset=(-40, 20))

        if diff:
            self._a21 = self.plotCurve(
                name="ROI2/ROI1", pen=FoamColor.mkPen("g"))
        else:
            # same color as ROI1
            self._a13 = self.plotCurve(
                name="ROI1/ROI3", pen=FoamColor.mkPen("b"))
            # same color as ROI2
            self._a23 = self.plotCurve(
                name="ROI2/ROI3", pen=FoamColor.mkPen("r"))

        self._count = self.plotBar(
            name="Count", y2=True, brush=FoamColor.mkBrush('i', alpha=70))

    def refresh(self):
        """Override."""
        data = self._data

        centers1 = data["centers1"]
        if centers1 is None:
            return

        if self._diff:
            self._a21.setData(centers1, data["a21_stats"])
        else:
            self._a13.setData(centers1, data["a13_stats"])
            self._a23.setData(centers1, data["a23_stats"])
        self._count.setData(centers1, data["counts1"])

    def onXLabelChanged(self, label):
        self.setLabel('bottom', label)


class TrXasHeatmap(TimedImageViewF):
    """TrXasHeatmap class.

    Visualize 2D binning of absorption.
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(hide_axis=False, parent=parent)
        self.invertY(False)
        self.setAspectLocked(False)

        self.setTitle("XAS (ROI2/ROI1)")

    def refresh(self):
        """Override."""
        data = self._data

        centers2 = data["centers2"]
        centers1 = data["centers1"]
        heat = data["a21_heat"]

        if centers2 is None or centers1 is None:
            return

        # do not update if FOM is None
        if heat is not None:
            self.setImage(heat,
                          pos=[centers2[0], centers1[0]],
                          scale=[(centers2[-1] - centers2[0])/len(centers2),
                                 (centers1[-1] - centers1[0])/len(centers1)])

    def onXLabelChanged(self, label):
        self.setLabel('bottom', label)

    def onYLabelChanged(self, label):
        self.setLabel('left', label)


@create_app(TrXasCtrlWidget,
            TrXasProcessor,
            QThreadFoamClient)
class TrXas(_FoamLightApp):
    """Time-resolved XAS application."""

    icon = "tr_xas.png"
    _title = "tr-XAS"
    _long_title = "Time-resolved X-ray Absorption Spectroscopy"

    def __init__(self, topic):
        """Initialization."""
        super().__init__(topic, with_levels=True)

        self._roi1_image = TrXasRoiImageView(1, parent=self)
        self._roi2_image = TrXasRoiImageView(2, parent=self)
        self._roi3_image = TrXasRoiImageView(3, parent=self)

        self._a13_a23 = TrXasSpectraPlot(parent=self)
        self._a21 = TrXasSpectraPlot(True, parent=self)
        self._a21_heatmap = TrXasHeatmap(parent=self)

        self.initUI()
        self.initConnections()
        self.startWorker()

    def initUI(self):
        """Override."""
        middle_panel = QSplitter(Qt.Vertical)
        middle_panel.addWidget(self._roi1_image)
        middle_panel.addWidget(self._roi2_image)
        middle_panel.addWidget(self._roi3_image)

        right_panel = QSplitter(Qt.Vertical)
        right_panel.addWidget(self._a13_a23)
        right_panel.addWidget(self._a21)
        right_panel.addWidget(self._a21_heatmap)
        right_panel.setSizes([self._TOTAL_H / 3.0] * 3)

        cw = self.centralWidget()
        cw.addWidget(middle_panel)
        cw.addWidget(right_panel)

        self.resize(self._TOTAL_W, self._TOTAL_H)

    def initConnections(self):
        """Override."""
        self._ctrl_widget_st.device_id1_le.value_changed_sgn.connect(
            self._worker_st.onDeviceId1Changed)
        self._ctrl_widget_st.ppt1_le.value_changed_sgn.connect(
            self._worker_st.onProperty1Changed)

        self._ctrl_widget_st.device_id1_le.returnPressed.emit()
        self._ctrl_widget_st.ppt1_le.returnPressed.emit()

        self._ctrl_widget_st.device_id2_le.value_changed_sgn.connect(
            self._worker_st.onDeviceId2Changed)
        self._ctrl_widget_st.ppt2_le.value_changed_sgn.connect(
            self._worker_st.onProperty2Changed)

        self._ctrl_widget_st.device_id2_le.returnPressed.emit()
        self._ctrl_widget_st.ppt2_le.returnPressed.emit()

        self._ctrl_widget_st.n_bins1_le.value_changed_sgn.connect(
            self._worker_st.onNBins1Changed)
        self._ctrl_widget_st.bin_range1_le.value_changed_sgn.connect(
            self._worker_st.onBinRange1Changed)

        self._ctrl_widget_st.n_bins1_le.returnPressed.emit()
        self._ctrl_widget_st.bin_range1_le.returnPressed.emit()

        self._ctrl_widget_st.n_bins2_le.value_changed_sgn.connect(
            self._worker_st.onNBins2Changed)
        self._ctrl_widget_st.bin_range2_le.value_changed_sgn.connect(
            self._worker_st.onBinRange2Changed)

        self._ctrl_widget_st.n_bins2_le.returnPressed.emit()
        self._ctrl_widget_st.bin_range2_le.returnPressed.emit()

        self._ctrl_widget_st.label1_le.value_changed_sgn.connect(
            self._a21.onXLabelChanged)
        self._ctrl_widget_st.label1_le.value_changed_sgn.connect(
            self._a13_a23.onXLabelChanged)
        self._ctrl_widget_st.label1_le.value_changed_sgn.connect(
            self._a21_heatmap.onYLabelChanged)
        self._ctrl_widget_st.label2_le.value_changed_sgn.connect(
            self._a21_heatmap.onXLabelChanged)

        self._ctrl_widget_st.label1_le.returnPressed.emit()
        self._ctrl_widget_st.label2_le.returnPressed.emit()

    @staticmethod
    def edges2centers(edges):
        if edges is None:
            return None, None
        return (edges[1:] + edges[:-1]) / 2.0, edges[1] - edges[0]

    @staticmethod
    def searchsorted(edges, v):
        """A wrapper for np.searchsorted.

        This is to match the behavior of scipy.stats.binned_statistic:

        All but the last (righthand-most) bin is half-open. In other words,
        if bins is [1, 2, 3, 4], then the first bin is [1, 2) (including 1,
        but excluding 2) and the second [2, 3). The last bin, however,
        is [3, 4], which includes 4.
        """
        s = len(edges)
        if s <= 1:
            return -1
        if v == edges[-1]:
            return s - 2
        # use side = 'right' to match the result from scipy
        return np.searchsorted(edges, v, side='right') - 1

    @staticmethod
    def get_actual_range(data, bin_range, auto_range):
        # It is guaranteed that bin_range[0] < bin_range[1]
        if not auto_range[0] and not auto_range[1]:
            return bin_range

        if auto_range[0]:
            v_min = None if data.size == 0 else data.min()
        else:
            v_min = bin_range[0]

        if auto_range[1]:
            v_max = None if data.size == 0 else data.max()
        else:
            v_max = bin_range[1]

        # The following three cases caused by zero-sized array.
        if v_min is None and v_max is None:
            return 0., 1.
        if v_min is None:
            return v_max - 1., v_max
        if v_max is None:
            return v_min, v_min + 1.

        if auto_range[0] and auto_range[1]:
            if v_min == v_max:
                # all elements have the same value
                return v_min - 0.5, v_max + 0.5
        elif v_min >= v_max:
            # two tricky corner cases
            if auto_range[0]:
                v_min = v_max - 1.0
            elif auto_range[1]:
                v_max = v_min + 1.0
            # else cannot happen
        return v_min, v_max
