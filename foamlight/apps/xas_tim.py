"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file BSD_LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.

Modified from https://github.com/European-XFEL/EXtra-foam, version 1.0.0
Copyright (C) Jun Zhu
"""
from functools import partial

import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator, QIntValidator
from PyQt5.QtWidgets import (
    QButtonGroup, QCheckBox, QHBoxLayout, QSplitter, QTabWidget
)

from foamgraph import (
    FoamColor, PlotWidgetF, SmartLineEdit, SmartStringLineEdit,
    TimedPlotWidgetF
)
from pyfoamalgo import (
    compute_spectrum_1d, SimpleSequence, SimplePairSequence
)

from ..config import config
from ..core import (
    create_app, profiler, QThreadWorker, QThreadKbClient,
    _BaseAnalysisCtrlWidgetS, _FoamLightApp
)
from ..exceptions import ProcessingError

_DIGITIZER_CHANNEL_COLORS = ['r', 'b', 'o', 'k']
_DEFAULT_N_PULSES_PER_TRAIN = 1
_DEFAULT_I0_THRESHOLD = 0.0
_MAX_WINDOW = 180000  # 60 s * 10 train/s * 300 pulses/train
_MAX_CORRELATION_WINDOW = 3000
_MAX_N_BINS = 999
_DEFAULT_N_BINS = 80
# MCP 1 - 4
_DIGITIZER_CHANNEL_NAMES = ['D', 'C', 'B', 'A']


class XasTimProcessor(QThreadWorker):
    """XAS-TIM processor.

    Attributes:
        _xgm_output_channel (str): XGM output channel name.
        _xgm_ppt (str): XGM property name for pulse-resolved intensity.
        _digitizer_output_channel (str): Digitizer output channel name.
        _digitizer_ppts (list): A list of property names for different
            digitizer channels.
        _mono_device_id (str): Soft mono device ID.
        _mono_ppt (str): Soft mono property name for energy.
        _digitizer_channels (list): A list of boolean to indicates the
            required digitizer channel.
        _n_pulses_per_train (int): Number of pulses per train.
        _apd_stride (int): Pulse index stride of the digitizer APD data.
        _i0_threshold (float): Lower boundary of the XGM intensity. Pulses
            will be dropped if the intensity is below the threshold.
        _window (int): Maximum number of pulses used to calculating spectra.
        _correlation_window (int): Maximum number of pulses in correlation
            plots. It includes the pulses which are dropped by the filter.
        _n_bins (int): Number of bins in spectra calculation.
        _i0 (SimpleSequence): Store XGM pulse intensities.
        _i1 (list): A list of SimpleSequence, which stores pulsed apd data
            for each digitizer channel.
        _energy (SimpleSequence): Store pulse energies.
        _energy_scan (SimplePairSequence): A sequence of (train ID, energy).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._xgm_output_channel = ""
        self._xgm_ppt = "data.intensitySa3TD"
        self._digitizer_output_channel = ""
        self._digitizer_ppts = [
            f"digitizers.channel_1_{ch}.apd.pulseIntegral"
            for ch in _DIGITIZER_CHANNEL_NAMES
        ]
        self._mono_device_id = ""
        self._mono_ppt = "actualEnergy"

        self._digitizer_channels = [False] * 4

        self._n_pulses_per_train = _DEFAULT_N_PULSES_PER_TRAIN
        self._apd_stride = 1
        self._i0_threshold = _DEFAULT_I0_THRESHOLD
        self._window = _MAX_WINDOW
        self._correlation_window = _MAX_CORRELATION_WINDOW
        self._n_bins = _DEFAULT_N_BINS

        self._i0 = SimpleSequence(max_len=_MAX_WINDOW)
        self._i1 = [SimpleSequence(max_len=_MAX_WINDOW)
                    for _ in _DIGITIZER_CHANNEL_NAMES]
        self._energy = SimpleSequence(max_len=_MAX_WINDOW)

        self._energy_scan = SimplePairSequence(max_len=_MAX_WINDOW)

    def onXgmOutputChannelChanged(self, ch: str):
        self._xgm_output_channel = ch

    def onDigitizerOutputChannelChanged(self, ch: str):
        self._digitizer_output_channel = ch

    def onDigitizerChannelsChanged(self, index: int, value: bool):
        self._digitizer_channels[index] = value
        if value:
            # reset the data history when a new channel is added in order to
            # ensure the same length of data history
            self.reset()

    def onMonoDeviceChanged(self, device: str):
        self._mono_device_id = device

    def onNPulsesPerTrainChanged(self, value: str):
        self._n_pulses_per_train = int(value)

    def onApdStrideChanged(self, value: str):
        self._apd_stride = int(value)

    def onI0ThresholdChanged(self, value: str):
        self._i0_threshold = float(value)

    def onPulseWindowChanged(self, value: str):
        self._window = int(value)

    def onCorrelationWindowChanged(self, value: str):
        self._correlation_window = int(value)

    def onNoBinsChanged(self, value: str):
        self._n_bins = int(value)

    def sources(self):
        """Override."""
        return [
            (self._xgm_output_channel, self._xgm_ppt, 1),
            *[(self._digitizer_output_channel, ppt, 1)
              for ppt in self._digitizer_ppts],
            (self._mono_device_id, self._mono_ppt, 0)
        ]

    def _update_data_history(self, data):
        data, meta = data["raw"], data["meta"]

        tid = self.getTrainId(meta)

        xgm_intensity = self.getPropertyData(
            data, self._xgm_output_channel, self._xgm_ppt)

        digitizer_apds = []
        if sum(self._digitizer_channels) == 0:
            raise ProcessingError(
                "At least one digitizer channel is required!")
        for i, ppt in enumerate(self._digitizer_ppts):
            if self._digitizer_channels[i]:
                apd = self.getPropertyData(
                    data, self._digitizer_output_channel, ppt)
                if apd is None:
                    raise ProcessingError(
                        f"Digitizer channel {ppt} not found!")
                digitizer_apds.append(apd)
            else:
                digitizer_apds.append(None)

        energy = self.getPropertyData(
            data, self._mono_device_id, self._mono_ppt)

        # Check and slice XGM intensity.
        pulse_slicer = slice(0, self._n_pulses_per_train)
        if len(xgm_intensity) < self._n_pulses_per_train:
            raise ProcessingError(f"Length of {self._xgm_ppt} is less "
                                  f"than {self._n_pulses_per_train}: "
                                  f"actual {len(xgm_intensity)}")
        xgm_intensity = xgm_intensity[pulse_slicer]

        # Check and slice digitizer APD data.
        for i, (apd, ppt) in enumerate(zip(digitizer_apds,
                                           self._digitizer_ppts)):
            if self._digitizer_channels[i]:
                v = apd[::self._apd_stride]
                if len(v) < self._n_pulses_per_train:
                    raise ProcessingError(
                        f"Length of {ppt} (sliced) is less than "
                        f"{self._n_pulses_per_train}: actual {len(v)}")
                digitizer_apds[i] = v[:self._n_pulses_per_train]

        # update data history
        self._i0.extend(xgm_intensity)
        for i, apd in enumerate(digitizer_apds):
            if self._digitizer_channels[i]:
                self._i1[i].extend(apd)
        self._energy.extend([energy] * len(xgm_intensity))

        self._energy_scan.append((tid, energy))

        return tid, xgm_intensity, digitizer_apds, energy

    @profiler("XAS-TIM Processor")
    def process(self, data):
        """Override."""
        tid, xgm_intensity, digitizer_apds, energy = \
            self._update_data_history(data)

        # apply filter
        flt = self._i0.data() > self._i0_threshold
        i0 = self._i0.data()[flt][-self._window:]
        i1 = [None] * 4
        for i, _item in enumerate(self._i1):
            if self._digitizer_channels[i]:
                i1[i] = _item.data()[flt][-self._window:]
        energy = self._energy.data()[flt][-self._window:]

        # compute spectra
        stats = []
        for i, item in enumerate(i1):
            if self._digitizer_channels[i]:
                mcp_stats, _, _ = compute_spectrum_1d(
                    energy, item, n_bins=self._n_bins)
                stats.append(mcp_stats)
            else:
                # Do not calculate spectrum which is not requested to display
                stats.append(None)

        i0_stats, centers, counts = compute_spectrum_1d(
            energy, i0, n_bins=self._n_bins)
        for i, _item in enumerate(stats):
            if _item is not None:
                if i < 3:
                    stats[i] = -np.log(-_item / i0_stats)
                else:
                    # MCP4 has a different spectrum
                    stats[i] = -_item / i0_stats
        stats.append(i0_stats)

        self.log.info(f"Train {tid} processed")

        return {
            "xgm_intensity": xgm_intensity,
            "digitizer_apds": digitizer_apds,
            "energy_scan": self._energy_scan.data(),
            "correlation_length": self._correlation_window,
            "i0": i0,
            "i1": i1,
            "spectra": (stats, centers, counts),
        }

    def reset(self):
        """Override."""
        self._i0.reset()
        for item in self._i1:
            item.reset()
        self._energy.reset()
        self._energy_scan.reset()


class XasTimCtrlWidget(_BaseAnalysisCtrlWidgetS):
    """XAS-TIM analysis control widget.

    XAS-TIM stands for X-ray Absorption Spectroscopy with transmission
    intensity monitor.
    """

    # True if spectrum from one MCP channel can be visualized at a time.
    _MCP_EXCLUSIVE = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.xgm_output_ch_le = SmartStringLineEdit(
            "SCS_BLU_XGM/XGM/DOOCS:output")

        self.digitizer_output_ch_le = SmartStringLineEdit(
            "SCS_UTC1_ADQ/ADC/1:network")
        self.digitizer_channels = QButtonGroup()
        self.digitizer_channels.setExclusive(False)
        for i, ch in enumerate(_DIGITIZER_CHANNEL_NAMES, 1):
            cb = QCheckBox(ch, self)
            cb.setChecked(True)
            self.digitizer_channels.addButton(cb, i-1)

        self.mono_device_le = SmartStringLineEdit(
            "SA3_XTD10_MONO/MDL/PHOTON_ENERGY")

        self.n_pulses_per_train_le = SmartLineEdit(
            str(_DEFAULT_N_PULSES_PER_TRAIN))
        self.n_pulses_per_train_le.setValidator(
            QIntValidator(1, config["MAX_N_PULSES_PER_TRAIN"]))

        self.apd_stride_le = SmartLineEdit("1")

        self.spectra_displayed = QButtonGroup()
        self.spectra_displayed.setExclusive(self._MCP_EXCLUSIVE)
        for i, _ in enumerate(_DIGITIZER_CHANNEL_NAMES, 1):
            cb = QCheckBox(f"MCP{i}", self)
            cb.setChecked(True)
            self.spectra_displayed.addButton(cb, i-1)

        self.i0_threshold_le = SmartLineEdit(str(_DEFAULT_I0_THRESHOLD))
        self.i0_threshold_le.setValidator(QDoubleValidator())

        self.pulse_window_le = SmartLineEdit(str(_MAX_WINDOW))
        self.pulse_window_le.setValidator(QIntValidator(1, _MAX_WINDOW))

        self.correlation_window_le = SmartLineEdit(
            str(_MAX_CORRELATION_WINDOW))
        self.correlation_window_le.setValidator(
            QIntValidator(1, _MAX_CORRELATION_WINDOW))

        self.n_bins_le = SmartLineEdit(str(_DEFAULT_N_BINS))
        self.n_bins_le.setValidator(QIntValidator(1, _MAX_N_BINS))

        self._non_reconfigurable_widgets = [
            self.xgm_output_ch_le,
            self.digitizer_output_ch_le,
            *self.digitizer_channels.buttons(),
            self.mono_device_le,
        ]

        self.initUI()
        self.initConnections()

    def initUI(self):
        """Override."""
        layout = self.layout()

        digitizer_channels_layout = QHBoxLayout()
        for cb in self.digitizer_channels.buttons():
            digitizer_channels_layout.addWidget(cb)

        spectra_displayed_layout = QHBoxLayout()
        for cb in self.spectra_displayed.buttons():
            spectra_displayed_layout.addWidget(cb)

        layout.addRow("XGM output channel: ", self.xgm_output_ch_le)
        layout.addRow("Digitizer output channel: ", self.digitizer_output_ch_le)
        layout.addRow("Digitizer channels: ", digitizer_channels_layout)
        layout.addRow("Mono device ID: ", self.mono_device_le)
        layout.addRow("# of pulses/train: ", self.n_pulses_per_train_le)
        layout.addRow("APD stride: ", self.apd_stride_le)
        layout.addRow('XGM intensity threshold: ', self.i0_threshold_le)
        layout.addRow('Pulse window: ', self.pulse_window_le)
        layout.addRow('Correlation window: ', self.correlation_window_le)
        layout.addRow('# of energy bins: ', self.n_bins_le)
        layout.addRow("Show spectra: ", spectra_displayed_layout)

    def initConnections(self):
        """Override."""
        pass


class XasTimXgmPulsePlot(PlotWidgetF):
    """XasTimXgmPulsePlot class.

    Visualize XGM intensity in the current train.
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self.setTitle("Pulse intensities (SA3)")
        self.setLabel('left', "Intensity (arb.)")
        self.setLabel('bottom', "Pulse index")

        self._plot = self.plotCurve(pen=FoamColor.mkPen("g"))

    def updateF(self, data):
        """Override."""
        y = data['xgm_intensity']
        x = np.arange(len(y))
        self._plot.setData(x, y)


class XasTimDigitizerPulsePlot(PlotWidgetF):
    """XasTimDigitizerPulsePlot class.

    Visualize pulse integral of each channel of the digitizer
    in the current train.
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self.setTitle("Digitizer pulse integrals")
        self.setLabel('left', "Pulse integral (arb.)")
        self.setLabel('bottom', "Pulse index")
        self.addLegend(offset=(-40, 60))

        self._plots = []
        for ch, c in zip(_DIGITIZER_CHANNEL_NAMES, _DIGITIZER_CHANNEL_COLORS):
            self._plots.append(self.plotCurve(
                name=f"Digitizer channel {ch}", pen=FoamColor.mkPen(c)))

    def updateF(self, data):
        """Override."""
        for p, apd in zip(self._plots, data['digitizer_apds']):
            if apd is None:
                p.setData([], [])
            else:
                x = np.arange(len(apd))
                p.setData(x, apd)


class XasTimMonoScanPlot(TimedPlotWidgetF):
    """XasTimMonoScanPlot class.

    Visualize path of soft mono energy scan.
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self.setTitle("Softmono energy scan")
        self.setLabel('left', "Energy (eV)")
        self.setLabel('bottom', "Train ID")

        self._plot = self.plotCurve(pen=FoamColor.mkPen('b'))

    def refresh(self):
        """Override."""
        self._plot.setData(*self._data['energy_scan'])


class XasTimCorrelationPlot(TimedPlotWidgetF):
    """XasTimCorrelationPlot class.

    Visualize correlation between I0 and I1 for single channel.
    """
    def __init__(self, idx, *, parent=None):
        """Initialization.

        :param int idx: channel index.
        """
        super().__init__(parent=parent)

        self.setLabel('left', "I1 (arb.)")
        self.setLabel('bottom', "I0 (micro J)")
        self.setTitle(f"MCP{idx+1} correlation")
        self._idx = idx

        self._plot = self.plotScatter(
            brush=FoamColor.mkBrush(_DIGITIZER_CHANNEL_COLORS[idx], alpha=150))

    def refresh(self):
        """Override."""
        data = self._data
        i1 = data['i1'][self._idx]
        if i1 is None:
            self._plot.setData([], [])
        else:
            s = data['correlation_length']
            self._plot.setData(data['i0'][-s:], i1[-s:])


class XasTimSpectraPlot(TimedPlotWidgetF):
    """XasTimSpectraPlot class.

    Visualize spectrum for all MCPs.
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self.setTitle("MCP spectra")
        self.setLabel('left', "Absorption (arb.)")
        self.setLabel('right', "Count")
        self.setLabel('bottom', "Energy (eV)")
        self.addLegend(offset=(-40, 20))

        self._displayed = [False] * 4

        self._plots = []
        for i, c in enumerate(_DIGITIZER_CHANNEL_COLORS):
            self._plots.append(
                self.plotCurve(name=f"MCP{i+1}", pen=FoamColor.mkPen(c)))
        self._count = self.plotBar(
            y2=True, brush=FoamColor.mkBrush('i', alpha=70))

    def refresh(self):
        """Override."""
        stats, centers, counts = self._data["spectra"]
        for i, p in enumerate(self._plots):
            v = stats[i]
            if v is not None and self._displayed[i]:
                p.setData(centers, v)
            else:
                p.setData([], [])
        self._count.setData(centers, counts)

    def onSpectraDisplayedChanged(self, index: int, value: bool):
        self._displayed[index] = value


class XasTimXgmSpectrumPlot(TimedPlotWidgetF):
    """XasTimXgmSpectrumPlot class.

    Visualize spectrum of I0 (XGM).
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self.setTitle("XGM spectrum")
        self.setLabel('left', "I0 (arb.)")
        self.setLabel('right', "Count")
        self.setLabel('bottom', "Energy (eV)")

        self._plot = self.plotScatter(brush=FoamColor.mkBrush("w"))
        self._count = self.plotBar(
            y2=True, brush=FoamColor.mkBrush('i', alpha=70))

    def refresh(self):
        """Override."""
        stats, centers, counts = self._data["spectra"]
        self._plot.setData(centers, stats[4])
        self._count.setData(centers, counts)


@create_app(XasTimCtrlWidget,
            XasTimProcessor,
            QThreadKbClient)
class XasTim(_FoamLightApp):
    """XAS-TIM Application."""

    icon = "xas_tim.png"
    _title = "XAS-TIM"
    _long_title = "X-ray Absorption Spectroscopy with transmission " \
                  "intensity monitor"

    def __init__(self, topic):
        super().__init__(topic, with_dark=False, with_levels=False)

        self._xgm = XasTimXgmPulsePlot(parent=self)
        self._digitizer = XasTimDigitizerPulsePlot(parent=self)
        self._mono = XasTimMonoScanPlot(parent=self)

        self._correlations = [XasTimCorrelationPlot(i, parent=self)
                              for i in range(4)]
        self._spectra = XasTimSpectraPlot(parent=self)
        self._i0_spectrum = XasTimXgmSpectrumPlot(parent=self)

        self.initUI()
        self.initConnections()
        self.startWorker()

    def initUI(self):
        """Override."""
        right_panel = QTabWidget()

        right_panel1 = QSplitter(Qt.Vertical)
        right_panel1.addWidget(self._xgm)
        right_panel1.addWidget(self._digitizer)
        right_panel1.addWidget(self._mono)

        right_panel2 = QSplitter(Qt.Horizontal)
        correlation_panel = QSplitter(Qt.Vertical)
        for w in self._correlations:
            correlation_panel.addWidget(w)
        spectra_panel = QSplitter(Qt.Vertical)
        spectra_panel.addWidget(self._spectra)
        spectra_panel.addWidget(self._i0_spectrum)
        right_panel2.addWidget(correlation_panel)
        right_panel2.addWidget(spectra_panel)
        right_panel2.setSizes([100, 200])

        right_panel.addTab(right_panel1, "Raw data")
        right_panel.addTab(right_panel2, "Correlation and spectra")
        right_panel.setTabPosition(QTabWidget.TabPosition.South)

        cw = self.centralWidget()
        cw.addWidget(right_panel)
        cw.setSizes([self._TOTAL_W / 4, 3 * self._TOTAL_W / 4])

        self.resize(self._TOTAL_W, self._TOTAL_H)

    def initConnections(self):
        """Override."""
        self._ctrl_widget_st.xgm_output_ch_le.value_changed_sgn.connect(
            self._worker_st.onXgmOutputChannelChanged)
        self._ctrl_widget_st.xgm_output_ch_le.returnPressed.emit()

        self._ctrl_widget_st.digitizer_output_ch_le.value_changed_sgn.connect(
            self._worker_st.onDigitizerOutputChannelChanged)
        self._ctrl_widget_st.digitizer_output_ch_le.returnPressed.emit()

        for i, cb in enumerate(self._ctrl_widget_st.digitizer_channels.buttons()):
            cb.toggled.connect(
                partial(self._worker_st.onDigitizerChannelsChanged, i))
            cb.toggled.emit(cb.isChecked())

        self._ctrl_widget_st.mono_device_le.value_changed_sgn.connect(
            self._worker_st.onMonoDeviceChanged)
        self._ctrl_widget_st.mono_device_le.returnPressed.emit()

        self._ctrl_widget_st.n_pulses_per_train_le.value_changed_sgn.connect(
            self._worker_st.onNPulsesPerTrainChanged)
        self._ctrl_widget_st.n_pulses_per_train_le.returnPressed.emit()

        self._ctrl_widget_st.apd_stride_le.value_changed_sgn.connect(
            self._worker_st.onApdStrideChanged)
        self._ctrl_widget_st.apd_stride_le.returnPressed.emit()

        self._ctrl_widget_st.i0_threshold_le.value_changed_sgn.connect(
            self._worker_st.onI0ThresholdChanged)
        self._ctrl_widget_st.i0_threshold_le.returnPressed.emit()

        self._ctrl_widget_st.pulse_window_le.value_changed_sgn.connect(
            self._worker_st.onPulseWindowChanged)
        self._ctrl_widget_st.pulse_window_le.returnPressed.emit()

        self._ctrl_widget_st.correlation_window_le.value_changed_sgn.connect(
            self._worker_st.onCorrelationWindowChanged)
        self._ctrl_widget_st.correlation_window_le.returnPressed.emit()

        self._ctrl_widget_st.n_bins_le.value_changed_sgn.connect(
            self._worker_st.onNoBinsChanged)
        self._ctrl_widget_st.n_bins_le.returnPressed.emit()

        for i, cb in enumerate(self._ctrl_widget_st.spectra_displayed.buttons()):
            cb.toggled.connect(
                partial(self._spectra.onSpectraDisplayedChanged, i))
            cb.toggled.emit(cb.isChecked())
