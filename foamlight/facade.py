"""
Distributed under the terms of the MIT License.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
from collections import OrderedDict
import os.path as osp

from PyQt5.QtCore import pyqtSignal, QSize
from PyQt5.QtWidgets import (
    QGridLayout, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget
)
from PyQt5.QtGui import QIcon

from . import __version__
from .apps import (
    CameraView, Gotthard, GotthardPumpProbe, MultiCameraView, TrXas, VectorView,
    XasTim, XasTimXmcd, XesTiming
)


def create_icon_button(filename, size, *, description=""):
    """Create a QPushButton with icon.

    :param str filename: name of the icon file.
    :param int size: size of the icon (button).
    :param str description: tool tip of the button.
    """
    root_dir = osp.dirname(osp.abspath(__file__))
    btn = QPushButton()
    icon = QIcon(osp.join(root_dir, "apps/icons/" + filename))
    btn.setIcon(icon)
    btn.setIconSize(QSize(size, size))
    btn.setFixedSize(btn.minimumSizeHint())
    if description:
        btn.setToolTip(description)
    return btn


class FacadeController:
    def __init__(self):
        self._facade = None
        self._window = None

    def showFacade(self, topic):
        self._facade = create_facade(topic)
        self._facade.open_analysis_sgn.connect(self._showAnalysis)
        self._facade.show()

    def _showAnalysis(self, analysis_type, topic):
        self._window = analysis_type(topic)
        self._facade.close()
        self._window.show()


class _FacadeBase(QMainWindow):
    """Base class for special analysis suite."""
    _ICON_WIDTH = 160
    _ROW_HEIGHT = 220
    _WIDTH = 720

    open_analysis_sgn = pyqtSignal(object, str)

    def __init__(self, topic):
        super().__init__()

        self.setWindowTitle(f"foamlight {__version__}")

        # StatusBar to display topic name
        self.statusBar().showMessage(f"TOPIC: {topic}")
        self.statusBar().setStyleSheet("QStatusBar{font-weight:bold;}")

        self._topic = topic

        self._buttons = OrderedDict()

        self._cw = QWidget()
        self.setCentralWidget(self._cw)

    def initUI(self):
        layout = QVBoxLayout()
        layout_row = None
        for i, (title, btn) in enumerate(self._buttons.items()):
            if i % 4 == 0:
                layout_row = QGridLayout()
                layout.addLayout(layout_row)
            layout_row.addWidget(QLabel(title), 0, i % 4)
            layout_row.addWidget(btn, 1, i % 4)
            layout_row.setColumnStretch(3, 2)
            layout_row.setRowStretch(2, 2)
        self._cw.setLayout(layout)

        h = len(self._buttons) // 4
        if len(self._buttons) % 4 != 0:
            h += 1
        self.setFixedSize(self._WIDTH, h * self._ROW_HEIGHT)

    def addSpecial(self, instance_type):
        """Add a button for the given analysis."""
        btn = create_icon_button(instance_type.icon, self._ICON_WIDTH)
        btn.clicked.connect(lambda: self.open_analysis_sgn.emit(
            instance_type, self._topic))

        title = instance_type._title
        if title in self._buttons:
            raise RuntimeError(f"Duplicated special analysis title: {title}")
        self._buttons[title] = btn

    def addCommonSpecials(self):
        self.addSpecial(CameraView)
        self.addSpecial(VectorView)
        self.addSpecial(MultiCameraView)


class SpbFacade(_FacadeBase):
    def __init__(self):
        super().__init__("SPB")

        self.addSpecial(Gotthard)
        self.addSpecial(GotthardPumpProbe)
        self.addCommonSpecials()

        self.initUI()
        self.show()


class FxeFacade(_FacadeBase):
    def __init__(self):
        super().__init__("FXE")

        self.addSpecial(XesTiming)
        self.addSpecial(Gotthard)
        self.addSpecial(GotthardPumpProbe)
        self.addCommonSpecials()

        self.initUI()
        self.show()


class ScsFacade(_FacadeBase):
    def __init__(self):
        super().__init__("SCS")

        self.addSpecial(XasTim)
        self.addSpecial(XasTimXmcd)
        self.addSpecial(TrXas)
        self.addSpecial(Gotthard)
        self.addSpecial(GotthardPumpProbe)
        self.addCommonSpecials()

        self.initUI()
        self.show()


class MidFacade(_FacadeBase):
    def __init__(self):
        super().__init__("MID")

        self.addSpecial(Gotthard)
        self.addSpecial(GotthardPumpProbe)
        self.addCommonSpecials()

        self.initUI()
        self.show()


class HedFacade(_FacadeBase):
    def __init__(self):
        super().__init__("HED")

        self.addSpecial(Gotthard)
        self.addSpecial(GotthardPumpProbe)
        self.addCommonSpecials()

        self.initUI()
        self.show()


def create_facade(topic):
    if topic == "SPB":
        return SpbFacade()

    if topic == "FXE":
        return FxeFacade()

    if topic == "SCS":
        return ScsFacade()

    if topic == "MID":
        return MidFacade()

    if topic == "HED":
        return HedFacade()

    raise ValueError(f"{topic} does not have a special analysis suite")
