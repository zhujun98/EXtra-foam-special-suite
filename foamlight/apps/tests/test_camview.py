from unittest.mock import MagicMock, patch, PropertyMock
from collections import Counter

import pytest
import numpy as np
from xarray import DataArray

from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest

from foamlight import mkQApp
from foamlight.apps.camera_view import (
    CameraView, CameraViewProcessor, CameraViewRoiHist, CameraViewImage
)
from foamlight.logger import logger

from . import _AppTestBase, _ProcessorTestBase, _RawDataMixin

app = mkQApp()

logger.setLevel('INFO')


class TestCamViewApp(_AppTestBase):
    @classmethod
    def setUpClass(cls):
        cls._app = CameraView('SCS')

    @classmethod
    def tearDownClass(cls):
        # explicitly close the MainGUI to avoid error in GuiLogger
        cls._app.close()

    @staticmethod
    def data4visualization():
        """Override."""
        return {
            "displayed": np.arange(20).reshape(4, 5),
            "roi_hist": (np.arange(4), np.arange(4), 1, 2, 3)
        }

    def testWidgets(self):
        self.assertEqual(2, len(self._app._plot_widgets_st))
        counter = Counter()
        for key in self._app._plot_widgets_st:
            counter[key.__class__] += 1

        self.assertEqual(1, counter[CameraViewImage])
        self.assertEqual(1, counter[CameraViewRoiHist])

        self._check_update_plots()

    def testCtrl(self):
        from foamlight.apps.camera_view import (
            _DEFAULT_OUTPUT_CHANNEL, _DEFAULT_PROPERTY, _DEFAULT_N_BINS,
            _DEFAULT_BIN_RANGE
        )

        ctrl_widget = self._app._ctrl_widget_st
        proc = self._app._worker_st

        # test default values
        self.assertEqual(_DEFAULT_OUTPUT_CHANNEL, proc._output_channel)
        self.assertEqual(_DEFAULT_PROPERTY, proc._ppt)
        self.assertEqual(1, proc.__class__._raw_ma.window)
        self.assertTupleEqual(tuple(float(v) for v in _DEFAULT_BIN_RANGE.split(',')),
                              proc._bin_range)
        self.assertEqual(int(_DEFAULT_N_BINS), proc._n_bins)

        # test set new values
        widget = ctrl_widget.output_ch_le
        widget.clear()
        QTest.keyClicks(widget, "new/output/channel")
        QTest.keyPress(widget, Qt.Key_Enter)
        self.assertEqual("new/output/channel", proc._output_channel)

        widget = ctrl_widget.property_le
        widget.clear()
        QTest.keyClicks(widget, "new/property")
        QTest.keyPress(widget, Qt.Key_Enter)
        self.assertEqual("new/property", proc._ppt)

        widget = ctrl_widget.ma_window_le
        widget.clear()
        QTest.keyClicks(widget, "9")
        QTest.keyPress(widget, Qt.Key_Enter)
        self.assertEqual(9, proc.__class__._raw_ma.window)

        widget = ctrl_widget.bin_range_le
        widget.clear()
        QTest.keyClicks(widget, "-1.0, 1.0")
        QTest.keyPress(widget, Qt.Key_Enter)
        self.assertTupleEqual((-1.0, 1.0), proc._bin_range)

        widget = ctrl_widget.n_bins_le
        widget.clear()
        QTest.keyClicks(widget, "1000")
        QTest.keyPress(widget, Qt.Key_Enter)
        self.assertEqual(100, proc._n_bins)  # maximum is 999 and one can not put the 3rd 0 in
        widget.clear()
        QTest.keyClicks(widget, "999")
        QTest.keyPress(widget, Qt.Key_Enter)
        self.assertEqual(999, proc._n_bins)


class TestCameraViewProcessor(_RawDataMixin, _ProcessorTestBase):
    @pytest.fixture(autouse=True)
    def setUp(self):
        self._proc = CameraViewProcessor(object(), object())

        self._proc._output_channel = "camera:output"
        self._proc._ppt = "data.image"
        self._img_data = np.random.randint(0, 100, size=(4, 4), dtype=np.uint16)

    def _get_data(self, tid, times=1):
        # data, meta
        return self._gen_data(tid, {
            "camera:output": [
                ("data.image", times * self._img_data),
                ("data.squeezable.3d", np.ones((2, 2, 1))),
                ("data.3d", np.ones((4, 2, 2)))
        ]})

    def testPreprocessImageData(self):
        proc = self._proc
        data = self._get_data(12345)

        with patch.object(CameraViewProcessor, "_ppt",
                          new_callable=PropertyMock, create=True, return_value="data.3d"):
            assert(proc.process(data) is None)

        with patch.object(CameraViewProcessor, "_ppt",
                          new_callable=PropertyMock, create=True, return_value="data.squeezable.3d"):
            np.testing.assert_array_equal(np.ones((2, 2), dtype=np.float32),
                                          proc.process(data)['displayed'])

    @patch("foamlight.core.QThreadWorker._loadRunDirectoryST")
    def testLoadDarkRun(self, load_run):
        proc = self._proc

        load_run.return_value = None
        # nothing should happen
        proc.onLoadDarkRun("run/path")

        data_collection = MagicMock()
        load_run.return_value = data_collection
        with patch.object(proc.log, "error") as error:
            # get_array returns a wrong shape
            data_collection.get_array.return_value = DataArray(np.random.randn(4, 3))
            proc.onLoadDarkRun("run/path")
            error.assert_called_once()
            error.reset_mock()

            # get_array returns a correct shape
            data_collection.get_array.return_value = DataArray(np.random.randn(4, 3, 2))
            with patch.object(proc.log, "info") as info:
                proc.onLoadDarkRun("run/path")
                info.assert_called_once()
                error.assert_not_called()

    def testProcessingWhenRecordingDark(self):
        from foamlight.apps.camera_view import _IMAGE_DTYPE

        proc = self._proc
        assert 2147483647 == proc.__class__._dark_ma.window

        proc._recording_dark_st = True
        proc._subtract_dark_st = True  # take no effect

        imgdata_gt = self._img_data.astype(_IMAGE_DTYPE)
        imgdata_gt2 = 2.0 * self._img_data
        imgdata_gt_avg = 1.5 * self._img_data

        # 1st train
        processed = proc.process(self._get_data(12345))
        np.testing.assert_array_almost_equal(imgdata_gt, proc._dark_ma)
        np.testing.assert_array_almost_equal(imgdata_gt, processed["displayed"])

        # 2nd train
        processed = proc.process(self._get_data(12346, 2))
        np.testing.assert_array_almost_equal(imgdata_gt_avg, proc._dark_ma)
        np.testing.assert_array_almost_equal(imgdata_gt_avg, processed["displayed"])

        # 3nd train
        processed = proc.process(self._get_data(12347, 3))
        np.testing.assert_array_almost_equal(imgdata_gt2, proc._dark_ma)
        np.testing.assert_array_almost_equal(imgdata_gt2, processed["displayed"])

        # reset
        proc.reset()
        assert proc._dark_ma is None

    @pytest.mark.parametrize("subtract_dark", [(True, ), (False,)])
    def testProcessing(self, subtract_dark):
        from foamlight.apps.camera_view import _IMAGE_DTYPE

        proc = self._proc
        proc._recording_dark = False
        proc._poi_index = 1

        proc._subtract_dark = subtract_dark
        offset = np.ones_like(self._img_data).astype(_IMAGE_DTYPE)
        proc._dark_ma = offset

        imgdata_gt = self._img_data.astype(_IMAGE_DTYPE)
        imgdata_gt2 = 2.0 * self._img_data
        imgdata_gt_avg = 1.5 * self._img_data
        if subtract_dark:
            imgdata_gt -= offset
            imgdata_gt2 -= offset
            imgdata_gt_avg -= offset

        # 1st train
        processed = proc.process(self._get_data(12345))
        self._check_processed_data_structure(processed)
        np.testing.assert_array_almost_equal(imgdata_gt, processed["displayed"])

        # 2nd train
        proc.__class__._raw_ma.window = 3
        processed = proc.process(self._get_data(12346, 2))
        np.testing.assert_array_almost_equal(imgdata_gt_avg, processed["displayed"])

        # 3nd train
        processed = proc.process(self._get_data(12347, 3))
        np.testing.assert_array_almost_equal(imgdata_gt2, processed["displayed"])

        # reset
        proc.reset()
        assert proc._raw_ma is None

    def _check_processed_data_structure(self, ret):
        """Override."""
        data_gt = TestCamViewApp.data4visualization().keys()
        assert set(ret.keys()) == set(data_gt)
