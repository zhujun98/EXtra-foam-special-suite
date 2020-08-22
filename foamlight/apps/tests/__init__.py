import abc
import time
import unittest

import numpy as np

from foamgraph import TimedPlotWidgetF, TimedImageViewF

from foamlight.logger import logger


class _AppTestBase(unittest.TestCase):
    @staticmethod
    def data4visualization():
        raise NotImplementedError

    def _check_update_plots(self):
        app = self._app
        worker = app._worker_st

        with self.assertLogs(logger, level="ERROR") as cm:
            logger.error("dummy")  # workaround

            app.updateWidgetsST()  # with empty data

            worker._output_st.put_pop(self.data4visualization())
            app.updateWidgetsST()
            for widget in app._plot_widgets_st:
                if isinstance(widget, TimedPlotWidgetF):
                    widget.refresh()
            for widget in app._image_views_st:
                if isinstance(widget, TimedImageViewF):
                    widget.refresh()

            self.assertEqual(1, len(cm.output))


class _ProcessorTestBase:
    @abc.abstractmethod
    def _check_processed_data_structure(self, processed):
        raise NotImplementedError

    @abc.abstractmethod
    def _check_reset(self, proc):
        raise NotImplementedError


class _RawDataMixin:
    """Generate raw data used in test."""
    @staticmethod
    def _update_metadata(meta, src, timestamp, tid):
        sec, frac = str(timestamp).split('.')
        meta[src] = {
            'source': src,
            'timestamp': timestamp,
            'timestamp.tid': tid,
            'timestamp.sec': sec,
            'timestamp.frac': frac.ljust(18, '0')  # attosecond resolution
        }
        return meta

    def _gen_kb_data(self, tid, mapping):
        """Generate empty data in European XFEL data format.

        :param int tid: train ID.
        :param dict mapping: a dictionary with keys being the device IDs /
            output channels and values being a list of (property, value).
        """
        meta, data = {}, {}

        for src, ppts in mapping.items():
            self._update_metadata(meta, src, time.time(), tid)

            data[src] = dict()
            for ppt, value in ppts:
                data[src][ppt] = value

        return data, meta

    def _gen_data(self, tid, mapping, *, source_type=None):
        """Generate empty data in EXtra-foam data format.

        :param int tid: train ID.
        :param dict mapping: a dictionary with keys being the device IDs /
            output channels and values being the list of (property, value).
        """
        meta, data = {}, {}

        for name, ppts in mapping.items():
            for ppt, value in ppts:
                if ".value" in ppt:
                    # slow data from files
                    ppt = ppt[:-6]
                src = f"{name} {ppt}"
                data[src] = value
                meta[src] = {"train_id": tid, "source_type": source_type}

        return {
            "raw": data,
            "processed": None,
            "meta": meta,
            "catalog": None,
        }
