"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from collections import namedtuple
from functools import partial
import pickle
from threading import Thread
from socket import gethostname
from getpass import getuser
from time import time, sleep

import zmq
import msgpack

from karabo_bridge import deserialize
from karabo_bridge import Client as KaraboBridgeClient


class FoamZmqClient:
    """Internal zmq client.

    It uses pickle to serialize and deserialize data.

    It keeps the same interface as karabo_bridge.Client.
    """
    def __init__(self, endpoint, timeout=None):

        self._ctx = zmq.Context()

        self._socket = self._ctx.socket(zmq.REQ)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(endpoint)

        if timeout is not None:
            self._socket.setsockopt(zmq.RCVTIMEO, int(timeout * 1000))
        self._recv_ready = False

    def next(self):
        """Request next data container.

        This function call is blocking.

        :return: data, meta
        :rtype: dict, dict

        :raise TimeoutError: If timeout is reached before receiving data.
        """
        if not self._recv_ready:
            self._socket.send(b'')
            self._recv_ready = True

        try:
            msg = self._socket.recv(copy=False)
        except zmq.error.Again:
            raise TimeoutError(
                'No data received from {} in the last {} ms'.format(
                    self._socket.getsockopt_string(zmq.LAST_ENDPOINT),
                    self._socket.getsockopt(zmq.RCVTIMEO)))
        self._recv_ready = False
        return pickle.loads(msg)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._ctx.destroy(linger=0)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()


SlowData = namedtuple('SlowData', 'deviceid, key, value, timestamp, trainId')


# TODO: the implementation of KaraboGateClient is temporarily during the
#       experimental phase
class KaraboGateClient:
    """Karabo bridge client for Karabo pipeline data.
    This class can request data to a Karabo bridge server.
    Create the client with::
        from karabo_bridge import Client
        krb_client = Client("tcp://153.0.55.21:12345")
    then call ``data = krb_client.next()`` to request next available data
    container.
    Parameters
    ----------
    endpoint : str
        server socket you want to connect to (only support TCP socket).
    sock : str, optional
        socket type - supported: REQ, SUB.
    ser : str, DEPRECATED
        Serialization protocol to use to decode the incoming message (default
        is msgpack) - supported: msgpack.
    context : zmq.Context
        To run the Client's sockets using a provided ZeroMQ context.
    timeout : int
        Timeout on :method:`next` (in seconds)
        Data transfered at the EuXFEL for Mega-pixels detectors can be very
        large. Setting a too small timeout might end in never getting data.
        Some example of transfer timing for 1Mpix detector (AGIPD, LPD):
            32 pulses per train (125 MB): ~0.1 s
            128 pulses per train (500 MB): ~0.4 s
            350 pulses per train (1.37 GB): ~1 s
    Raises
    ------
    NotImplementedError
        if socket type or serialization algorythm is not supported.
    ZMQError
        if provided endpoint is not valid.
    """
    def __init__(self, endpoint, timeout=None, context=None, connect=True):
        # TODO
        # [ ] ask for devices
        # [ ] ask for keys/channels for device
        # [ ] server as MDL?
        #   [ ] handles several client connections
        #   [ ] spawn bound device per channel connection (client select shared/copy)

        self.dumps = msgpack.Packer(use_bin_type=True).pack
        self.loads = partial(msgpack.loads, raw=False,
                             max_bin_len=0x7fffffff)
        self.monitored = set()

        self.ctx = context or zmq.Context()
        self.request = self.ctx.socket(zmq.PAIR)
        self.request.SNDTIMEO = 5000
        self.request.RCVTIMEO = 5000
        self.request.setsockopt(zmq.LINGER, 0)
        self.request.connect(endpoint)

        self.data = None
        self.timeout = timeout
        self.connected = False
        self._hb = None

        if connect:
            self.connect()

    def connect(self):
        if self.connected:
            raise RuntimeError('Client is already connected')
        # receive socket interfaces for slow and pipeline data
        msg = self.ask({'request': 'hello', 'hostname': gethostname(),
                        'username': getuser(), 'timestamp': time()})
        self.data = self.ctx.socket(zmq.PULL)
        self.data.set_hwm(50)
        if self.timeout is not None:
            self.data.RCVTIMEO = int(1000 * self.timeout)
        self.data.connect(msg['data_addr'])
        self.data.connect(msg['pipe_addr'])
        self.ask({'request': 'hello', 'status': 'connected'})

        self.connected = True
        self._hb = Thread(target=self._heartbeat, daemon=True)
        self._hb.start()  # start pinging server

    def _heartbeat(self):
        while self.connected:
            sleep(10)
            try:
                self.ask({'request': 'ping'})
            except zmq.error.ZMQError:
                pass

    def ask(self, msg):
        try:
            self.request.send(self.dumps(msg))
        except zmq.error.Again:
            raise TimeoutError(f"Could not reach {self.request.LAST_ENDPOINT}")

        try:
            reply = self.loads(self.request.recv())
        except zmq.error.Again:
            raise TimeoutError(f'No reply from server')

        if reply['status'] == 'failure':
            raise RuntimeError(reply['error'])
        return reply

    def set_hwm(self):
        self.data.set_hwm(2*len(self.monitored))

    def next(self):
        """Request next data container.
        This function call is blocking.
        Returns
        -------
        data : dict
            The data for this train, keyed by source name.
        meta : dict
            The metadata for this train, keyed by source name.
            This dictionary is populated for protocol version 1.0 and 2.2.
            For other protocol versions, metadata information is available in
            `data` dict.
        Raises
        ------
        TimeoutError
            If timeout is reached before receiving data.
        """
        try:
            return deserialize(self.data.recv_multipart(copy=False))
        except zmq.error.Again:
            raise TimeoutError

    def monitor_property(self, device, key):
        """Start monitoring a device property.
        :device: device name
        :key: name of the property to monitor
        """
        self.ask({'request': 'add_property_monitor',
                  'device': device, 'key': key})
        self.monitored.add(f'{device}/{key}')
        self.set_hwm()

    def monitor_channel(self, device, channel):
        """Start monitoring an output channel
        :device: name of the device
        :channel: name of the output channel to monitor
        """
        self.ask({'request': 'add_channel_monitor',
                  'channel': f'{device}:{channel}'})
        self.monitored.add(f'{device}:{channel}')
        self.set_hwm()

    def forget_property(self, device, key):
        """Stop monitoring a device property

        :device: device name
        :key: name of the property to stop monitoring
        """
        self.ask({'request': 'remove_property_monitor',
                  'device': device, 'key': key})
        self.monitored.discard(f'{device}/{key}')
        self.set_hwm()

    def forget_channel(self, device, channel):
        self.ask({'request': 'remove_channel_monitor',
                  'channel': f'{device}:{channel}'})
        self.monitored.discard(f'{device}:{channel}')
        self.set_hwm()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.connected:
            self.ask({'request': 'bye'})
            self.ctx.destroy(linger=0)
            self.connected = False  # disables ping

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()
