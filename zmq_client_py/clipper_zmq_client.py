import zmq
# import numpy as np
import struct
# import time
from datetime import datetime
# import sys

from threading import Lock, Thread
from Queue import Queue

DATA_TYPE_BYTES = 0
DATA_TYPE_INTS = 1
DATA_TYPE_FLOATS = 2
DATA_TYPE_DOUBLES = 3
DATA_TYPE_STRINGS = 4

NUM_REQUESTS_SEND = 10
NUM_RESPONSES_RECV = 10

BYTES_PER_INT = 4
BYTES_PER_FLOAT = 4
BYTES_PER_BYTE = 1
BYTES_PER_CHAR = 1


class Client:
    active = False

    def __init__(self, clipper_host, clipper_port):
        self.clipper_host = clipper_host
        self.clipper_port = clipper_port
        self.request_queue = Queue()
        self.recv_count = 0

    def start(self):
        global active
        active = True
        self.thread = Thread(target=self._run, args=[])
        self.start_time = datetime.now()
        self.thread.start()

    def stop(self):
        global active
        if active:
            active = False
            self.thread.join()

    def send_request(self, app_name, input_item):
        self.request_queue.put((app_name, input_item))

    def _run(self):
        global active
        clipper_address = "tcp://{0}:{1}".format(self.clipper_host, self.clipper_port)
        context = zmq.Context()
        socket = context.socket(zmq.DEALER)
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)

        socket.connect(clipper_address)
        while active:
            if self.request_queue.empty():
                receivable_sockets = dict(poller.poll(1000))
                if socket in receivable_sockets and receivable_sockets[socket] == zmq.POLLIN:
                    self._receive_response(socket)
                    for i in range(NUM_REQUESTS_SEND - 1):
                        receivable_sockets = dict(poller.poll(0))
                        if socket in receivable_sockets and receivable_sockets[socket] == zmq.POLLIN:
                            self._receive_response(socket)
            else:
                for i in range(NUM_RESPONSES_RECV):
                    receivable_sockets = dict(poller.poll(1000))
                    if socket in receivable_sockets and receivable_sockets[socket] == zmq.POLLIN:
                        self._receive_response(socket)

            self._send_string_requests(socket)

    def _receive_response(self, socket):
        # Receive delimiter between routing identity and content
        socket.recv()
        data_type_bytes = socket.recv()
        output_data = socket.recv()
        self.recv_count += 1
        if self.recv_count % 200 == 0:
            curr_time = datetime.now()
            latency = (curr_time - self.start_time).total_seconds()
            print("Throughput: {} qps\n".format(200.0 / latency))
            self.start_time = curr_time

    def _send_string_requests(self, socket):
        i = NUM_REQUESTS_SEND
        while (not self.request_queue.empty()) and i > 0:
            # Create message buffer
            app_name, input_item = self.request_queue.get()
            num_inputs = 1
            input_len = len(input_item)
            input_buffer_size = BYTES_PER_INT + BYTES_PER_INT * num_inputs + input_len
            input_buffer = bytearray(input_buffer_size)
            memview = memoryview(input_buffer)
            struct.pack_into("<I", input_buffer, 0, num_inputs)
            content_end_position = BYTES_PER_INT + (BYTES_PER_INT * num_inputs)
            current_input_sizes_position = BYTES_PER_INT
            struct.pack_into("<I", input_buffer, current_input_sizes_position, input_len)
            current_input_sizes_position += BYTES_PER_INT
            memview[content_end_position: content_end_position + input_len] = input_item
            content_end_position += input_len
            print("Content end position: {}, buffer size: {}".format(
                content_end_position, input_buffer_size))

            socket.send("", zmq.SNDMORE)
            socket.send_string(app_name, zmq.SNDMORE)
            socket.send(struct.pack("<I", DATA_TYPE_STRINGS), zmq.SNDMORE)
            socket.send(struct.pack("<I", content_end_position), zmq.SNDMORE)
            socket.send(input_buffer[0:content_end_position])
            i -= 1
