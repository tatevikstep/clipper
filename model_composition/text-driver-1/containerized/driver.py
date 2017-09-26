from __future__ import print_function

import sys
import os
import logging
import numpy as np
import time
from clipper_admin import ClipperConnection, DockerContainerManager
# from datetime import datetime
from multiprocessing import Process
from zmq_client import Client

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%y-%m-%d:%H:%M:%S',
    level=logging.INFO)

logger = logging.getLogger(__name__)


DATA_TYPE_BYTES = 0
DATA_TYPE_INTS = 1
DATA_TYPE_FLOATS = 2
DATA_TYPE_DOUBLES = 3
DATA_TYPE_STRINGS = 4

CIFAR_SIZE_DOUBLES = (299 * 299 * 3) / 2


def load_reviews():
    base_path = "/home/ubuntu/clipper/model_composition/text-driver-1/workload_data/aclImdb/test/"
    reviews = []
    pos_path = os.path.join(base_path, "pos")
    for rev_file in os.listdir(pos_path):
        with open(os.path.join(pos_path, rev_file), "r") as f:
            reviews.append(f.read().strip())

    neg_path = os.path.join(base_path, "neg")
    for rev_file in os.listdir(neg_path):
        with open(os.path.join(neg_path, rev_file), "r") as f:
            reviews.append(f.read().strip())
    # Shuffle in place
    np.random.shuffle(reviews)
    return reviews


def run(proc_num):
    """
    Note: Throughput logging is performed by the ZMQ Frontend Client
    (clipper_zmq_client.py)
    """
    client = Client("localhost", 4456, 4455)
    client.start()
    reviews = load_reviews()
    logger.info("Loaded {} reviews".format(len(reviews)))

    def req_callback(response):
        print("LSTM response: {}".format(response))
    for r in reviews:
        client.send_request("theano-lstm", r, req_callback)
        time.sleep(5)


def setup_heavy_node(clipper_conn,
                     name,
                     input_type,
                     model_image,
                     slo=1000000,
                     num_replicas=1,
                     gpus=None):
    clipper_conn.register_application(name=name,
                                      default_output="TIMEOUT",
                                      slo_micros=slo,
                                      input_type=input_type)

    clipper_conn.deploy_model(name=name,
                              version=1,
                              image=model_image,
                              input_type=input_type,
                              num_replicas=num_replicas,
                              gpus=gpus)

    clipper_conn.link_model_to_app(app_name=name, model_name=name)


def setup_clipper():
    cl = ClipperConnection(DockerContainerManager(redis_port=6380))
    cl.start_clipper(query_frontend_image="clipper/zmq_frontend:develop")
    time.sleep(10)
    setup_heavy_node(cl, "theano-lstm", "strings", "model-comp/theano-lstm", gpus=[0])
    setup_heavy_node(cl, "tf-autocomplete", "strings", "model-comp/tf-autocomplete", gpus=[1])
    time.sleep(30)
    logger.info("Clipper is set up")
    time.sleep(15)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python zmq_benchmark.py <NUM_PROCS>")
        sys.exit(1)

    num_procs = int(sys.argv[1])

    setup_clipper()

    processes = []

    for i in range(num_procs):
        p = Process(target=run, args=('%d'.format(i),))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
