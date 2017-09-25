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

input_type = "strings"
app_name = "text-driver-1"
model_name = "sentiment-analysis"


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
        print(response)
    for r in reviews:
        client.send_request(app_name, r, req_callback)
        time.sleep(5)


def setup_clipper():
    cl = ClipperConnection(DockerContainerManager(redis_port=6380))
    cl.start_clipper(query_frontend_image="clipper/zmq_frontend:develop")
    time.sleep(10)
    cl.register_application(name=app_name,
                            default_output="TIMEOUT",
                            slo_micros=1000000,
                            input_type="strings")
    cl.deploy_model(name=model_name,
                    version=1,
                    image="model-comp/theano-lstm",
                    input_type="strings",
                    num_replicas=2,
                    gpus=[0, 1])
    time.sleep(30)
    cl.link_model_to_app(app_name=app_name, model_name=model_name)
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
