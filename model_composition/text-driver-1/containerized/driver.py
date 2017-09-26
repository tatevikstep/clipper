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
from datetime import datetime
import argparse

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
    predictor = Predictor()
    reviews = load_reviews()
    logger.info("Loaded {} reviews".format(len(reviews)))

    for r in reviews[20:30]:
        logger.info("sending prediction")
        predictor.predict(r)
        time.sleep(5)


class InflightReq(object):

    def __init__(self):
        self.raw_sentiment = False
        # self.raw_sentiment = True
        self.auto_compl_sentiment = False
        self.start_time = datetime.now()

    def raw_complete(self):
        self.raw_sentiment = True
        return self.check_complete()

    def auto_compl_complete(self):
        self.auto_compl_sentiment = True
        return self.check_complete()

    def check_complete(self):
        if self.auto_compl_sentiment and self.raw_sentiment:
            self.latency = (datetime.now() - self.start_time).total_seconds()
            logger.info("Completed in {} seconds".format(self.latency))
            return True
        else:
            return False


class Predictor(object):

    def __init__(self):
        self.outstanding_reqs = {}
        self.client = Client("localhost", 4456, 4455)
        self.client.start()
        self.latencies = []
        self.num_complete = 0
        self.cur_req_id = 0

    def predict(self, review):
        self.outstanding_reqs[self.cur_req_id] = InflightReq()
        self.get_raw_sentiment(self.cur_req_id, review)
        self.get_auto_compl_sentiment(self.cur_req_id, review)
        self.cur_req_id += 1

    def get_raw_sentiment(self, req_id, review):
        def callback(response):
            logger.info("received raw sentiment")
            if self.outstanding_reqs[req_id].raw_complete():
                self.latencies.append(self.outstanding_reqs[req_id].latency)
                self.num_complete += 1
                del self.outstanding_reqs[req_id]

        logger.info("getting raw sentiment")
        self.client.send_request("theano-lstm", review, callback)

    def get_auto_compl_sentiment(self, req_id, review):
        def sentiment_callback(response):
            logger.info("received auto compl sentiment: {}".format(response))
            if self.outstanding_reqs[req_id].auto_compl_complete():
                self.latencies.append(self.outstanding_reqs[req_id].latency)
                self.num_complete += 1
                del self.outstanding_reqs[req_id]

        def auto_compl_callback(response):
            logger.info("received completion, requesting sentiment. Response: {}".format(response))
            self.client.send_request("theano-lstm", response, sentiment_callback)

        logger.info("requesting completion")
        self.client.send_request("tf-autocomplete", review[:len(review)/2], auto_compl_callback)


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
    setup_heavy_node(cl, "tf-autocomplete", "strings", "model-comp/tf-autocomplete", slo=10000000, gpus=[1])
    time.sleep(10)
    logger.info("Clipper is set up")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Text-Driver-1')
    parser.add_argument('--setup', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--num_procs', type=int, default=1)
    args = parser.parse_args()
    if args.setup:
        setup_clipper()
    if args.run:
        processes = []
        for i in range(args.num_procs):
            p = Process(target=run, args=('%d'.format(i),))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
