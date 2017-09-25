from __future__ import print_function
import sys
import os
import rpc

import pickle
import re
import numpy as np
import lstm_utils, imdb_utils

class MovieSentimentContainer(rpc.ModelContainerBase):

    def __init__(self, model_path, options_path, imdb_pkl_dict_path):
        model_options_file = open(options_path, "r")
        model_options = {}
        for line in model_options_file.readlines():
            key, value = line.split('@')
            value = value.strip()
            try:
                value = int(value)
            except ValueError as e:
                pass
            model_options[key] = value

        model_options_file.close()

        self.n_words = model_options['n_words']
        params = lstm_utils.init_params(model_options)
        params = lstm_utils.load_params(model_path, params)
        tparams = lstm_utils.init_tparams(params)

        (use_noise, x, mask, y, f_pred_prob, f_pred, cost) = lstm_utils.build_model(tparams, model_options)
        self.predict_function = f_pred

        dict_file = open(imdb_pkl_dict_path, 'rb')
        self.imdb_dict = pickle.load(dict_file)
        dict_file.close()
        print("Done!")

    def predict_strings(self, inputs):
        reviews_features = self._get_reviews_features(inputs)
        x, mask, y = self._prepare_reviews_data(reviews_features)
        predictions = self.predict_function(x, mask)
        pred_strs = [str(pred) for pred in predictions]
        print(pred_strs)
        return pred_strs

    def _get_reviews_features(self, reviews):
        all_review_indices = []
        for review in reviews:
            review_indices = self._get_imdb_indices(review)
            all_review_indices.append(review_indices)
        return all_review_indices

    def _get_imdb_indices(self, input_str):
        split_input = input_str.split(" ")
        indices = np.ones(len(split_input))
        for i in range(0, len(split_input)):
            term = split_input[i]
            term = re.sub('[^a-zA-Z\d\s:]', '', term)
            if term in self.imdb_dict:
                index = self.imdb_dict[term]
                if index < self.n_words:
                    indices[i] = index
        return indices

    def _prepare_reviews_data(self, reviews_features):
        x, mask, y = imdb_utils.prepare_data(reviews_features, [], maxlen=None)
        return x, mask, y

if __name__ == "__main__":
    print("Starting Theano Sentiment Analysis Container")
    try:
        model_name = os.environ["CLIPPER_MODEL_NAME"]
    except KeyError:
        print(
            "ERROR: CLIPPER_MODEL_NAME environment variable must be set",
            file=sys.stdout)
        sys.exit(1)
    try:
        model_version = os.environ["CLIPPER_MODEL_VERSION"]
    except KeyError:
        print(
            "ERROR: CLIPPER_MODEL_VERSION environment variable must be set",
            file=sys.stdout)
        sys.exit(1)
    try:
        model_path = os.environ["CLIPPER_MODEL_PATH"]
    except KeyError:
        print(
            "ERROR: CLIPPER_MODEL_PATH environment variable must be set",
            file=sys.stdout)
        sys.exit(1)
    try:
        model_options_path = os.environ["CLIPPER_MODEL_OPTIONS_PATH"]
    except KeyError:
        print(
            "ERROR: CLIPPER_MODEL_OPTIONS_PATH environment variable must be set",
            file=sys.stdout)
        sys.exit(1)
    try:
        model_dict_path = os.environ["CLIPPER_MODEL_DICT_PATH"]
    except KeyError:
        print(
            "ERROR: CLIPPER_MODEL_DICT_PATH environment variable must be set",
            file=sys.stdout)
        sys.exit(1)

    ip = "127.0.0.1"
    if "CLIPPER_IP" in os.environ:
        ip = os.environ["CLIPPER_IP"]
    else:
        print("Connecting to Clipper on localhost")

    print("CLIPPER IP: {}".format(ip))

    port = 7000
    if "CLIPPER_PORT" in os.environ:
        port = int(os.environ["CLIPPER_PORT"])
    else:
        print("Connecting to Clipper with default port: 7000")

    input_type = "strings"
    container = MovieSentimentContainer(model_path, model_options_path, model_dict_path)
    rpc_service = rpc.RPCService()
    rpc_service.start(container, ip, port, model_name, model_version,
                      input_type)
