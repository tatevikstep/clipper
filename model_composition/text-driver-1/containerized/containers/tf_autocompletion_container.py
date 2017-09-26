from __future__ import print_function
import sys
import os
import rpc

import tensorflow as tf
import google.protobuf as pb
import numpy as np

import lm1b_data_utils as data_utils

# Sampling parameters
MAX_WORD_LEN = 50
MAX_SAMPLE_WORDS = 100
BATCH_SIZE = 1
NUM_TIMESTEPS = 1

NUM_AUTOCOMPLETION_WORDS = 6

GPU_MEM_FRAC = .9

class AutocompletionModelContainer(rpc.ModelContainerBase):

	def __init__(self, vocab_path, graph_def_path, checkpoint_dir_path):
		self.vocab = data_utils.CharsVocabulary(vocab_path, MAX_WORD_LEN)
		self.sess, self.tf_layers = self._load_model(graph_def_path, checkpoint_dir_path)

	def predict_strings(self, inputs):
		# Only render predictions based on the last five words of the input text.
		# Larger prefixes result in substantial increases in computation time.
		input_last_fives = [input_text[-5:] for input_text in inputs]
		completions = self._predict_words(input_last_fives, self.vocab)
		return ["{} {}".format(inputs[i], completions[i]) for i in range(0, len(inputs))]

	def _load_model(self, graph_def_path, ckpt_dir_path):
		with tf.device("/gpu:0"):
			sys.stderr.write('Recovering graph.\n')
			with tf.gfile.FastGFile(graph_def_path, 'r') as f:
				s = f.read().decode()
				gd = tf.GraphDef()
				pb.text_format.Merge(s, gd)

				tf.logging.info('Recovering Graph %s', graph_def_path)
				t = {}
				[
					t['states_init'], t['lstm/lstm_0/control_dependency'],
					t['lstm/lstm_1/control_dependency'], t['softmax_out'], t['class_ids_out'],
					t['class_weights_out'], t['log_perplexity_out'], t['inputs_in'],
					t['targets_in'], t['target_weights_in'], t['char_inputs_in'],
					t['all_embs'], t['softmax_weights'], t['global_step']
				] = tf.import_graph_def(gd, {}, [
					'states_init',
					'lstm/lstm_0/control_dependency:0',
					'lstm/lstm_1/control_dependency:0',
					'softmax_out:0',
					'class_ids_out:0',
					'class_weights_out:0',
					'log_perplexity_out:0',
					'inputs_in:0',
					'targets_in:0',
					'target_weights_in:0',
					'char_inputs_in:0',
					'all_embs_out:0',
					'Reshape_3:0',
					'global_step:0'
				], name='')

				sys.stderr.write('Recovering checkpoint %s\n' % ckpt_dir_path)
				gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=GPU_MEM_FRAC)
				sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
				sess.run('save/restore_all', {'save/Const:0': ckpt_dir_path})
				sess.run(t['states_init'])

		return sess, t

	def _predict_words(self, prefix_sentences, vocab):
		"""
		Parameters
		----------
		prefix_sentences: list
			A list of sentences that need to be autocompleted
		vocab: Vocabulary
			Class used for numerical word mapping and related utilities

		Returns
		----------
		list
			A list of autocompleted sentences
		"""
		batch_size = len(prefix_sentences)

		targets = np.zeros([batch_size, NUM_TIMESTEPS], np.int32)
		weights = np.ones([batch_size, NUM_TIMESTEPS], np.float32)

		sentence_word_lengths = [len(prefix_sentence.split()) for prefix_sentence in prefix_sentences]

		parsed_sentences = []
		for i in range(0, len(prefix_sentences)):
			prefix_sentence = prefix_sentences[i]
			if prefix_sentence.find('<S>') != 0:
				prefix_sentence = '<S> ' + prefix_sentence
			prefix_words_list = prefix_sentence.split()
			prefix_word_ids = [vocab.word_to_id(w) for w in prefix_words_list]
			prefix_char_ids = [vocab.word_to_char_ids(w) for w in prefix_words_list]

			parsed_sentences.append([prefix_word_ids, prefix_char_ids])

		inputs = np.zeros([batch_size, NUM_TIMESTEPS], np.int32)
		char_ids_inputs = np.zeros([batch_size, NUM_TIMESTEPS, vocab.max_word_length], np.int32)

		processing_indices = range(0, batch_size)
		autocompleted_sentences = ['' for _ in prefix_sentences]
		while True:
			inputs[:, 0] = [parsed_sentence[0][0] for parsed_sentence in parsed_sentences]
			char_ids_inputs[:, 0, :] = [parsed_sentence[1][0] for parsed_sentence in parsed_sentences]
			for i in processing_indices:
				parsed_sentences[i][0] = parsed_sentences[i][0][1:]
				parsed_sentences[i][1] = parsed_sentences[i][1][1:]

			softmaxes = [
				self._run_prediction_session(
					np.array([inputs[i]]), 
					np.array([char_ids_inputs[i]]), 
					np.array([targets[i]]), 
					np.array([weights[i]])) 
				for i in processing_indices]

			predicted_word_ids = [self._sample_softmax(softmax) for softmax in softmaxes]
			predicted_char_ids = [vocab.word_to_char_ids(vocab.id_to_word(word_id)) for word_id in predicted_word_ids]

			softmax_index = 0
			for i in processing_indices:
				if not parsed_sentences[i][0]:
					parsed_sentences[i][0] = [predicted_word_ids[softmax_index]]
					parsed_sentences[i][1] = [predicted_char_ids[softmax_index]]
					softmax_index += 1

			for i in processing_indices:
				new_word = vocab.id_to_word(parsed_sentences[i][0][0])
				if new_word == '<S>':
					processing_indices.remove(i)
				else:
					new_sentence = autocompleted_sentences[i] + new_word + ' '
					autocompleted_sentences[i] = new_sentence
					if len(new_sentence.split()) >= sentence_word_lengths[i] + NUM_AUTOCOMPLETION_WORDS:
						processing_indices.remove(i)

			if len(processing_indices) == 0:
				break

		return autocompleted_sentences

	def _run_prediction_session(self, word_ids_input, char_ids_input, targets, weights):
		softmaxes = self.sess.run(self.tf_layers['softmax_out'],
						feed_dict={
									self.tf_layers['char_inputs_in']: char_ids_input,
									self.tf_layers['inputs_in']: word_ids_input,
									self.tf_layers['targets_in']: targets,
									self.tf_layers['target_weights_in']: weights})
		# Due to limitations of the open source lm_1b model, only one input can be processed
		# at a time, so softmaxes is a 1-element list. We take the element at index zero
		return softmaxes[0]

	def _sample_softmax(self, softmax):
	  return min(np.sum(np.cumsum(softmax) < np.random.rand()), len(softmax) - 1)

if __name__ == "__main__":
    print("Starting Lm_1b Autocompletion Container")
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
        model_vocab_path = os.environ["CLIPPER_MODEL_VOCAB_PATH"]
    except KeyError:
        print(
            "ERROR: CLIPPER_MODEL_VOCAB_PATH environment variable must be set",
            file=sys.stdout)
        sys.exit(1)
    try:
        model_graph_path = os.environ["CLIPPER_MODEL_GRAPH_PATH"]
    except KeyError:
        print(
            "ERROR: CLIPPER_MODEL_GRAPH_PATH environment variable must be set",
            file=sys.stdout)
        sys.exit(1)
    try:
        model_checkpoint_path = os.environ["CLIPPER_MODEL_CHECKPOINT_PATH"]
    except KeyError:
        print(
            "ERROR: CLIPPER_MODEL_CHECKPOINT_PATH environment variable must be set",
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
    container = AutocompletionModelContainer(model_vocab_path, model_graph_path, model_checkpoint_path)
    rpc_service = rpc.RPCService()
    rpc_service.start(container, ip, port, model_name, model_version,
                      input_type)