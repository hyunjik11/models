# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A script to run training for sequential latent variable models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import runners

# Shared flags.
tf.app.flags.DEFINE_string("mode", "train",
                           "The mode of the binary. Must be 'train' or 'eval'.")
tf.app.flags.DEFINE_string("model", "vrnn",
                           "Model choice. Currently only 'vrnn' is supported.")
tf.app.flags.DEFINE_integer("latent_size", 64,
                            "The size of the latent state of the model.")
tf.app.flags.DEFINE_string("dataset_type", "pianoroll",
                           "The type of dataset, either 'pianoroll' or 'speech'.")
tf.app.flags.DEFINE_string("dataset_path", "",
                           "Path to load the dataset from.")
tf.app.flags.DEFINE_integer("data_dimension", None,
                            "The dimension of each vector in the data sequence. "
                            "Defaults to 88 for pianoroll datasets and 200 for speech "
                            "datasets. Should not need to be changed except for "
                            "testing.")
tf.app.flags.DEFINE_integer("batch_size", 4,
                            "Batch size.")
tf.app.flags.DEFINE_integer("num_samples", 4,
                           "The number of samples (or particles) for multisample "
                           "algorithms.")
tf.app.flags.DEFINE_string("logdir", "~/models/tmp/smc_vi",
                           "The directory to keep checkpoints and summaries in.")
tf.app.flags.DEFINE_integer("random_seed", None,
                            "A random seed for seeding the TensorFlow graph.")
tf.app.flags.DEFINE_string("train_path", "",
                           "Path to load the video training dataset from.")
tf.app.flags.DEFINE_string("valid_path", "",
                           "Path to load the video validation dataset from.")
tf.app.flags.DEFINE_integer("seq_len", 10,
                            "Initial length of sequences of training batches."
                            "Only relevant for mnist.")
tf.app.flags.DEFINE_integer("max_seq_len", 10,
                            "Max length of sequences for validation batches. "
                            "Only relevant for mnist.")
tf.app.flags.DEFINE_integer("valid_batch_size", 1000,
                            "Batch size for validation")
tf.app.flags.DEFINE_integer("sample_batch_size", 4,
                            "Batch size for sampling and reconstructions. "
                            "Use batch_size if it is smaller. ")
tf.app.flags.DEFINE_integer("stage_itr", 1000,
                            "Number of iterations between increments of seq_len by 1. "
                            "Only relevant for mnist.")
tf.app.flags.DEFINE_float("fixed_sigma", None,
                          "Fixed sigma of likelihood. Learned by default.")
tf.app.flags.DEFINE_integer("H", 50,
                          "Height of image.")
tf.app.flags.DEFINE_integer("W", 50,
                          "Width of image.")
tf.app.flags.DEFINE_integer("C", 1,
                          "Number of channels of image.")
tf.app.flags.DEFINE_string("activation_fn", "relu",
                           "Activation function for hidden layers of model.")
tf.app.flags.DEFINE_integer("num_hidden_units", None,
                           "Number of hidden units for the fc parts of model. "
                           "Defaults to latent_size.")
tf.app.flags.DEFINE_integer("num_hidden_layers", 1,
                           "Number of hidden layers for the fc parts of model.")
tf.app.flags.DEFINE_integer("gpu", 0,
                           "Index of GPU used.")

# Training flags.
tf.app.flags.DEFINE_string("bound", "fivo",
                           "The bound to optimize. Can be 'elbo', 'iwae', or 'fivo'.")
tf.app.flags.DEFINE_boolean("normalize_by_seq_len", False,
                            "If true, normalize the loss by the number of timesteps "
                            "per sequence.")
tf.app.flags.DEFINE_string("optimizer", "adam",
                           "Optimizer used for learning. Must be 'adam' or 'rmsprop'")
tf.app.flags.DEFINE_float("learning_rate", 0.00001,
                          "The learning rate for optimizer.")
tf.app.flags.DEFINE_float("momentum", 0.9,
                          "The momentum for RMSProp optimizer.")
tf.app.flags.DEFINE_integer("max_steps", int(1e9),
                            "The number of gradient update steps to train for.")
tf.app.flags.DEFINE_integer("summarize_every", 50,
                            "The number of steps between summaries.")

# Distributed training flags.
tf.app.flags.DEFINE_string("master", "",
                           "The BNS name of the TensorFlow master to use.")
tf.app.flags.DEFINE_integer("task", 0,
                            "Task id of the replica running the training.")
tf.app.flags.DEFINE_integer("ps_tasks", 0,
                            "Number of tasks in the ps job. If 0 no ps job is used.")
tf.app.flags.DEFINE_boolean("stagger_workers", True,
                            "If true, bring one worker online every 1000 steps.")

# Evaluation flags.
tf.app.flags.DEFINE_string("split", "train",
                           "Split to evaluate the model on. Can be 'train', 'valid', or 'test'.")
tf.app.flags.DEFINE_integer("num_eval", 1000,
                            "Number of data points for evaluation.")
FLAGS = tf.app.flags.FLAGS

PIANOROLL_DEFAULT_DATA_DIMENSION = 88
SPEECH_DEFAULT_DATA_DIMENSION = 200


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  if FLAGS.data_dimension is None: # not needed for video data
    if FLAGS.dataset_type == "pianoroll":
      FLAGS.data_dimension = PIANOROLL_DEFAULT_DATA_DIMENSION
    elif FLAGS.dataset_type == "speech":
      FLAGS.data_dimension = SPEECH_DEFAULT_DATA_DIMENSION
  if FLAGS.mode == "train":
    runners.run_train(FLAGS)
  elif FLAGS.mode == "eval":
    runners.run_eval(FLAGS)

if __name__ == "__main__":
  tf.app.run()
