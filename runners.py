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

"""High-level code for creating and running FIVO-related Tensorflow graphs.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import time

import numpy as np
import tensorflow as tf

import bounds
import utils
from data import datasets
from models import vrnn

from seq_air.experiment_tools import print_num_params, print_variables_by_scope

def create_dataset_and_model(config, split, shuffle, repeat):
  """Creates the dataset and model for a given config.

  Args:
    config: A configuration object with config values accessible as properties.
      Most likely a FLAGS object. This function expects the properties
      batch_size, dataset_path, dataset_type, and latent_size to be defined.
    split: The dataset split to load.
    shuffle: If true, shuffle the dataset randomly.
    repeat: If true, repeat the dataset endlessly.
  Returns:
    inputs: A batch of input sequences represented as a dense Tensor of shape
      [time, batch_size, data_dimension].
    targets: A batch of target sequences represented as a dense Tensor of
      shape [time, batch_size, data_dimension].
    lens: An int Tensor of shape [batch_size] representing the lengths of each
      sequence in the batch.
    model: A vrnn.VRNNCell model object.
  """
  if config.dataset_type == "pianoroll":
    inputs, targets, lengths, mean = datasets.create_pianoroll_dataset(
        config.dataset_path, split, config.batch_size, shuffle=shuffle,
        repeat=repeat)
    # Convert the mean of the training set to logit space so it can be used to
    # initialize the bias of the generative distribution.
    generative_bias_init = -tf.log(
        1. / tf.clip_by_value(mean, 0.0001, 0.9999) - 1)
    generative_distribution_class = vrnn.ConditionalBernoulliDistribution
  elif config.dataset_type == "speech":
    inputs, targets, lengths = datasets.create_speech_dataset(
        config.dataset_path, config.batch_size,
        samples_per_timestep=config.data_dimension, prefetch_buffer_size=1,
        shuffle=False, repeat=False)
    generative_bias_init = None
    generative_distribution_class = vrnn.ConditionalNormalDistribution
  elif config.dataset_type == "mnist":
    inputs, targets, lengths, mean_image = datasets.create_mnist_dataset(config.train_path,
        config.valid_path, split, config.batch_size, config.seq_len, config.stage_itr)
    generative_bias_init = None
    generative_distribution_class = vrnn.ConditionalNormalDistribution

  # set architecture for model
  if config.activation_fn == 'relu':
    activation_fn = tf.nn.relu
  elif config.activation_fn == 'elu':
    activation_fn = tf.nn.elu

  if config.num_hidden_units is None:
    fcnet_hidden_sizes = None
  else:
    fcnet_hidden_sizes = [config.num_hidden_units]*config.num_hidden_layers    

  # whether to initialise decoder with mean image
  if config.mean_image_init:
    mean_init = mean_image
  else:
    mean_init = None

  model = vrnn.create_vrnn(inputs.get_shape().as_list()[2],
                           config.latent_size,
                           generative_distribution_class,
                           generative_bias_init=generative_bias_init,
                           raw_sigma_bias=0.5,
                           lkhd_fixed_sigma=config.fixed_sigma,
                           hidden_activation_fn=activation_fn,
                           fcnet_hidden_sizes=fcnet_hidden_sizes,
                           mean_init=mean_init)
  return inputs, targets, lengths, model


def restore_checkpoint_if_exists(saver, sess, logdir):
  """Looks for a checkpoint and restores the session from it if found.

  Args:
    saver: A tf.train.Saver for restoring the session.
    sess: A TensorFlow session.
    logdir: The directory to look for checkpoints in.
  Returns:
    True if a checkpoint was found and restored, False otherwise.
  """
  checkpoint = tf.train.get_checkpoint_state(logdir)
  if checkpoint:
    checkpoint_name = os.path.basename(checkpoint.model_checkpoint_path)
    full_checkpoint_path = os.path.join(logdir, checkpoint_name)
    saver.restore(sess, full_checkpoint_path)
    return True
  return False


def wait_for_checkpoint(saver, sess, logdir):
  """Loops until the session is restored from a checkpoint in logdir.

  Args:
    saver: A tf.train.Saver for restoring the session.
    sess: A TensorFlow session.
    logdir: The directory to look for checkpoints in.
  """
  while True:
    if restore_checkpoint_if_exists(saver, sess, logdir):
      break
    else:
      tf.logging.info("Checkpoint not found in %s, sleeping for 60 seconds."
                      % logdir)
      time.sleep(60)


def run_train(config):
  """Runs training for a sequential latent variable model.

  Args:
    config: A configuration object with config values accessible as properties.
      Most likely a FLAGS object. For a list of expected properties and their
      meaning see the flags defined in fivo.py.
  """

  def create_logging_hook(step, bound_value, cur_seq_len, sigma, elbo, iwae, 
                          fivo, kl, rec):
    """Creates a logging hook that prints the bound value periodically."""
    bound_label = "train " + config.bound
    if config.normalize_by_seq_len:
      bound_label += " per timestep"
    else:
      bound_label += " per sequence"
    def summary_formatter(log_dict):
      return ("Step %d, Seq Len %d, lkhd sigma %.3f, %s: %.1f, test elbo: %.1f, iwae: %.1f, fivo: %.1f, kl:%.1f, rec:%.1f") % (
          log_dict["step"], log_dict["cur_seq_len"], log_dict["sigma"], 
          bound_label, log_dict["bound_value"], log_dict["elbo"], log_dict["iwae"], 
          log_dict["fivo"], log_dict["kl"], log_dict["rec"])
    logging_hook = tf.train.LoggingTensorHook(
        {"step": step, "bound_value": bound_value, "cur_seq_len": cur_seq_len, 
         "sigma": sigma, "elbo":elbo, "iwae":iwae, "fivo":fivo, "kl":kl, "rec":rec},
        every_n_iter=config.summarize_every,
        formatter=summary_formatter)
    return logging_hook

  def create_loss():
    """Creates the loss to be optimized, and all logged quantities.

    Returns:
      bound: A float Tensor containing the value of the bound that is
        being optimized.
      loss: A float Tensor that when differentiated yields the gradients
        to apply to the model. Should be optimized via gradient descent.
    """
    inputs, targets, lengths, model = create_dataset_and_model(
        config, split="train", shuffle=True, repeat=True)
    # Compute lower bounds on the log likelihood.
    if config.bound == "elbo":
      ll_per_seq, _, _, _ = bounds.iwae(
          model, (inputs, targets), lengths, num_samples=1)
    elif config.bound == "iwae":
      ll_per_seq, _, _, _ = bounds.iwae(
          model, (inputs, targets), lengths, num_samples=config.num_samples)
    elif config.bound == "fivo":
      ll_per_seq, _, _, _, _ = bounds.fivo(
          model, (inputs, targets), lengths, num_samples=config.num_samples,
          resampling_criterion=bounds.ess_criterion)
    # Compute loss scaled by number of timesteps.
    ll_per_t = tf.reduce_mean(ll_per_seq / tf.to_float(lengths))
    ll_per_seq = tf.reduce_mean(ll_per_seq)
    # Obtain sequence length
    cur_seq_len = lengths[0]
    # Obtain sigma of likelihood 
    if config.fixed_sigma is None:
        sigma_min = model.generative.sigma_min
        raw_sigma_bias = model.generative.raw_sigma_bias
        with tf.variable_scope("vrnn/decoder", reuse=tf.AUTO_REUSE):
            lkhd_preproc_sigma = tf.get_variable("lkhd_preproc_sigma")
        lkhd_sigma = tf.maximum(tf.nn.softplus(lkhd_preproc_sigma + raw_sigma_bias),
                                sigma_min)
    else:
        lkhd_sigma = tf.constant(config.fixed_sigma)

    # Obtain validation data
    valid_config = config
    valid_config.seq_len = config.max_seq_len
    valid_config.stage_itr = 0
    valid_config.batch_size = config.valid_batch_size
    valid_inputs, valid_targets, valid_lengths, _ = create_dataset_and_model(
        config, split="valid", shuffle=True, repeat=False)
    
    # Compute lower bounds, kl and reconstruction lkhd on the test data.
    elbo_test_per_seq, kl_test_per_seq, _, _ = bounds.iwae(
          model, (valid_inputs, valid_targets), valid_lengths, num_samples=1)
    iwae_test_per_seq, _, _, _ = bounds.iwae(
          model, (valid_inputs, valid_targets), valid_lengths, num_samples=config.num_samples)
    fivo_test_per_seq, _, _, _, _ = bounds.fivo(
          model, (valid_inputs, valid_targets), valid_lengths, num_samples=config.num_samples,
          resampling_criterion=bounds.ess_criterion)

    # Take mean of all quantities across batch
    elbo_test_per_seq = tf.reduce_mean(elbo_test_per_seq)
    iwae_test_per_seq = tf.reduce_mean(iwae_test_per_seq)   
    fivo_test_per_seq = tf.reduce_mean(fivo_test_per_seq)
    kl_test_per_seq = tf.reduce_mean(kl_test_per_seq)
    rec_test_per_seq = elbo_test_per_seq + kl_test_per_seq
    
    tf.summary.scalar("test_elbo_per_seq", elbo_test_per_seq)
    tf.summary.scalar("test_iwae_per_seq", iwae_test_per_seq)
    tf.summary.scalar("test_fivo_per_seq", fivo_test_per_seq)
    tf.summary.scalar("test_kl_per_seq", kl_test_per_seq)
    tf.summary.scalar("test_rec_per_seq", rec_test_per_seq)
    tf.summary.scalar("train_ll_per_seq", ll_per_seq)
    tf.summary.scalar("train_ll_per_t", ll_per_t)
    tf.summary.scalar("cur_seq_len", cur_seq_len)
    tf.summary.scalar("lkhd_sigma", lkhd_sigma)

    # Compute reconstructions and samples from validation data.
    sample_size = min(config.batch_size, config.sample_batch_size)
    subset_inputs = valid_inputs[:,0:sample_size,:]
    subset_targets = valid_targets[:,0:sample_size,:]
    subset_lengths = valid_lengths[0:sample_size]
    originals, reconstructions = utils.reconstruct(model, (subset_inputs, subset_targets),
                                                   subset_lengths)
    seq_len, batch_size, ndims = originals.get_shape().as_list()
    samples = utils.sample(model, seq_len, ndims, num_samples=sample_size)

    # reshape images to be suitable arguments for tf.summary.image
    def preprocess(batch, seq_len, batch_size, ndims):
      """Reshape image from [seq_len, batch_size, ndims] to [batch_size*seq_len, H, W, C]
         for appropriate input to tf.summary.image(). Batch shape must be known apriori.
      """
      H, W, C = [config.H, config.W, config.C]
      assert ndims == H * W * C
      batch = tf.transpose(batch, [1, 0, 2]) # [batch_size, seq_len, ndims]
      return tf.reshape(batch, shape=[batch_size * seq_len, H, W, C])
    originals = preprocess(originals, seq_len, batch_size, ndims)
    reconstructions = preprocess(reconstructions, seq_len, batch_size, ndims)
    samples = preprocess(samples, seq_len, batch_size, ndims)
    tf.summary.image("originals", originals, max_outputs=(batch_size*seq_len))
    tf.summary.image("reconstructions", reconstructions, max_outputs=(batch_size*seq_len))
    tf.summary.image("samples", samples, max_outputs=(batch_size*seq_len))

    if config.normalize_by_seq_len:
      return (ll_per_t, -ll_per_t, cur_seq_len, lkhd_sigma, elbo_test_per_seq, 
              iwae_test_per_seq, fivo_test_per_seq, kl_test_per_seq, rec_test_per_seq)
    else:
      return (ll_per_seq, -ll_per_seq, cur_seq_len, lkhd_sigma, elbo_test_per_seq, 
              iwae_test_per_seq, fivo_test_per_seq, kl_test_per_seq, rec_test_per_seq)

  def create_graph():
    """Creates the training graph."""
    global_step = tf.train.get_or_create_global_step()
    (bound, loss, cur_seq_len, lkhd_sigma, elbo_test_per_seq, iwae_test_per_seq,
     fivo_test_per_seq, kl_test_per_seq, rec_test_per_seq)  = create_loss()
    if config.optimizer == 'adam':
        opt = tf.train.AdamOptimizer(config.learning_rate)
    elif config.optimizer == 'rmsprop':
        opt = tf.train.RMSPropOptimizer(config.learning_rate, momentum=config.momentum)
    grads = opt.compute_gradients(loss, var_list=tf.trainable_variables())    
    train_op = opt.apply_gradients(grads, global_step=global_step)
    return (bound, train_op, global_step, cur_seq_len, lkhd_sigma, elbo_test_per_seq, 
            iwae_test_per_seq, fivo_test_per_seq, kl_test_per_seq, rec_test_per_seq)

  device = tf.train.replica_device_setter(ps_tasks=config.ps_tasks)
  gpu_config = tf.ConfigProto(); gpu_config.gpu_options.visible_device_list = str(config.gpu) 
  with tf.Graph().as_default():
    if config.random_seed: tf.set_random_seed(config.random_seed)
    with tf.device(device):
      (bound, train_op, global_step, cur_seq_len, lkhd_sigma, elbo_test_per_seq, 
       iwae_test_per_seq, fivo_test_per_seq, kl_test_per_seq, rec_test_per_seq) = create_graph()
      print_variables_by_scope()
      print_num_params()
      log_hook = create_logging_hook(global_step, bound, cur_seq_len, lkhd_sigma, elbo_test_per_seq, 
       iwae_test_per_seq, fivo_test_per_seq, kl_test_per_seq, rec_test_per_seq)
      start_training = not config.stagger_workers
      with tf.train.MonitoredTrainingSession(
          config=gpu_config,
          master=config.master,
          is_chief=config.task == 0,
          hooks=[log_hook],
          checkpoint_dir=config.logdir,
          save_checkpoint_secs=3600,
          save_summaries_steps=config.summarize_every,
          log_step_count_steps=config.summarize_every) as sess:
        cur_step = -1
        while True:
          if sess.should_stop() or cur_step > config.max_steps: break
          if config.task > 0 and not start_training:
            cur_step = sess.run(global_step)
            tf.logging.info("task %d not active yet, sleeping at step %d" %
                            (config.task, cur_step))
            time.sleep(30)
            if cur_step >= config.task * 1000:
              start_training = True
          else:
            _, cur_step = sess.run([train_op, global_step])


def run_eval(config):
  """Runs evaluation for a sequential latent variable model.

  This method runs only one evaluation over the dataset, writes summaries to
  disk, and then terminates. It does not loop indefinitely.

  Args:
    config: A configuration object with config values accessible as properties.
      Most likely a FLAGS object. For a list of expected properties and their
      meaning see the flags defined in fivo.py.
  """

  def create_graph():
    """Creates the evaluation graph.

    Returns:
      lower_bounds: A tuple of float Tensors containing the values of the 3
        evidence lower bounds, summed across the batch.
      total_batch_length: The total number of timesteps in the batch, summed
        across batch examples.
      batch_size: The batch size.
      global_step: The global step the checkpoint was loaded from.
    """
    global_step = tf.train.get_or_create_global_step()
    inputs, targets, lengths, model = create_dataset_and_model(
        config, split=config.split, shuffle=False, repeat=False)
    # Compute lower bounds on the log likelihood.
    elbo_ll_per_seq, _, _, _ = bounds.iwae(
        model, (inputs, targets), lengths, num_samples=1)
    iwae_ll_per_seq, _, _, _ = bounds.iwae(
        model, (inputs, targets), lengths, num_samples=config.num_samples)
    fivo_ll_per_seq, _, _, _, _ = bounds.fivo(
        model, (inputs, targets), lengths, num_samples=config.num_samples,
        resampling_criterion=bounds.ess_criterion)
    elbo_ll = tf.reduce_sum(elbo_ll_per_seq)
    iwae_ll = tf.reduce_sum(iwae_ll_per_seq)
    fivo_ll = tf.reduce_sum(fivo_ll_per_seq)
    batch_size = tf.shape(lengths)[0]
    total_batch_length = tf.reduce_sum(lengths)
    return ((elbo_ll, iwae_ll, fivo_ll), total_batch_length, batch_size,
            global_step)

  def average_bounds_over_dataset(lower_bounds, total_batch_length, batch_size,
                                  sess):
    """Computes the values of the bounds, averaged over config.num_eval data points.

    Args:
      lower_bounds: Tuple of float Tensors containing the values of the bounds
        evaluated on a single batch.
      total_batch_length: Integer Tensor that represents the total number of
        timesteps in the current batch.
      batch_size: Integer Tensor containing the batch size. This can vary if the
        requested batch_size does not evenly divide the size of the dataset.
      sess: A TensorFlow Session object.
    Returns:
      ll_per_t: A length 3 numpy array of floats containing each bound's average
        value, normalized by the total number of timesteps in the datset. Can
        be interpreted as a lower bound on the average log likelihood per
        timestep in the dataset.
      ll_per_seq: A length 3 numpy array of floats containing each bound's
        average value, normalized by the number of sequences in the dataset.
        Can be interpreted as a lower bound on the average log likelihood per
        sequence in the datset.
    """
    total_ll = np.zeros(3, dtype=np.float64)
    total_n_elems = 0.0
    total_length = 0.0
    while total_n_elems < config.num_eval:
      try:
        outs = sess.run([lower_bounds, batch_size, total_batch_length])
      except tf.errors.OutOfRangeError:
        break
      total_ll += outs[0]
      total_n_elems += outs[1]
      total_length += outs[2]
    ll_per_t = total_ll / total_length
    ll_per_seq = total_ll / total_n_elems
    return ll_per_t, ll_per_seq

  def summarize_lls(lls_per_t, lls_per_seq, summary_writer, step):
    """Creates log-likelihood lower bound summaries and writes them to disk.

    Args:
      lls_per_t: An array of 3 python floats, contains the values of the
        evaluated bounds normalized by the number of timesteps.
      lls_per_seq: An array of 3 python floats, contains the values of the
        evaluated bounds normalized by the number of sequences.
      summary_writer: A tf.SummaryWriter.
      step: The current global step.
    """
    def scalar_summary(name, value):
      value = tf.Summary.Value(tag=name, simple_value=value)
      return tf.Summary(value=[value])

    for i, bound in enumerate(["elbo", "iwae", "fivo"]):
      per_t_summary = scalar_summary("%s/%s_ll_per_t" % (config.split, bound),
                                     lls_per_t[i])
      per_seq_summary = scalar_summary("%s/%s_ll_per_seq" %
                                       (config.split, bound),
                                       lls_per_seq[i])
      summary_writer.add_summary(per_t_summary, global_step=step)
      summary_writer.add_summary(per_seq_summary, global_step=step)
    summary_writer.flush()

  with tf.Graph().as_default():
    if config.random_seed: tf.set_random_seed(config.random_seed)
    gpu_config = tf.ConfigProto(); gpu_config.gpu_options.visible_device_list = str(config.gpu) 
    lower_bounds, total_batch_length, batch_size, global_step = create_graph()
    summary_dir = config.logdir + "/" + config.split
    summary_writer = tf.summary.FileWriter(
        summary_dir, flush_secs=15, max_queue=100)
    saver = tf.train.Saver()
    with tf.train.SingularMonitoredSession(config=gpu_config) as sess:
      wait_for_checkpoint(saver, sess, config.logdir)
      step = sess.run(global_step)
      tf.logging.info("Model restored from step %d, evaluating." % step)
      ll_per_t, ll_per_seq = average_bounds_over_dataset(
          lower_bounds, total_batch_length, batch_size, sess)
      summarize_lls(ll_per_t, ll_per_seq, summary_writer, step)
      tf.logging.info("%s elbo ll/t: %.1f, iwae ll/t: %.1f fivo ll/t: %.1f",
                      config.split, ll_per_t[0], ll_per_t[1], ll_per_t[2])
      tf.logging.info("%s elbo ll/seq: %.1f, iwae ll/seq: %.1f fivo ll/seq: %.1f",
                      config.split, ll_per_seq[0], ll_per_seq[1], ll_per_seq[2])
