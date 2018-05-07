"""Implementation of utility functions such as 
reconstructions + samples from generative model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def reconstruct(cell, inputs, seq_lengths, parallel_iterations=30, swap_memory=True):
  """ Compute mean reconstructions of inputs.
  Args:
    cell: A callable that implements one timestep of the model. See
      models/vrnn.py for an example.
    inputs: The inputs to the model. A potentially nested list or tuple of
      Tensors each of shape [max_seq_len, batch_size, ...]. The Tensors must
      have a rank at least two and have matching shapes in the first two
      dimensions, which represent time and the batch respectively. At each
      timestep 'cell' will be called with a slice of the Tensors in inputs.
    seq_lengths: A [batch_size] Tensor of ints encoding the length of each
      sequence in the batch (sequences can be padded to a common length).
    parallel_iterations: The number of parallel iterations to use for the
      internal while loop.
    swap_memory: Whether GPU-CPU memory swapping should be enabled for the
      internal while loop.
  Returns:
    originals: Original batch of data. [max_seq_len, batch_size, ndims]
    reconstructions: Reconstructions of originals. [max_seq_len, batch_size, ndims]
  """
  inputs, targets = inputs
  max_seq_len, batch_size, ndims = targets.get_shape().as_list()
  t0 = tf.constant(0, tf.int32)
  init_states = cell.zero_state(batch_size, tf.float32)
  ta = tf.TensorArray(tf.float32, max_seq_len, name='rec_ta')
  def while_predicate(t, *unused_args):
    return t < max_seq_len

  def while_step(t, rnn_state, ta):
    log_q_z, log_p_z, log_p_x_given_z, _, new_state, rec, _ = cell((inputs[t,:,:], 
                                                                    targets[t,:,:]), rnn_state)
    # log_q_z, log_p_z, log_p_x_given_z are of size [num_samples * batch_size]
    # rec is of size [num_samples * batch_size, ndims]
    # form weights for the num_samples, and resample according to these weights to get final rec.
    log_weights = log_p_x_given_z + log_p_z - log_q_z
    log_weights = tf.reshape(log_weights, [-1, batch_size]) # [num_samples, batch_size]
    resampling_dist = tf.contrib.distributions.Categorical(
        logits=tf.transpose(log_weights, perm=[1, 0])) 
    # [batch_size] Categorical dist with num_samples categories
    inds = resampling_dist.sample() # [batch_size] of indices in range(num_samples)
    offset = tf.range(batch_size)
    inds = inds * batch_size + offset # convert indices to lie in range(num_samples * batch_size)
    rec = tf.gather(rec, inds) # size [batch_size, ndims]
    
    new_ta = ta.write(t, rec)
    return t + 1, new_state, new_ta

  _, _, ta = tf.while_loop(
      while_predicate,
      while_step,
      loop_vars=(t0, init_states, ta),
      back_prop=False,
      parallel_iterations=parallel_iterations,
      swap_memory=swap_memory)
  reconstructions = ta.stack()
  return targets, reconstructions

def sample(cell, max_seq_len, ndims, num_samples=4, parallel_iterations=30, swap_memory=True):
  """ Obtain samples from generative model.
  Args:
    cell: A callable that implements one timestep of the model. See
      models/vrnn.py for an example.
    max_seq_len: maximum sequence length.
    ndims: The dimension of the vectors that make up the data sequences.
    num_samples: number of samples
    parallel_iterations: The number of parallel iterations to use for the
      internal while loop.
    swap_memory: Whether GPU-CPU memory swapping should be enabled for the
      internal while loop.
  Returns:
    samples: Samples from generative model. [max_seq_len, num_samples, ndims]
  """
  init_inputs = tf.zeros([num_samples, ndims])
  dummy_target = tf.zeros([num_samples, ndims]) # not used
  t0 = tf.constant(0, tf.int32)
  init_states = cell.zero_state(num_samples, tf.float32)
  ta = tf.TensorArray(tf.float32, max_seq_len, name='rec_ta')
  def while_predicate(t, *unused_args):
    return t < max_seq_len

  def while_step(t, rnn_state, inputs, ta):
    _, _, _, _, new_state, _, sample = cell((inputs, dummy_target), rnn_state)
    new_ta = ta.write(t, sample)
    return t + 1, new_state, sample, new_ta

  _, _, _, ta = tf.while_loop(
      while_predicate,
      while_step,
      loop_vars=(t0, init_states, init_inputs, ta),
      back_prop=False,
      parallel_iterations=parallel_iterations,
      swap_memory=swap_memory)
  samples = ta.stack()
  return samples
