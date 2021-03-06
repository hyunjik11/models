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

"""VRNN classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import sonnet as snt
import tensorflow as tf

import utils


class VRNNCell(snt.AbstractModule):
  """Implementation of a Variational Recurrent Neural Network (VRNN).

  Introduced in "A Recurrent Latent Variable Model for Sequential data"
  by Chung et al. https://arxiv.org/pdf/1506.02216.pdf.

  The VRNN is a sequence model similar to an RNN that uses stochastic latent
  variables to improve its representational power. It can be thought of as a
  sequential analogue to the variational auto-encoder (VAE).

  The VRNN has a deterministic RNN as its backbone, represented by the
  sequence of RNN hidden states h_t. At each timestep, the RNN hidden state h_t
  is conditioned on the previous sequence element, x_{t-1}, as well as the
  latent state from the previous timestep, z_{t-1}.

  In this implementation of the VRNN the latent state z_t is Gaussian. The
  model's prior over z_t is distributed as Normal(mu_t, diag(sigma_t^2)) where
  mu_t and sigma_t are the mean and standard deviation output from a fully
  connected network that accepts the rnn hidden state h_t as input.

  The approximate posterior (also known as q or the encoder in the VAE
  framework) is similar to the prior except that it is conditioned on the
  current target, x_t, as well as h_t via a fully connected network.

  This implementation uses the 'res_q' parameterization of the approximate
  posterior, meaning that instead of directly predicting the mean of z_t, the
  approximate posterior predicts the 'residual' from the prior's mean. This is
  explored more in section 3.3 of https://arxiv.org/pdf/1605.07571.pdf.

  During training, the latent state z_t is sampled from the approximate
  posterior and the reparameterization trick is used to provide low-variance
  gradients.

  The generative distribution p(x_t|z_t, h_t) is conditioned on the latent state
  z_t as well as the current RNN hidden state h_t via a fully connected network.

  To increase the modeling power of the VRNN, two additional networks are
  used to extract features from the data and the latent state. Those networks
  are called data_feat_extractor and latent_feat_extractor respectively.

  There are a few differences between this exposition and the paper.
  First, the indexing scheme for h_t is different than the paper's -- what the
  paper calls h_t we call h_{t+1}. This is the same notation used by Fraccaro
  et al. to describe the VRNN in the paper linked above. Also, the VRNN paper
  uses VAE terminology to refer to the different internal networks, so it
  refers to the approximate posterior as the encoder and the generative
  distribution as the decoder. This implementation also renamed the functions
  phi_x and phi_z in the paper to data_feat_extractor and latent_feat_extractor.
  """

  def __init__(self,
               rnn_cell,
               data_feat_extractor,
               latent_feat_extractor,
               prior,
               approx_posterior,
               generative,
               random_seed=None,
               name="vrnn"):
    """Creates a VRNN cell.

    Args:
      rnn_cell: A subclass of tf.nn.rnn_cell.RNNCell that will form the
        deterministic backbone of the VRNN. The inputs to the RNN will be the
        encoded latent state of the previous timestep with shape
        [batch_size, encoded_latent_size] as well as the encoded input of the
        current timestep, a Tensor of shape [batch_size, encoded_data_size].
      data_feat_extractor: A callable that accepts a batch of data x_t and
        'encodes' it, e.g. runs it through a fully connected network. Must
        accept as argument the inputs x_t, a Tensor of the shape
        [batch_size, data_size] and return a Tensor of shape
        [batch_size, encoded_data_size]. This callable will be called multiple
        times in the VRNN cell so if scoping is not handled correctly then
        multiple copies of the variables in this network could be made. It is
        recommended to use a snt.nets.MLP module, which takes care of this for
        you.
      latent_feat_extractor: A callable that accepts a latent state z_t and
        'encodes' it, e.g. runs it through a fully connected network. Must
        accept as argument a Tensor of shape [batch_size, latent_size] and
        return a Tensor of shape [batch_size, encoded_latent_size].
        This callable must also have the property 'output_size' defined,
        returning encoded_latent_size.
      prior: A callable that implements the prior p(z_t|h_t). Must accept as
        argument the previous RNN hidden state and return a
        tf.contrib.distributions.Normal distribution conditioned on the input.
      approx_posterior: A callable that implements the approximate posterior
        q(z_t|h_t,x_t). Must accept as arguments the encoded target of the
        current timestep and the previous RNN hidden state. Must return
        a tf.contrib.distributions.Normal distribution conditioned on the
        inputs.
      generative: A callable that implements the generative distribution
        p(x_t|z_t, h_t). Must accept as arguments the encoded latent state
        and the RNN hidden state and return a subclass of
        tf.contrib.distributions.Distribution that can be used to evaluate
        the logprob of the targets.
      random_seed: The seed for the random ops. Used mainly for testing.
      name: The name of this VRNN.
    """
    super(VRNNCell, self).__init__(name=name)
    self.rnn_cell = rnn_cell
    self.data_feat_extractor = data_feat_extractor
    self.latent_feat_extractor = latent_feat_extractor
    self.prior = prior
    self.approx_posterior = approx_posterior
    self.generative = generative
    self.random_seed = random_seed
    self.encoded_z_size = latent_feat_extractor.output_size
    self.state_size = (self.rnn_cell.state_size, self.encoded_z_size)

  def zero_state(self, batch_size, dtype):
    """The initial state of the VRNN.

    Contains the initial state of the RNN as well as a vector of zeros
    corresponding to z_0.
    Args:
      batch_size: The batch size.
      dtype: The data type of the VRNN.
    Returns:
      zero_state: A tuple of the initial state of the VRNN and z0
    """
    return (self.rnn_cell.zero_state(batch_size, dtype),
            tf.zeros([batch_size, self.encoded_z_size], dtype=dtype))

  def _build(self, observations, state, mask=None):
    """Computes one timestep of the VRNN.

    Args:
      observations: The observations at the current timestep, a tuple
        containing the model inputs and targets as Tensors of shape
        [num_samples * batch_size, data_size].
      state: Tuple of current state of the VRNN & previous encoded latents
      mask: Tensor of shape [num_samples * batch_size], 1.0 if the current timestep is
        active, 0.0 if it is not active.

    Returns:
      log_q_z: The logprob of the latent state according to the approximate
        posterior. Tensor of shape [num_samples * batch_size].
        So tf.reshape([num_samples, batch_size]) gives correct reshaping. 
      log_p_z: The logprob of the latent state according to the prior. 
        Tensor of shape [num_samples * batch_size]
      log_p_x_given_z: The conditional log-likelihood, i.e. logprob of the
        observation according to the generative distribution.
        Tensor of shape [num_samples * batch_size]
      kl: The analytic kl divergence from q(z) to p(z).
        Tensor of shape [num_samples * batch_size]
      state: Tuple of next state of the VRNN & current encoded latents.
      rec: reconstruction of observation.
        Tensor of shape [num_samples * batch_size, ndims]
      sample: sample from generative distribution.
        Tensor of shape [num_samples * batch_size, ndims]
      z_mean: posterior means.
        Tensor of shape [num_samples * batch_size, latent_size]
    """
    inputs, targets = observations # inputs[t,:]: x_{t-1} (can be mean-centred), targets[t,:]: x_t (raw data)
    rnn_state, prev_latent_encoded = state
    # Encode the data.
    inputs_encoded = self.data_feat_extractor(inputs) 
    targets_encoded = self.data_feat_extractor(targets)
    # Run the RNN cell.
    rnn_inputs = tf.concat([inputs_encoded, prev_latent_encoded], axis=1)
    rnn_out, new_rnn_state = self.rnn_cell(rnn_inputs, rnn_state) # o_t,h_t=f(x_{t-1},z_{t-1},h_{t-1})
    # Create the prior and approximate posterior distributions.
    latent_dist_prior = self.prior(rnn_out) # p(z_t|o_t), a normal distribution.
    latent_dist_q = self.approx_posterior(rnn_out, targets_encoded,
                                          prior_mu=latent_dist_prior.loc) # q(z_t) = D(o_t,x_t), a normal distribution
    # Sample the new latent state z and encode it.
    z_mean = latent_dist_q.mean()
    latent_state = latent_dist_q.sample(seed=self.random_seed)
    latent_encoded = self.latent_feat_extractor(latent_state)
    # Calculate probabilities of the latent state according to the prior p
    # and approximate posterior q.
    log_q_z = tf.reduce_sum(latent_dist_q.log_prob(latent_state), axis=-1)
    log_p_z = tf.reduce_sum(latent_dist_prior.log_prob(latent_state), axis=-1)
    analytic_kl = tf.reduce_sum(
        tf.contrib.distributions.kl_divergence(
            latent_dist_q, latent_dist_prior),
        axis=-1)
    # Create the generative dist. and calculate the logprob of the targets and reconstructions
    generative_dist = self.generative(latent_encoded, rnn_out)
    log_p_x_given_z = tf.reduce_sum(generative_dist.log_prob(targets), axis=-1)
    rec = generative_dist.mean()

    # Compute samples from generative model
    prior_sample = latent_dist_prior.sample(seed=self.random_seed)
    prior_sample_encoded = self.latent_feat_extractor(prior_sample)
    sample_dist = self.generative(prior_sample_encoded, rnn_out)
    sample = sample_dist.mean()
    
    return (log_q_z, log_p_z, log_p_x_given_z, analytic_kl,
            (new_rnn_state, latent_encoded), rec, sample, z_mean)

_DEFAULT_INITIALIZERS = {"w": tf.contrib.layers.xavier_initializer(),
                         "b": tf.zeros_initializer()}


def create_vrnn(
    data_size,
    latent_size,
    generative_class,
    rnn_hidden_size=None,
    fcnet_hidden_sizes=None,
    encoded_data_size=None,
    encoded_latent_size=None,
    conv=False,
    output_channels=[32,32,32,64,64,64,64,64,64],#[32,32,64,64],#[32,32,32,64,64,64],####
    kernel_shapes=[3]*9,#[3]*4,#[3]*6,####
    strides=[1,2,1,2,1,2,1,1,1],#[2,2,2,2],#[1,2,1,2,1,2],####
    sigma_min=0.01,
    raw_sigma_bias=-1.,
    generative_bias_init=0.0,
    lkhd_fixed_sigma=None,
    hidden_activation_fn=tf.nn.relu,
    mean_init=None,
    initializers=None,
    random_seed=None):
  """A factory method for creating VRNN cells.

  Args:
    data_size: The dimension of the vectors that make up the data sequences.
    latent_size: The size of the stochastic latent state of the VRNN.
    generative_class: The class of the generative distribution. Can be either
      ConditionalNormalDistribution or ConditionalBernoulliDistribution.
    rnn_hidden_size: The hidden state dimension of the RNN that forms the
      deterministic part of this VRNN. If None, then it defaults
      to latent_size.
    fcnet_hidden_sizes: A list of python integers, the size of the hidden
      layers of the fully connected networks that parameterize the conditional
      distributions of the VRNN. If None, then it defaults to one hidden
      layer of size latent_size.
    encoded_data_size: The size of the output of the data encoding network. If
      None, defaults to latent_size.
    encoded_latent_size: The size of the output of the latent state encoding
      network. If None, defaults to latent_size.
    conv: Boolean to determine whether to use convolutions for data encoding.
    output_channels: List of number of output channels (feature maps) at each 
      convolutional layer of data feature extractor. Only relevant if conv=True.
    kernel_shapes: List of size of each dim of feature map at each
      convolutional layer of data feature extractor. Only relevant if conv=True.
    strides: List of sizes of strides of data feature extractor.
      Only relevant if conv=True.
    sigma_min: The minimum value that the standard deviation of the
      distribution over the latent state can take.
    raw_sigma_bias: A scalar that is added to the raw standard deviation
      output from the neural networks that parameterize the prior and
      approximate posterior. Useful for preventing standard deviations close
      to zero.
    generative_bias_init: A bias to added to the raw output of the fully
      connected network that parameterizes the generative distribution. Useful
      for initalizing the mean of the distribution to a sensible starting point
      such as the mean of the training data. Only used with Bernoulli generative
      distributions.
    mean_init: tensor of size [data_size] that is added on to output of decoder 
      MLP that corresponds to the mean of the Normal distribution. 0 by default.
    initializers: The variable intitializers to use for the fully connected
      networks and RNN cell. Must be a dictionary mapping the keys 'w' and 'b'
      to the initializers for the weights and biases. Defaults to xavier for
      the weights and zeros for the biases when initializers is None.
    random_seed: A random seed for the VRNN resampling operations.
  Returns:
    model: A VRNNCell object.
  """
  if rnn_hidden_size is None:
    rnn_hidden_size = latent_size
  if fcnet_hidden_sizes is None:
    fcnet_hidden_sizes = [latent_size]
  if encoded_data_size is None:
    encoded_data_size = latent_size
  if encoded_latent_size is None:
    encoded_latent_size = latent_size
  if initializers is None:
    initializers = _DEFAULT_INITIALIZERS
  if conv:
    data_feat_extractor = conv_data_feat_extractor(
      size=encoded_data_size,
      input_size=data_size,
      output_channels=output_channels,
      kernel_shapes=kernel_shapes,
      strides=strides,
      hidden_layer_sizes=fcnet_hidden_sizes,
      hidden_activation_fn=hidden_activation_fn,
      initializers=initializers)
  else:
    data_feat_extractor = snt.nets.MLP(
      output_sizes=fcnet_hidden_sizes + [encoded_data_size],
      initializers=initializers,
      name="data_feat_extractor")
  latent_feat_extractor = snt.nets.MLP(
      output_sizes=fcnet_hidden_sizes + [encoded_latent_size],
      initializers=initializers,
      name="latent_feat_extractor")
  prior = ConditionalNormalDistribution(
      size=latent_size,
      hidden_layer_sizes=fcnet_hidden_sizes,
      hidden_activation_fn=hidden_activation_fn,
      sigma_min=sigma_min,
      raw_sigma_bias=0.25,
      initializers=initializers,
      name="prior")
  approx_posterior = NormalApproximatePosterior(
      size=latent_size,
      hidden_layer_sizes=fcnet_hidden_sizes,
      hidden_activation_fn=hidden_activation_fn,
      sigma_min=sigma_min,
      raw_sigma_bias=raw_sigma_bias,
      initializers=initializers,
      name="approximate_posterior")
  if generative_class == ConditionalBernoulliDistribution:
    generative = ConditionalBernoulliDistribution(
        size=data_size,
        hidden_layer_sizes=fcnet_hidden_sizes,
        hidden_activation_fn=hidden_activation_fn,
        initializers=initializers,
        bias_init=generative_bias_init,
        name="generative")
  else:
    if conv:
      generative = ConditionalNormalDistribution_fixed_var_deconv(
        size=data_size,
        output_channels=output_channels[::-1][:-1] + [1],
        kernel_shapes=kernel_shapes[::-1],
        strides=strides[::-1],
        hidden_layer_sizes=fcnet_hidden_sizes[::-1],
        feature_channels=output_channels[0],
        hidden_activation_fn=hidden_activation_fn,
        initializers=initializers,
        fixed_sigma=lkhd_fixed_sigma,
        mean_init=mean_init,
        name="generative")
    else:
      generative = ConditionalNormalDistribution_fixed_var(
        size=data_size,
        hidden_layer_sizes=fcnet_hidden_sizes,
        hidden_activation_fn=hidden_activation_fn,
        initializers=initializers,
        fixed_sigma=lkhd_fixed_sigma,
        mean_init=mean_init,
        name="generative")
  rnn_cell = tf.nn.rnn_cell.LSTMCell(rnn_hidden_size,
                                     initializer=initializers["w"])
  return VRNNCell(rnn_cell, data_feat_extractor, latent_feat_extractor,
                  prior, approx_posterior, generative, random_seed=random_seed)

def conv_data_feat_extractor(size, input_size, output_channels, kernel_shapes, strides, 
               hidden_layer_sizes, padding='SAME', hidden_activation_fn=tf.nn.relu,
               initializers=None, name="data_feat_extractor"):
  """Convolutional data feature extractor. Applies convnet then fcnet to data.

  Args:
    size: The dimension of the output.
    input_size: The dimensions of the input: H*W. Assumes H=W.
    output_channels: List of number of output channels (feature maps) at each layer.
    kernel_shapes: List of size of each dim of feature map.
    strides: List of sizes of strides
    padding: String indicating type of padding.
    hidden_layer_sizes: List of sizes of all but last hidden layers of the fc
      network applied to the output of conv.
    hidden_activation_fn: The activation function to use on fcnet and convnet.
    initializers: The variable intitializers to use for the fully connected
      network. The network is implemented using snt.nets.MLP so it must
      be a dictionary mapping the keys 'w' and 'b' to the initializers for
      the weights and biases. Defaults to xavier for the weights and zeros
      for the biases when initializers is None.
    name: The name of this distribution, used for sonnet scoping.
  """
  if initializers is None:
    initializers = _DEFAULT_INITIALIZERS
  num_conv_layers = len(output_channels)
  num_channels = 1 # number of channels (1 for greyscale 3 for colour)
  input_shape = [int(np.sqrt(input_size))]*2
  unflatten = snt.BatchReshape(shape=input_shape + [num_channels])

  paddings = [padding]*num_conv_layers
  convnet = snt.nets.ConvNet2D(
      output_channels=output_channels,
      kernel_shapes=kernel_shapes,
      strides=strides,
      paddings=paddings,
      activation=hidden_activation_fn,
      initializers=initializers,
      activate_final=True,
      use_bias=True,
      name=name + "_convnet")
  flatten = snt.BatchFlatten()    
  fcnet = snt.nets.MLP(
      output_sizes=hidden_layer_sizes + [size],
      activation=hidden_activation_fn,
      initializers=initializers,
      activate_final=False,
      use_bias=True,
      name=name + "_fcnet")
  return snt.Sequential([unflatten, convnet, flatten, fcnet])

class ConditionalNormalDistribution_fixed_var(object):
  """A Normal distribution conditioned on Tensor inputs via a fc network.
     Has same variance across dimensions and data points"""

  def __init__(self, size, hidden_layer_sizes, fixed_sigma=None, sigma_min = 0.0, 
               raw_sigma_bias=-1., hidden_activation_fn=tf.nn.relu, mean_init=None,
               initializers=None, name="conditional_normal_distribution_fixed_sigma"):
    """Creates a conditional Normal distribution with fixed variance.

    Args:
      size: The dimension of the random variable.
      hidden_layer_sizes: List of sizes of the hidden layers of the fully connected
        network used to condition the distribution on the inputs.
      fixed_sigma: Value of fixed sigma. Learned by default.
      sigma_min: The minimum standard deviation allowed, a scalar.
      raw_sigma_bias: A scalar that is added to the raw standard deviation
        output from the fully connected network.
      hidden_activation_fn: The activation function to use on the hidden layers
        of the fully connected network.
      mean_init: np.array[size] that forms bias of MLP output 
        that corresponds to the mean of the Normal distribution. 0 by default.
      initializers: The variable intitializers to use for the fully connected
        network up to last hidden layer. The network is implemented 
        using snt.nets.MLP so it must be a dictionary mapping the keys 'w' and 'b' 
        to the initializers for the weights and biases. Defaults to 
        xavier for the weights and zeros for the biases when initializers is None.
      name: The name of this distribution, used for sonnet scoping.
    """
    self.fixed_sigma = fixed_sigma
    self.size = size
    self.sigma_min = sigma_min
    self.raw_sigma_bias = raw_sigma_bias
    self.name = name
    if initializers is None:
      initializers = _DEFAULT_INITIALIZERS
    fcnet_hidden = snt.nets.MLP(
        output_sizes=hidden_layer_sizes,
        activation=hidden_activation_fn,
        initializers=initializers,
        activate_final=True,
        use_bias=True,
        name=name + "_fcnet_hidden")
    if mean_init is not None:
      initializers = {"w": tf.contrib.layers.xavier_initializer(),
                         "b": tf.constant_initializer(mean_init)}
    fcnet_out = snt.nets.MLP(
        output_sizes=[size],
        activation=tf.identity,
        initializers=initializers,
        activate_final=False,
        use_bias=True,
        name=name + "_fcnet_out")
    self.fcnet = snt.Sequential([fcnet_hidden, fcnet_out])

  def condition(self, tensor_list, **unused_kwargs):
    """Computes the parameters of a normal distribution based on the inputs."""
    inputs = tf.concat(tensor_list, axis=1)
    outs = self.fcnet(inputs)
    mu = outs

    if self.fixed_sigma is None:
      with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
        sigma_tensor = tf.get_variable(initializer=0., name="lkhd_preproc_sigma")
        sigma = tf.maximum(tf.nn.softplus(sigma_tensor + self.raw_sigma_bias),
                   self.sigma_min)
    else:
      sigma=self.fixed_sigma
    sigma = sigma + tf.zeros_like(mu)
    return mu, sigma

  def __call__(self, *args, **kwargs):
    """Creates a normal distribution conditioned on the inputs."""
    mu, sigma = self.condition(args, **kwargs)
    return tf.contrib.distributions.Normal(loc=mu, scale=sigma)

class ConditionalNormalDistribution_fixed_var_deconv(object):
  """A Normal distribution conditioned on Tensor inputs via a deconv network.
     Has same variance across dimensions and data points"""

  def __init__(self, size, output_channels, kernel_shapes, strides, hidden_layer_sizes,
               feature_channels, padding='SAME', fixed_sigma=None, sigma_min = 0.0, 
               raw_sigma_bias=-1., hidden_activation_fn=tf.nn.relu, mean_init=None,
               initializers=None, name="conditional_normal_distribution_fixed_sigma"):
    """Creates a conditional Normal distribution with fixed variance.

    Args:
      size: The dimension of the random variable.
      output_channels: List of number of output channels (feature maps) at each layer.
      kernel_shapes: List of size of each dim of feature map.
      strides: List of sizes of strides
      padding: String indicating type of padding.
      hidden_layer_sizes: List of sizes of all but last hidden layers of the fc
        network initially applied to the inputs.
      feature_channels: number of channels of input that will be fed into deconv
      fixed_sigma: Value of fixed sigma. Learned by default.
      sigma_min: The minimum standard deviation allowed, a scalar.
      raw_sigma_bias: A scalar that is added to the raw standard deviation
        output from the fully connected network.
      hidden_activation_fn: The activation function to use on deconvnet and fcnet.
      mean_init: np.array[size] that forms bias of MLP output 
        that corresponds to the mean of the Normal distribution. 0 by default.
      initializers: The variable intitializers to use for the fully connected
        network up to last hidden layer. The network is implemented 
        using snt.nets.MLP so it must be a dictionary mapping the keys 'w' and 'b' 
        to the initializers for the weights and biases. Defaults to 
        xavier for the weights and zeros for the biases when initializers is None.
      name: The name of this distribution, used for sonnet scoping.
    """
    self.fixed_sigma = fixed_sigma
    self.size = size
    self.sigma_min = sigma_min
    self.raw_sigma_bias = raw_sigma_bias
    self.name = name
    target_shape = [int(np.sqrt(size))]*2
    assert np.prod(target_shape) == size

    input_shape, output_shapes = utils.compute_deconv_output_shapes(
                                   target_shape, strides)
    flat_size = np.prod(input_shape)*feature_channels
    hidden_layer_sizes.append(flat_size)
    if initializers is None:
      initializers = _DEFAULT_INITIALIZERS
    fcnet = snt.nets.MLP(
        output_sizes=hidden_layer_sizes,
        activation=hidden_activation_fn,
        initializers=initializers,
        activate_final=True,
        use_bias=True,
        name=name + "_fcnet")

    unflatten = snt.BatchReshape(shape=input_shape + [feature_channels])

    num_deconv_layers = len(output_channels)
    paddings = [padding]*num_deconv_layers
    """
    deconvnet = snt.nets.ConvNet2DTranspose(
        output_channels=output_channels,
        output_shapes=output_shapes,
        kernel_shapes=kernel_shapes,
        strides=strides,
        paddings=paddings,
        activation=hidden_activation_fn,
        initializers=initializers,
        activate_final=False,
        use_bias=True,
        name=name + "_deconvnet")
    flatten = snt.BatchFlatten()
    if mean_init is not None:
      with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
        bias_tensor = tf.get_variable(initializer=mean_init, name="output_bias")
      additive_bias_fn = lambda x: x + bias_tensor
    else:
      additive_bias_fn = lambda x: x
    self.dnn = snt.Sequential([fcnet, 
                               unflatten, 
                               deconvnet, 
                               flatten,
                               additive_bias_fn])
    """
    deconvnet_hidden = snt.nets.ConvNet2DTranspose(
        output_channels=output_channels[:-1],
        output_shapes=output_shapes[:-1],
        kernel_shapes=kernel_shapes[:-1],
        strides=strides[:-1],
        paddings=paddings[:-1],
        activation=hidden_activation_fn,
        initializers=initializers,
        activate_final=True,
        use_bias=True,
        name=name + "_deconvnet_hidden")
    deconvnet_out = snt.nets.ConvNet2DTranspose(
        output_channels=[output_channels[-1]],
        output_shapes=[output_shapes[-1]],
        kernel_shapes=[kernel_shapes[-1]],
        strides=[strides[-1]],
        paddings=[paddings[-1]],
        activation=hidden_activation_fn,
        initializers=initializers,
        activate_final=False,
        use_bias=True,
        name=name + "_deconvnet_out")
    flatten = snt.BatchFlatten()
    if mean_init is not None:
      with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
        bias_tensor = tf.get_variable(initializer=mean_init, name="output_bias")
      additive_bias_fn = lambda x: x + bias_tensor
    else:
      additive_bias_fn = lambda x: x
    self.dnn = snt.Sequential([fcnet, 
                               unflatten, 
                               deconvnet_hidden, 
                               deconvnet_out, 
                               flatten,
                               additive_bias_fn])
  
  def condition(self, tensor_list, **unused_kwargs):
    """Computes the parameters of a normal distribution based on the inputs."""
    inputs = tf.concat(tensor_list, axis=1)
    outs = self.dnn(inputs)
    mu = outs

    if self.fixed_sigma is None:
      with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
        sigma_tensor = tf.get_variable(initializer=0., name="lkhd_preproc_sigma")
        sigma = tf.maximum(tf.nn.softplus(sigma_tensor + self.raw_sigma_bias),
                       self.sigma_min)
    else:
      sigma=self.fixed_sigma
    sigma = sigma + tf.zeros_like(mu)
    return mu, sigma

  def __call__(self, *args, **kwargs):
    """Creates a normal distribution conditioned on the inputs."""
    mu, sigma = self.condition(args, **kwargs)
    return tf.contrib.distributions.Normal(loc=mu, scale=sigma)

class ConditionalNormalDistribution(object):
  """A Normal distribution conditioned on Tensor inputs via a fc network."""

  def __init__(self, size, hidden_layer_sizes, sigma_min=0.0,
               raw_sigma_bias=-1.0, hidden_activation_fn=tf.nn.relu,
               initializers=None, name="conditional_normal_distribution"):
    """Creates a conditional Normal distribution.

    Args:
      size: The dimension of the random variable.
      hidden_layer_sizes: The sizes of the hidden layers of the fully connected
        network used to condition the distribution on the inputs.
      sigma_min: The minimum standard deviation allowed, a scalar.
      raw_sigma_bias: A scalar that is added to the raw standard deviation
        output from the fully connected network. Set to 0.25 by default to
        prevent standard deviations close to 0.
      hidden_activation_fn: The activation function to use on the hidden layers
        of the fully connected network.
      initializers: The variable intitializers to use for the fully connected
        network. The network is implemented using snt.nets.MLP so it must
        be a dictionary mapping the keys 'w' and 'b' to the initializers for
        the weights and biases. Defaults to xavier for the weights and zeros
        for the biases when initializers is None.
      name: The name of this distribution, used for sonnet scoping.
    """
    self.sigma_min = sigma_min
    self.raw_sigma_bias = raw_sigma_bias
    self.name = name
    if initializers is None:
      initializers = _DEFAULT_INITIALIZERS
    self.fcnet = snt.nets.MLP(
        output_sizes=hidden_layer_sizes + [2*size],
        activation=hidden_activation_fn,
        initializers=initializers,
        activate_final=False,
        use_bias=True,
        name=name + "_fcnet")

  def condition(self, tensor_list, **unused_kwargs):
    """Computes the parameters of a normal distribution based on the inputs."""
    inputs = tf.concat(tensor_list, axis=1)
    outs = self.fcnet(inputs)
    mu, sigma = tf.split(outs, 2, axis=1)
    sigma = tf.maximum(tf.nn.softplus(sigma + self.raw_sigma_bias),
                       self.sigma_min)
    return mu, sigma

  def __call__(self, *args, **kwargs):
    """Creates a normal distribution conditioned on the inputs."""
    mu, sigma = self.condition(args, **kwargs)
    return tf.contrib.distributions.Normal(loc=mu, scale=sigma)

class ConditionalBernoulliDistribution(object):
  """A Bernoulli distribution conditioned on Tensor inputs via a fc net."""

  def __init__(self, size, hidden_layer_sizes, hidden_activation_fn=tf.nn.relu,
               initializers=None, bias_init=0.0,
               name="conditional_bernoulli_distribution"):
    """Creates a conditional Bernoulli distribution.

    Args:
      size: The dimension of the random variable.
      hidden_layer_sizes: The sizes of the hidden layers of the fully connected
        network used to condition the distribution on the inputs.
      hidden_activation_fn: The activation function to use on the hidden layers
        of the fully connected network.
      initializers: The variable intiializers to use for the fully connected
        network. The network is implemented using snt.nets.MLP so it must
        be a dictionary mapping the keys 'w' and 'b' to the initializers for
        the weights and biases. Defaults to xavier for the weights and zeros
        for the biases when initializers is None.
      bias_init: A scalar or vector Tensor that is added to the output of the
        fully-connected network that parameterizes the mean of this
        distribution.
      name: The name of this distribution, used for sonnet scoping.
    """
    self.bias_init = bias_init
    if initializers is None:
      initializers = _DEFAULT_INITIALIZERS
    self.fcnet = snt.nets.MLP(
        output_sizes=hidden_layer_sizes + [size],
        activation=hidden_activation_fn,
        initializers=initializers,
        activate_final=False,
        use_bias=True,
        name=name + "_fcnet")

  def condition(self, tensor_list):
    """Computes the p parameter of the Bernoulli distribution."""
    inputs = tf.concat(tensor_list, axis=1)
    return self.fcnet(inputs) + self.bias_init

  def __call__(self, *args):
    p = self.condition(args)
    return tf.contrib.distributions.Bernoulli(logits=p)


class NormalApproximatePosterior(ConditionalNormalDistribution):
  """A Normally-distributed approx. posterior with res_q parameterization,
     that uses MLPs for the construction of q."""

  def condition(self, tensor_list, prior_mu):
    """Generates the mean and variance of the normal distribution.

    Args:
      tensor_list: The list of Tensors to condition on. Will be concatenated and
        fed through a fully connected network.
      prior_mu: The mean of the prior distribution associated with this
        approximate posterior. Will be added to the mean produced by
        this approximate posterior, in res_q fashion.
    Returns:
      mu: The mean of the approximate posterior.
      sigma: The standard deviation of the approximate posterior.
    """
    mu, sigma = super(NormalApproximatePosterior, self).condition(tensor_list)
    return mu + prior_mu, sigma
