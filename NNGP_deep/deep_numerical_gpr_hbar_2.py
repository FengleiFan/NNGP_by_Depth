# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gaussian process regression model based on GPflow.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean("print_kernel", False, "Option to print out kernel")


class DeepGaussianProcessRegression(object):
  """Gaussian process regression model based on GPflow.

  Args:
    input_x: numpy array, [data_size, input_dim]
    output_x: numpy array, [data_size, output_dim]
    kern: NNGPKernel class
  """

  def __init__(self, input_x, output_y, nonlin_fn, 
               weight_var=1., bias_var=1., depth=500, 
               random_seed = 521, repeated_time = 100, use_fixed_point_norm=False):
      
    with tf.name_scope("init"):
      self.input_x = input_x
      self.weight_var = weight_var
      self.bias_var = bias_var      
      self.output_y = output_y
      self.nonlin_fn = nonlin_fn
      self.depth = depth
      self.repeated_time = repeated_time
      self.num_train, self.input_dim = input_x.shape
      _, self.output_dim = output_y.shape
      self.use_fixed_point_norm = use_fixed_point_norm
      self.stability_eps = tf.identity(tf.placeholder(tf.float64))
      self.current_stability_eps = 1e-10
      self.middle_dim = 30
      self.y_pl = tf.placeholder(
          tf.float64, [self.num_train, self.output_dim], name="y_train")
      self.x_pl = tf.identity(
          tf.placeholder(tf.float64, [self.num_train, self.input_dim],
                         name="x_train"))

      self.l_np = None
      self.v_np = None
      self.k_np = None

    self.k_data_data = tf.identity(self._generate_k_ful(self.x_pl))

  def _build_predict(self, n_test, full_cov=True):
    with tf.name_scope("build_predict"):
      self.x_test_pl = tf.identity(
          tf.placeholder(tf.float64, [n_test, self.input_dim], name="x_test_pl")
      )

    tf.logging.info("Using pre-computed Kernel")
    self.k_data_test = self._generate_k_ful(self.x_pl, self.x_test_pl)
    
    with tf.name_scope("build_predict"):
      a = tf.matrix_triangular_solve(self.l, self.k_data_test)
      fmean = tf.matmul(a, self.v, transpose_a=True)

      if full_cov:
        fvar = self._generate_k_ful(self.x_test_pl) - tf.matmul(
            a, a, transpose_a=True)
        shape = [1, 1, self.y_pl.shape[1]]
        fvar = tf.tile(tf.expand_dims(fvar, 2), shape)


      self.fmean = fmean
      self.fvar = fvar
         

  def _build_cholesky(self):
    tf.logging.info("Computing Kernel")
    self.k_data_data_reg = self.k_data_data + tf.eye(
        self.input_x.shape[0], dtype=tf.float64) * self.stability_eps
    if FLAGS.print_kernel:
      self.k_data_data_reg = tf.Print(
          self.k_data_data_reg, [self.k_data_data_reg],
          message="K_DD = ", summarize=100)
    self.l = tf.cholesky(self.k_data_data_reg)
    self.v = tf.matrix_triangular_solve(self.l, self.y_pl)
    


  def _generate_k_ful(self, input1, input2=None):
    """Iteratively building the diagonal part (variance) of the NNGP kernel.

    Args:
      input_x: tensor of input of size [num_data, input_dim].
      return_full: boolean for output to be [num_data] sized or a scalar value
        for normalized inputs

    Sets self.layer_qaa_dict of {layer #: qaa at the layer}

    Returns:
      qaa: variance at the output.
    """

    if input2 is None:
      input2 = input1

   
    kernel = []
    
    with tf.name_scope("k_full"):
      
       
      for  r in np.arange(self.repeated_time):     
        weight_matrix_1 = tf.random.normal([self.input_dim, self.middle_dim], 
                                         mean=0.0, stddev=np.sqrt(self.weight_var/self.input_dim), dtype=tf.dtypes.float64) 
        
        bias_matrix_1 = tf.random.normal(
        [self.middle_dim], mean=0.0, stddev=np.sqrt(self.bias_var), dtype=tf.dtypes.float64, seed=None, name=None) 
        
                
        layer11 = self.nonlin_fn(tf.matmul(input1, weight_matrix_1) + bias_matrix_1)
      
        layer21 = self.nonlin_fn(tf.matmul(input2, weight_matrix_1) + bias_matrix_1)
        
        output1 = layer11
        output2 = layer21
        
        
        output1_array = []
        output2_array = []
        
        for i in range(self.depth): 
            
            weight_matrix = tf.random.normal([self.middle_dim, self.middle_dim], 
                                         mean=0.0, stddev=np.sqrt(self.weight_var/self.output_dim), dtype=tf.dtypes.float64) 
        
            bias_matrix = tf.random.normal([self.middle_dim], 
                                             mean=0.0, stddev=np.sqrt(self.bias_var), dtype=tf.dtypes.float64, seed=None, name=None) 
            output1 = self.nonlin_fn(tf.matmul(output1, weight_matrix) + bias_matrix)
            output2 = self.nonlin_fn(tf.matmul(output2, weight_matrix) + bias_matrix)
            
            if  i%2 == 0:
                output1_array.append(output1)
                output2_array.append(output2)
        
        output1_ = self.nonlin_fn(sum(output1_array))
        output2_ = self.nonlin_fn(sum(output2_array))
        
        weight_matrix_final = tf.random.normal([self.middle_dim, self.output_dim], 
                                         mean=0.0, stddev=np.sqrt(self.weight_var/self.input_dim), dtype=tf.dtypes.float64) 
        
        bias_matrix_final = tf.random.normal(
        [self.output_dim], mean=0.0, stddev=np.sqrt(self.bias_var), dtype=tf.dtypes.float64, seed=None, name=None) 
        
        output1 = self.nonlin_fn(tf.matmul(output1_, weight_matrix_final) + bias_matrix_final)
      
        output2 = self.nonlin_fn(tf.matmul(output2_, weight_matrix_final) + bias_matrix_final)
        
        kernel = tf.matmul(
        output1, output2, transpose_b=True)/self.middle_dim  
        
        kernel += kernel

    return kernel/self.repeated_time   
    

       
    
  def _input_layer_normalization(self, x):
    """Input normalization to unit variance or fixed point variance.
    """
    with tf.name_scope("input_layer_normalization"):
      # Layer norm, fix to unit variance
      eps = 1e-15
      mean, var = tf.nn.moments(x, axes=[1], keep_dims=True)
      x_normalized = (x - mean) / tf.sqrt(var + eps)
      if self.use_fixed_point_norm:
        x_normalized *= tf.sqrt(
            (self.var_fixed_point[0] - self.bias_var) / self.weight_var)
      return x_normalized      

  def predict(self, test_x, sess, get_var=True):
    """Compute mean and varaince prediction for test inputs.

    Raises:
      ArithmeticError: Cholesky fails even after increasing to large values of
        stability epsilon.
    """
    if self.l_np is None:
      
      KKDD = sess.run(self.k_data_data,
                           feed_dict={self.x_pl: self.input_x})
      print(KKDD)
      np.save('deep_kernel_400.npy', KKDD)
      
      self._build_cholesky()
      
      start_time = time.time()
      self.k_np = sess.run(self.k_data_data,
                           feed_dict={self.x_pl: self.input_x})
      tf.logging.info("Computed K_DD in %.3f secs" % (time.time() - start_time))

      while self.current_stability_eps < 1:
        try:
          start_time = time.time()
          self.l_np, self.v_np = sess.run(
              [self.l, self.v],
              feed_dict={self.y_pl: self.output_y,
                         self.k_data_data: self.k_np,
                         self.stability_eps: self.current_stability_eps})
          tf.logging.info(
              "Computed L_DD in %.3f secs"% (time.time() - start_time))
          break

        except tf.errors.InvalidArgumentError:
          self.current_stability_eps *= 10
          tf.logging.info("Cholesky decomposition failed, trying larger epsilon"
                          ": {}".format(self.current_stability_eps))

    if self.current_stability_eps > 0.2:
      raise ArithmeticError("Could not compute Cholesky decomposition.")

    n_test = test_x.shape[0]
    self._build_predict(n_test)
    feed_dict = {
        self.x_pl: self.input_x,
        self.x_test_pl: test_x,
        self.l: self.l_np,
        self.v: self.v_np
    }

    start_time = time.time()
    if get_var:
      mean_pred, var_pred = sess.run(
          [self.fmean, self.fvar], feed_dict=feed_dict)
      k_d_d = sess.run(
          [self.k_data_data], feed_dict=feed_dict)
      tf.logging.info("Did regression in %.3f secs"% (time.time() - start_time))
      return mean_pred, var_pred, self.current_stability_eps, k_d_d

    else:
      mean_pred = sess.run(self.fmean, feed_dict=feed_dict)
      tf.logging.info("Did regression in %.3f secs"% (time.time() - start_time))
      return mean_pred, self.current_stability_eps

