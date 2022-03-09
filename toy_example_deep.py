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

r"""Run experiments with NNGP Kernel.

Usage:

python run_experiments.py \
      --num_train=100 \
      --num_eval=1000 \
      --hparams='nonlinearity=relu,depth=10,weight_var=1.79,bias_var=0.83' \
      --n_gauss=501 --n_var=501 --n_corr=500 --max_gauss=10

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os.path
import time

import numpy as np
import tensorflow as tf

import deep_numerical_gpr
import load_dataset
import time

from sklearn.model_selection import train_test_split

# =============================================================================
# def del_all_flags(FLAGS):
#     flags_dict = FLAGS._flags()    
#     keys_list = [keys for keys in flags_dict]    
#     for keys in keys_list:
#         FLAGS.__delattr__(keys)
# 
# del_all_flags(tf.flags.FLAGS)
# =============================================================================

tf.logging.set_verbosity(tf.logging.INFO)



flags = tf.app.flags
FLAGS = flags.FLAGS



flags.DEFINE_string('hparams', '',
                    'Comma separated list of name=value hyperparameter pairs to'
                    'override the default setting.')
flags.DEFINE_string('experiment_dir', './tmp',
                    'Directory to put the experiment results.')
flags.DEFINE_string('grid_path', './grid_data',
                    'Directory to put or find the training data.')
flags.DEFINE_integer('num_train', 100, 'Number of training data.')
flags.DEFINE_integer('num_eval', 100,
                     'Number of evaluation data. Use 10_000 for full eval')
flags.DEFINE_integer('seed', 1234, 'Random number seed for data shuffling')
flags.DEFINE_boolean('save_kernel', False, 'Save Kernel do disk')
flags.DEFINE_string('dataset', 'fashion-mnist',
                    'Which dataset to use ["fashion-mnist", "cifar-10"]')
flags.DEFINE_boolean('use_fixed_point_norm', False,
                     'Normalize input variance to fixed point variance')

flags.DEFINE_integer('n_gauss', 501,
                     'Number of gaussian integration grid. Choose odd integer.')
flags.DEFINE_integer('n_var', 501,
                     'Number of variance grid points.')
flags.DEFINE_integer('n_corr', 500,
                     'Number of correlation grid points.')
flags.DEFINE_integer('max_var', 100,
                     'Max value for variance grid.')
flags.DEFINE_integer('max_gauss', 10,
                     'Range for gaussian integration.')


def set_default_hparams():
  return tf.contrib.training.HParams(
      nonlinearity='tanh', weight_var=1, bias_var=0, depth = 2, random_seed = 520, repeated_time = 40, use_fixed_point_norm=False)


def do_eval(sess, model, x_data, y_data, save_pred=False):
  """Run evaluation."""

  gp_prediction, variance, stability_eps, K_D_D = model.predict(x_data, sess)
  gp_prediction = gp_prediction/max(gp_prediction)

  
  print(K_D_D)
  print(gp_prediction)
  print(y_data)


  mse = np.mean(np.mean((gp_prediction - y_data)**2, axis=1))
  pred_norm = np.mean(np.linalg.norm(gp_prediction, axis=1))

  tf.logging.info('MSE: %.8f'%mse)

  if save_pred:
    with tf.gfile.Open(
        os.path.join(FLAGS.experiment_dir, 'gp_prediction_stats.npy'),
        'w') as f:
      np.save(f, gp_prediction)

  return gp_prediction, variance, mse, pred_norm, stability_eps


def run_nngp_eval(hparams, run_dir):
  """Runs experiments."""

  tf.gfile.MakeDirs(run_dir)
  # Write hparams to experiment directory.
  with tf.gfile.GFile(run_dir + '/hparams', mode='w') as f:
    f.write(hparams.to_proto().SerializeToString())

  tf.logging.info('Starting job.')
  tf.logging.info('Hyperparameters')
  tf.logging.info('---------------------')
  tf.logging.info(hparams)
  tf.logging.info('---------------------')
  tf.logging.info('Loading data')

  # Get the sets of images and labels for training, validation, and
  # # test on dataset.

  X = np.reshape(np.linspace(0,1,200),(-1,1))*np.pi/2
  Y = np.reshape(np.sin(X), (-1,1))
  
  train_image, train_label, valid_image, valid_label = train_test_split(X, Y, test_size=0.5, random_state=42)


  tf.logging.info('Building Model')

  if hparams.nonlinearity == 'tanh':
    nonlin_fn = tf.tanh
  elif hparams.nonlinearity == 'relu':
    nonlin_fn = tf.nn.relu
  else:
    raise NotImplementedError  
  
  with tf.Session() as sess:
    # Construct NNGP kernel

    # Construct Gaussian Process Regression model
    model = deep_numerical_gpr.DeepGaussianProcessRegression(
        train_image, train_label, nonlin_fn, weight_var=hparams.weight_var,
        bias_var=hparams.bias_var, depth=hparams.depth, random_seed = hparams.random_seed, repeated_time = hparams.repeated_time,
        use_fixed_point_norm=hparams.use_fixed_point_norm)
    
    
    start_time = time.time()
    tf.logging.info('Training')

    # For large number of training points, we do not evaluate on full set to
    # save on training evaluation time.
    start = time.time()
    if FLAGS.num_train <= 5000:
      prediction_train, var_train, mse_train, norm_train, final_eps = do_eval(
          sess, model, train_image[:FLAGS.num_eval],
          train_label[:FLAGS.num_eval])
      tf.logging.info('Evaluation of training set (%d examples) took '
                      '%.3f secs'%(
                          min(FLAGS.num_train, FLAGS.num_eval),
                          time.time() - start_time))


    start_time = time.time()
    tf.logging.info('Validation')
    prediction_valid, var_valid, mse_valid, norm_valid, _ = do_eval(
        sess, model, X,
        Y)
    tf.logging.info('Evaluation of valid set (%d examples) took %.3f secs'%(
        FLAGS.num_eval, time.time() - start_time))

    np.save('./gp_input_deep.npy', X)
    np.save('./gp_label_deep.npy', Y)
    np.save('./gp_prediction_deep.npy', prediction_valid)
    np.save('./gp_variance_deep.npy', var_valid)
    
    
    tf.logging.info('Evaluation of test set (%d examples) took %.3f secs'%(
        FLAGS.num_eval, time.time() - start_time))

  metrics = {
      'train_mse': float(mse_train),
      'train_norm': float(norm_train),
      'valid_mse': float(mse_valid),
      'valid_norm': float(norm_valid),
      'stability_eps': float(final_eps),
  }

  record_results = [
      FLAGS.num_train, hparams.weight_var,
      hparams.bias_var,
      mse_train, mse_valid, final_eps
  ]

  # Store data
  result_file = os.path.join(run_dir, 'results.csv')
  with tf.gfile.Open(result_file, 'a') as f:
    filewriter = csv.writer(f)
    filewriter.writerow(record_results)

  return metrics


def main(argv):
  del argv  # Unused
  hparams = set_default_hparams().parse(FLAGS.hparams)
  run_nngp_eval(hparams, FLAGS.experiment_dir)


if __name__ == '__main__':
  tf.app.run(main)

