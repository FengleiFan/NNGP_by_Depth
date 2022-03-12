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

"""Data loader for NNGP experiments.

Loading MNIST dataset with train/valid/test split as numpy array.

Usage:
mnist_data = load_dataset.load_mnist(num_train=50000, use_float64=True,
                                     mean_subtraction=True)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', './tmp/nngp/data/',
                    'Directory for data.')

def load_mnist(num_train=50000,
               use_float64=False,
               mean_subtraction=False,
               random_roated_labels=False):
  """Loads MNIST as numpy array."""

  data_dir = FLAGS.data_dir
  datasets = input_data.read_data_sets(
      data_dir, False, validation_size=10000, one_hot=True)
  mnist_data = _select_mnist_subset(
      datasets,
      num_train,
      use_float64=use_float64,
      mean_subtraction=mean_subtraction,
      random_roated_labels=random_roated_labels)

  return mnist_data




def _select_mnist_subset(datasets,
                         num_train=100,
                         digits=list(range(10)),
                         seed=9999,
                         sort_by_class=False,
                         use_float64=False,
                         mean_subtraction=False,
                         random_roated_labels=False):
  """Select subset of MNIST and apply preprocessing."""
  np.random.seed(seed)
  digits.sort()
  subset = copy.deepcopy(datasets)

  num_class = len(digits)
  num_per_class = num_train // num_class

  idx_list = np.array([], dtype='uint8')

  ys = np.argmax(subset.train.labels, axis=1)  # undo one-hot

  for digit in digits:
    if datasets.train.num_examples == num_train:
      idx_list = np.concatenate((idx_list, np.where(ys == digit)[0]))
    else:
      idx_list = np.concatenate((idx_list,
                                 np.where(ys == digit)[0][:num_per_class]))
  if not sort_by_class:
    np.random.shuffle(idx_list)

  data_precision = np.float64 if use_float64 else np.float32

  train_image = subset.train.images[idx_list][:num_train].astype(data_precision)
  train_label = subset.train.labels[idx_list][:num_train].astype(data_precision)
  valid_image = subset.validation.images.astype(data_precision)
  valid_label = subset.validation.labels.astype(data_precision)
  test_image = subset.test.images.astype(data_precision)
  test_label = subset.test.labels.astype(data_precision)

  if sort_by_class:
    train_idx = np.argsort(np.argmax(train_label, axis=1))
    train_image = train_image[train_idx]
    train_label = train_label[train_idx]

  if mean_subtraction:
    train_image_mean = np.mean(train_image)
    train_label_mean = np.mean(train_label)
    train_image -= train_image_mean
    train_label -= train_label_mean
    valid_image -= train_image_mean
    valid_label -= train_label_mean
    test_image -= train_image_mean
    test_label -= train_label_mean

  if random_roated_labels:
    r, _ = np.linalg.qr(np.random.rand(10, 10))
    train_label = np.dot(train_label, r)
    valid_label = np.dot(valid_label, r)
    test_label = np.dot(test_label, r)

  return (train_image, train_label,
          valid_image, valid_label,
          test_image, test_label)



def load_fashion_mnist(num_train=50000,
               use_float64=False,
               mean_subtraction=False,
               random_roated_labels=False):
  """Loads MNIST as numpy array."""

  data_dir = FLAGS.data_dir
  
  fashion_mnist = tf.keras.datasets.fashion_mnist

  (train_images, tr_labels), (test_images, te_labels) = fashion_mnist.load_data()
  
  train_images = np.reshape(train_images, (60000, -1))
  test_images = np.reshape(test_images, (10000, -1))
  index_tr = np.arange(60000)
  train_labels = np.zeros((60000,10))
  train_labels[index_tr, tr_labels] = 1
  index_te = np.arange(10000)
  test_labels = np.zeros((10000,10))
  test_labels[index_te, te_labels] = 1
    
  
  fashion_mnist_data = _select_fashion_mnist_subset(
      train_images, train_labels, test_images, test_labels,
      num_train,
      use_float64=use_float64,
      mean_subtraction=mean_subtraction,
      random_roated_labels=random_roated_labels)

  return fashion_mnist_data




def _select_fashion_mnist_subset(train_images, train_labels, test_images, test_labels,
                         num_train=100,
                         digits=list(range(10)),
                         seed=9999,
                         sort_by_class=False,
                         use_float64=False,
                         mean_subtraction=False,
                         random_roated_labels=False):
  """Select subset of MNIST and apply preprocessing."""
  np.random.seed(seed)
  digits.sort()
  

  num_class = len(digits)
  num_per_class = num_train // num_class

  idx_list = np.array([], dtype='uint8')

  ys = np.argmax(train_labels, axis=1)  # undo one-hot

  for digit in digits:
    if 50000 == num_train:
      idx_list = np.concatenate((idx_list, np.where(ys == digit)[0]))
    else:
      idx_list = np.concatenate((idx_list,
                                 np.where(ys == digit)[0][:num_per_class]))
  if not sort_by_class:
    np.random.shuffle(idx_list)

  data_precision = np.float64 if use_float64 else np.float32

  train_image = train_images[idx_list][:num_train].astype(data_precision)
  train_label = train_labels[idx_list][:num_train].astype(data_precision)
  valid_image = train_images[idx_list][num_train:num_train+1000].astype(data_precision)
  valid_label = train_labels[idx_list][num_train:num_train+1000].astype(data_precision)
  test_image = test_images.astype(data_precision)
  test_label = test_labels.astype(data_precision)

  if sort_by_class:
    train_idx = np.argsort(np.argmax(train_label, axis=1))
    train_image = train_image[train_idx]
    train_label = train_label[train_idx]

  if mean_subtraction:
    train_image_mean = np.mean(train_image)
    train_label_mean = np.mean(train_label)
    train_image -= train_image_mean
    train_label -= train_label_mean
    valid_image -= train_image_mean
    valid_label -= train_label_mean
    test_image -= train_image_mean
    test_label -= train_label_mean

  if random_roated_labels:
    r, _ = np.linalg.qr(np.random.rand(10, 10))
    train_label = np.dot(train_label, r)
    valid_label = np.dot(valid_label, r)
    test_label = np.dot(test_label, r)

  return (train_image, train_label,
          valid_image, valid_label,
          test_image, test_label)


def load_cifar10(num_train=40000,
               use_float64=False,
               mean_subtraction=False,
               random_roated_labels=False):
  """Loads MNIST as numpy array."""

  data_dir = FLAGS.data_dir
  
  cifar10 = tf.keras.datasets.cifar10

  (train_images, tr_labels), (test_images, te_labels) = cifar10.load_data()
  
  train_images = np.reshape(train_images, (50000, -1))
  test_images = np.reshape(test_images, (10000, -1))
  index_tr = np.arange(50000)
  train_labels = np.zeros((50000,10))
  train_labels[index_tr, tr_labels[:,0]] = 1
  index_te = np.arange(10000)
  test_labels = np.zeros((10000,10))
  test_labels[index_te, te_labels[:,0]] = 1
  
  
  cifar10_data = _select_cifar10_subset(
      train_images, train_labels, test_images, test_labels,
      num_train,
      use_float64=use_float64,
      mean_subtraction=mean_subtraction,
      random_roated_labels=random_roated_labels)

  return cifar10_data




def _select_cifar10_subset(train_images, train_labels, test_images, test_labels,
                         num_train=100,
                         digits=list(range(10)),
                         seed=9999,
                         sort_by_class=False,
                         use_float64=False,
                         mean_subtraction=False,
                         random_roated_labels=False):
  """Select subset of MNIST and apply preprocessing."""
  np.random.seed(seed)
  digits.sort()
  

  num_class = len(digits)
  num_per_class = num_train // num_class

  idx_list = np.array([], dtype='uint8')

  ys = np.argmax(train_labels, axis=1)  # undo one-hot

  for digit in digits:
    if 40000 == num_train:
      idx_list = np.concatenate((idx_list, np.where(ys == digit)[0]))
    else:
      idx_list = np.concatenate((idx_list,
                                 np.where(ys == digit)[0][:num_per_class]))
  if not sort_by_class:
    np.random.shuffle(idx_list)

  data_precision = np.float64 if use_float64 else np.float32

  train_image = train_images[idx_list][:num_train].astype(data_precision)
  train_label = train_labels[idx_list][:num_train].astype(data_precision)
  valid_image = train_images[idx_list][num_train:num_train+1000].astype(data_precision)
  valid_label = train_labels[idx_list][num_train:num_train+1000].astype(data_precision)
  test_image = test_images.astype(data_precision)
  test_label = test_labels.astype(data_precision)

  if sort_by_class:
    train_idx = np.argsort(np.argmax(train_label, axis=1))
    train_image = train_image[train_idx]
    train_label = train_label[train_idx]

  if mean_subtraction:
    train_image_mean = np.mean(train_image)
    train_label_mean = np.mean(train_label)
    train_image -= train_image_mean
    train_label -= train_label_mean
    valid_image -= train_image_mean
    valid_label -= train_label_mean
    test_image -= train_image_mean
    test_label -= train_label_mean

  if random_roated_labels:
    r, _ = np.linalg.qr(np.random.rand(10, 10))
    train_label = np.dot(train_label, r)
    valid_label = np.dot(valid_label, r)
    test_label = np.dot(test_label, r)

  return (train_image, train_label,
          valid_image, valid_label,
          test_image, test_label)