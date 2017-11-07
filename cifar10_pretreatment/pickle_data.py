
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import read
import cnn
import pickle


tf.logging.set_verbosity(tf.logging.INFO)
inputdict = "F:/pre/cifar-10-batches-py/"


def main(unused_argv):
  # Load training and eval data
  train_data = np.asarray(read.dataset(inputdict), dtype=np.float32)
  train_labels = np.asarray(read.labels(inputdict), dtype=np.int32)
  
  eval_data = np.asarray(read.dataset_test(inputdict), dtype=np.float32)
  eval_labels = np.asarray(read.labels_test(inputdict), dtype=np.int32)
  
  pickle.dump(train_data, open("train_data", "wb" ))
  pickle.dump(train_labels, open("train_labels", "wb" ))
  pickle.dump(eval_data, open("eval_data", "wb" ))
  pickle.dump(eval_labels, open("eval_labels", "wb" ))

if __name__ == "__main__":
  tf.app.run()

