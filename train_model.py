from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
from sklearn import utils as skutils
import math
import datetime

import configuration
from model.custom_model import Model

if __name__ == '__main__':

    # GPU settings
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    model = Model(dataset_name='sbhar', configuration_file=configuration, multi_task=True, lr='dynamic', model_type='resnet18_multi_branch', fold=0)
    model.create_dataset()
    model.load_data()
    model.build_model()
    model.print_model_summary()
    model.loss_opt_metric()
    model.train_multi_task()