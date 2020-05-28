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

    # 10-cross validation
    for model_type in ['resnet18', 'resnet18_multi_branch']:
        for dataset_name in ['unimib', 'sbhar', 'realdisp']:
            for task in ['single_task', 'multi_task']:
                for fold in range(10):
                    model = Model(dataset_name=dataset_name, configuration_file=configuration, multi_task=task, lr='dynamic', model_type='resnet18', fold=fold)
                    model.create_dataset()
                    model.load_data()
                    model.build_model()
                    model.print_model_summary()
                    model.loss_opt_metric()
                    model.train_multi_task()