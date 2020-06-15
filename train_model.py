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
    for model_type in ['resnet18']:
        for dataset_name in ['sbhar']:
            for task in [True, False]:
                for overlap in [5.0, 6.0, 7.0, 8.0, 9.0]:
                    for magnitude in [True, False]:
                        if magnitude:
                            outer_dir = 'OuterPartition_magnitude_'
                            save_dir = 'log_magnitude'
                        else:
                            outer_dir = 'OuterPartition_'
                            save_dir = 'log_no_magnitude'
                        for fold in [0]:
                            print(
                                f"Train on dataset {dataset_name}, with task {'multi_task' if task else 'single_task'}, on overlap {overlap}, on fold {fold}")
                            model = Model(dataset_name=dataset_name, configuration_file=configuration, multi_task=task, lr='dynamic',
                                        model_type=model_type, fold=fold, save_dir=save_dir, 
                                        outer_dir=outer_dir+str(overlap)+'/', overlap=overlap, magnitude=magnitude, log=False)
                            model.create_dataset()
                            model.load_data()
                            model.build_model()
                            model.print_model_summary()
                            model.loss_opt_metric()
                            model.train()

