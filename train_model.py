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
        for dataset_name in ['unimib_sbhar']:
            for multitask in [False]:
                for overlap in [5.0]:
                    for magnitude in [True]:
                        if magnitude:
                            outer_dir = 'OuterPartition_magnitude_'
                            save_dir = 'log_magnitude'
                        else:
                            outer_dir = 'OuterPartition_'
                            save_dir = 'log_no_magnitude'
                        save_dir = 'log_merged_dataset'
                        for fold in [0]:
                            print(
                                f"Train on dataset {dataset_name}, with task {'multi_task' if multitask else 'single_task'}, on overlap {overlap}, on fold {fold}")
                            model = Model(dataset_name=dataset_name, configuration_file=configuration, multi_task=multitask, lr='dynamic',
                                        model_type=model_type, fold=fold, save_dir=save_dir, 
                                        outer_dir=outer_dir+str(overlap)+'/', overlap=overlap, magnitude=magnitude, log=True)
                            model.create_dataset()
                            if dataset_name == 'unimib_sbhar':
                                model.load_data_merged(augmented=False)
                            else:
                                model.load_data(augmented=False)
                            model.build_model()
                            model.print_model_summary()
                            model.loss_opt_metric()
                            model.train()

