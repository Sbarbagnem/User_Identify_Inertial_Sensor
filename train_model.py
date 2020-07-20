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

    plot = False  # if true plot distribution of data with a heatmap of activity-subjects train and test
    train = False
    augmented_par = ['random_transformations']
    #augmented_par = []
    plot_augmented = True  # if true plot original and augmented samples
    # if true plot at the end of train % of correct and wrong pred based on act
    plot_pred_base_act = False
    delete_overlap = False  # if true delete overlap sequence between train and test
    augmented = True

    for model_type in ['resnet18_2D']:
        for dataset_name in ['sbhar']:
            for multitask in [False]:
                for overlap in [5.0]:
                    for magnitude in [True]:
                        # for augmented in [False]:
                        if magnitude:
                            #outer_dir = 'OuterPartition_magnitude_prova_balance_'
                            outer_dir = 'OuterPartition_magnitude_'
                            save_dir = 'log_magnitude'
                        else:
                            outer_dir = 'OuterPartition_'
                            save_dir = 'log_no_magnitude'
                        save_dir = 'log_prove_balanced'
                        for fold in [[0]]:
                            print(
                                f"Train on dataset {dataset_name}, with task {'multi_task' if multitask else 'single_task'}, on overlap {overlap}, on fold {fold}")
                            model = Model(dataset_name=dataset_name,
                                          configuration_file=configuration,
                                          multi_task=multitask, lr='dynamic',
                                          model_type=model_type,
                                          fold=fold,
                                          save_dir=save_dir,
                                          outer_dir=outer_dir +
                                          str(overlap)+'/',
                                          overlap=overlap,
                                          magnitude=magnitude,
                                          log=True)
                            model.create_dataset()

                            if augmented:
                                model.load_data(
                                    only_acc=False, normalize=False, delete=delete_overlap)
                            else:
                                model.load_data(
                                    only_acc=False, normalize=True, delete=delete_overlap)

                            if plot:
                                model.plot_distribution_data(
                                    title='no augmented')

                            if augmented:
                                model.augment_data(
                                    augmented_par, plot_augmented)

                            if plot and augmented:
                                model.plot_distribution_data(
                                    title='augmented')

                            if train:
                                model.build_model()
                                model.print_model_summary()
                                model.loss_opt_metric()
                                model.train_model()
                                if plot_pred_base_act:
                                    model.plot_pred_based_act()
