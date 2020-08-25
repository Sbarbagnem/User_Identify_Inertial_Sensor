from __future__ import absolute_import, division, print_function
import tensorflow as tf

import configuration
from model.custom_model import Model

FOLDER_LOG = 'log/'

if __name__ == '__main__':

    # GPU settings
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    plot = True  # if true plot distribution of data with a heatmap of activity-subjects train and test

    train = True

    augmented_par = ['random_warped', 'random_transformations']
    plot_augmented = False  # if true plot original and augmented samples

    # if true plot at the end of train % of correct and wrong user predictions based on activity class
    plot_pred_base_act = True
    # if true plot at the end of train % of correct and wrong user predictions based on user class
    plot_pred_base_user = False

    # if 'no_delete' don't delete overlap train sequence between train and test
    # if 'delete' default delete overlapping sequence in train
    # if 'noise' add gaussian noise to overlapping sequence and don't delete it
    delete_overlap = 'delete'

    augmented = True

    unify = False
    unify_method = 'sbhar_couple'

    for model_type in ['resnet18_2D']:
        for dataset_name in ['unimib']:
            for multitask in [False]:
                for overlap in [5.0]:
                    for magnitude in [True]:
                        if magnitude:
                            if dataset_name == 'sbhar_six_adl':
                                outer_dir = 'OuterPartition_magnitude_sbhar_six_adl_'
                            else:
                                outer_dir = 'OuterPartition_magnitude_'
                            save_dir = FOLDER_LOG + 'log_magnitude'
                        else:
                            outer_dir = 'OuterPartition_' 
                            save_dir = FOLDER_LOG + 'log_no_magnitude'
                        save_dir = 'log_scazzo'
                        fold_val = [0] # fold used as validation during training set
                        fold_test = [] # fold used as test set after train, if empty fold_val is used as test and validation
                        model = Model(dataset_name=dataset_name,
                                        configuration_file=configuration,
                                        multi_task=multitask, lr='dynamic',
                                        model_type=model_type,
                                        fold_test=fold_val,
                                        fold_val=fold_test,
                                        save_dir=save_dir,
                                        outer_dir=outer_dir +
                                        str(overlap)+'/',
                                        overlap=overlap,
                                        magnitude=magnitude,
                                        log=True)
                        model.create_dataset()
                        model.load_data(only_acc=False, delete=delete_overlap)

                        if unify:
                            model.unify_act(model.configuration.sbhar_mapping[unify_method])

                        # plot original distribution data train and test
                        if plot:
                            model.plot_distribution_data(test=True)

                        if augmented:
                            model.augment_data(
                                augmented_par, plot_augmented)
                            model.plot_distribution_data(test=False)

                        model.normalize_data()

                        # tf dataset to iterate over
                        model.tf_dataset(weighted=False)

                        if train:
                            model.build_model()
                            model.print_model_summary()
                            model.loss_opt_metric()
                            model.train_model()
                            if plot_pred_base_act:
                                model.plot_pred_based_act()
                            if plot_pred_base_user:
                                model.plot_pred_based_user()
