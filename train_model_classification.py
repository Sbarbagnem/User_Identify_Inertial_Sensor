from __future__ import absolute_import, division, print_function
import tensorflow as tf
import argparse

import configuration
from model.custom_model import Model

FOLDER_LOG = 'log/'

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="arguments for train classification model")

    parser.add_argument('-p', '--plot', type=int, default=0,
                        help='int to plot or not distribution of train and test')
    parser.add_argument('-t', '--train', type=int,
                        default=1, help='int to train or not model')
    parser.add_argument('-a', '--augmented', type=int, default=0,
                        help='int to apply or not data augmentation on train set')
    parser.add_argument('-ap', '--augmented_par', choices=[
                        'random_warped', 'random_transformations'], nargs='+', help='which data augmentation tecnique apply')
    parser.add_argument('-pa', '--plot_augmented', type=int,
                        default=0, help='int to plot data augmented or not')
    parser.add_argument('-ppba', '--plot_pred_base_act', type=int, default=1,
                        help='plot percentage error of predictions based on activity at the end of train')
    parser.add_argument('-d', '--delete_overlap', type=str, default='delete', choices=[
                        'delete', 'noise', 'no_delete'], help='delete, apply noise or not delete overlap sequence between train and test')
    parser.add_argument('-u', '--unify', type=int, default=0,
                        help='unify some act class, only for sbhar dataset')
    parser.add_argument('-um', '--unify_method', type=str, choices=['sbhar_couple', 'sbhar_all_in_one',
                                                                    'sbhar_complementary', 'sbhar_up_down'], default='sbhar_couple', help='unify method to apply')
    parser.add_argument('-dataset', '--dataset', type=str, choices=['unimib', 'unimib_75w', 'unimib_128w',
                                                                    'sbhar', 'realdisp', 'unimib_sbhar', 'sbhar_six_adl'], help='on which dataset train and test model')

    args = parser.parse_args()

    # GPU settings
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # if true plot distribution of data with a heatmap of activity-subjects train and test
    plot = args.plot

    train = args.train

    augmented = args.augmented
    augmented_par = args.augmented_par
    # if true plot original and augmented samples
    plot_augmented = args.plot_augmented

    # if true plot at the end of train % of correct and wrong user predictions based on activity class
    plot_pred_base_act = args.plot_pred_base_act

    # if 'no_delete' don't delete overlap train sequence between train and test
    # if 'delete' default delete overlapping sequence in train
    # if 'noise' add gaussian noise to overlapping sequence and don't delete it
    delete_overlap = args.delete_overlap

    unify = args.unify
    unify_method = args.unify_method

    for model_type in ['resnet18_2D']:
        for dataset_name in [args.dataset]:
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
                        # fold used as validation during training set
                        fold_val = [0]
                        fold_test = []  # fold used as test set after train, if empty fold_val is used as test and validation
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
                            model.unify_act(
                                model.configuration.sbhar_mapping[unify_method])

                        # plot original distribution data train and test
                        if plot:
                            model.plot_distribution_data(test=True)

                        if augmented:
                            model.augment_data(
                                augmented_par, plot_augmented)
                            model.plot_distribution_data(test=False)

                        model.normalize_data()

                        # tf dataset to iterate over
                        model.tf_dataset(weighted=True)

                        if train:
                            model.build_model()
                            model.print_model_summary()
                            model.loss_opt_metric()
                            model.train_model()
                            if plot_pred_base_act:
                                model.plot_pred_based_act()
