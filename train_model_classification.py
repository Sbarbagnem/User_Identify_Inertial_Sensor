from __future__ import absolute_import, division, print_function
import tensorflow as tf
import argparse
import sys
from pprint import pprint

import configuration
from model.custom_model import Model
from util.utils import str2bool
import matplotlib.pyplot as plt
import numpy as np

from util.utils import plot_pred_based_act, save_mean_performance_txt

FOLDER_LOG = 'log/'

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Arguments for train classification model")

    parser.add_argument(
        '-plot_distribution',
        '--plot_distribution',
        type=str2bool,
        default=False,
        help='bool to plot or not distribution of train, val and test')
    parser.add_argument('-t', '--train', type=str2bool,
                        default=True, help='bool to train or not model')
    parser.add_argument(
        '-plot_pred_based_act_test',
        '--plot_pred_based_act_test',
        type=str2bool,
        default=False,
        help='bool to plot or not accuracy based on activity on test fold')
    parser.add_argument(
        '-a',
        '--augmented',
        type=str2bool,
        default=False,
        help='bool to apply or not data augmentation on train set')
    parser.add_argument(
        '-ap',
        '--augmented_par',
        choices=[
            'random_warped',
            'random_transformations'],
        nargs='+',
        help='which data augmentation tecnique apply in  sequence')
    parser.add_argument(
        '-pa',
        '--plot_augmented',
        type=str2bool,
        default=False,
        help='bool to plot data augmented or not')
    parser.add_argument(
        '-plot_pred_base_act_val',
        '--plot_pred_base_act_val',
        type=str2bool,
        default=False,
        help='bool, if true plot percentage error of predictions based on activity at the end of train')
    parser.add_argument(
        '-u',
        '--unify',
        type=str2bool,
        default=False,
        help='bool to unify some act class, only for sbhar dataset')
    parser.add_argument(
        '-um',
        '--unify_method',
        type=str,
        choices=[
            'sbhar_couple',
            'sbhar_all_in_one',
            'sbhar_complementary',
            'sbhar_up_down'],
        default='sbhar_couple',
        help='unify method to apply')
    parser.add_argument(
        '-dataset',
        '--dataset',
        type=str,
        choices=[
            'unimib',
            'unimib_75w',
            'unimib_128w',
            'sbhar',
            'sbhar_128w',
            'unimib_sbhar',
            'sbhar_six_adl',
            'realdisp_RT',
            'realdisp_LT',
            'realdisp_RT_LT',
            'realdisp_RLA',
            'realdisp_LLA',
            'realdisp_RLA_LLA',
            'realdisp_RUA',
            'realdisp_LUA',
            'reladisp_RUA_LUA',
            'realdisp_RC',
            'realdisp_LC',
            'realdisp_RC_LC',
            'realdisp_BACK'],
        help='on which dataset train and test model',
        required=True)
    parser.add_argument(
        '-w',
        '--weighted',
        type=str,
        choices=[
            'no',
            'balance',
            'train_set'],
        default='no',
        help='chose a batch balance on act, same distribution of train set or random')
    parser.add_argument(
        '-compose',
        '--compose',
        type=str2bool,
        default=False,
        help='bool, if true apply also compose transformations')
    parser.add_argument(
        '-only_compose',
        '--only_compose',
        type=str2bool,
        default=False,
        help='bool, if true data augmentation method return only compose sequences')
    parser.add_argument(
        '-fold_test',
        '--fold_test',
        type=int,
        nargs='+',
        default=[0],
        help='list of int represent folds on wich testing model',
        required=True)
    parser.add_argument(
        '-wbo',
        '--weighted_based_on',
        type=str,
        choices=[
            'subject',
            'act',
            'act_subject'],
        default='',
        help='weighted samples in dataset based on activity or subject frequency')
    parser.add_argument(
        '-model',
        '--model',
        type=str,
        choices=[
            'resnet18_1D',
            'resnet18_2D',
            'resnet18_lstm_parallel',
            'resnet18_lstm_consecutive',
            'resnet18_multi_branch'],
        default='resnet18_2D',
        help='define model to train')
    parser.add_argument('-init_lr', '--init_lr', type=float,
                        default=0.001, help='init learning rate')
    parser.add_argument('-drop_factor', '--drop_factor', type=float,
                        default=0.25, help='drop factor for learning rate')
    parser.add_argument('-drop_epoch', '--drop_epoch', type=int,
                        default=20, help='drop learning rate every epoch')
    parser.add_argument(
        '-magnitude',
        '--magnitude',
        type=str2bool,
        default=True,
        help='bool use or not magnitude')
    parser.add_argument(
        '-aug_function',
        '--aug_function',
        choices=[
            'jitter',
            'scaling',
            'permutation',
            'rotation',
            'magnitude_warp',
            'time_warp',
            'random_sampling'],
        nargs='+',
        help='list of function applied to sequences to augment train set',
        default=[
            'jitter',
            'scaling',
            'permutation',
            'rotation',
            'magnitude_warp',
            'time_warp',
            'random_sampling'])
    parser.add_argument(
        '-ratio',
        '--ratio',
        type=float,
        nargs='+',
        default=[0],
        help='ratio of augmented data in random transformations after had equal number of samples act/sub, for every data augmentation technique choosen')
    parser.add_argument(
        '-n_func_to_apply',
        '--n_func_to_apply',
        type=int,
        default=1,
        help='how functions from choosen set apply on sequences'
    )
    parser.add_argument(
        '-epochs',
        '--epochs',
        type=int,
        default=100,
        help='number of epochs of training'
    )
    parser.add_argument(
        '-only_acc',
        '--only_acc',
        type=str2bool,
        default=False,
        help='bool to indicate to use only accelerometer data or all data'
    )
    parser.add_argument(
        '-only_acc_gyro',
        '--only_acc_gyro',
        type=str2bool,
        default=False,
        help='bool to indicate to use only accelerometer and gyroscope data or all data'
    )
    parser.add_argument(
        '-run_colab',
        '--run_colab',
        type=str2bool,
        default=False,
        help='bool to indicate if the code wiil run or not in colab, to change directory to data'
    )
    parser.add_argument(
        '-colab_path',
        '--colab_path',
        type=str,
        help='colab path to data'
    )
    parser.add_argument(
        '-overlap',
        '--overlap',
        type=float,
        nargs='+',
        default=[5.0],
        help='ratio of augmented data in random transformations after had equal number of samples act/sub, for every data augmentation technique choosen')
    parser.add_argument(
        '-path_best_model',
        '--path_best_model',
        type=str,
        default='best_seen.h5',
        help='path and name to save best model seen in training phase'
    )
    parser.add_argument(
        '-save_best_model',
        '--save_best_model',
        type=str2bool,
        default=False,
        help='bool, if true save the best seen model in path_best_seen '
    )
    parser.add_argument(
        '-log',
        '--log',
        type=str2bool,
        help='bool, if true log all train history',
        default=True
    )
    parser.add_argument(
        '-print_model_summary',
        '--print_model_summary',
        type=str2bool,
        help='bool, if true print summary of model choose',
        default=False
    )
    parser.add_argument(
        '-confusion_matrix',
        '--confusion_matrix',
        type=str2bool,
        help='bool, if true print confusion matrix on test set',
        default=False
    )
    ### FOR REALDISP ###
    parser.add_argument(
        '-sensor_displace',
        '--sensor_displace',
        type=str,
        nargs='+',
        default=['ideal', 'self', 'mutual'],
        choices=['ideal', 'self', 'mutual'],
        help='for realdisp dataset, chose type of sensor displacement one or more'
    )
    parser.add_argument(
        '-mean_perfomance_cross_validation',
        '--mean_perfomance_cross_validation',
        type=str2bool,
        default=False,
        help='mean, on cross validation, performance on user classification based on activity'
    )
    parser.add_argument(
        '-save_mean_perfomance_cross_validation',
        '--save_mean_perfomance_cross_validation',
        type=str2bool,
        default=False,
        help='save mean, on cross validation, performance on user classification')
    parser.add_argument(
        '-save_plot',
        '--save_plot',
        type=str2bool,
        default=False
    )
    parser.add_argument(
        '-win_len',
        '--win_len',
        type=int,
        default=100
    )
    parser.add_argument(
    '-stride',
    '--stride',
    type=int,
    default=1
    )
    args = parser.parse_args()

    performance_for_activity = []  # list of shape (n_fold, n_activity)
    mean_performances = {
        'acc': 0,
        #'precision': 0,
        #'recall': 0,
        'f1': 0
    }

    if args.run_colab:
        colab_path = args.colab_path
    else:
        colab_path = None

    # GPU settings
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    for model_type in [args.model]:
        for dataset_name in [args.dataset]:
            for multitask in [False]:
                for overlap in [*args.overlap]:
                    for magnitude in [args.magnitude]:
                        for fold_test in [*args.fold_test]:
                            if magnitude:
                                if dataset_name == 'sbhar_six_adl':
                                    outer_dir = 'OuterPartition_magnitude_sbhar_six_adl_'
                                elif dataset_name == 'unimib_75w':
                                    outer_dir = 'OuterPartition_magnitude_wl_75_'
                                elif dataset_name == 'unimib_128w' or dataset_name == 'sbhar_128w':
                                    outer_dir = 'OuterPartition_magnitude_wl_128_'
                                elif 'realdisp' in dataset_name and args.win_len == 128:
                                    outer_dir = f"OuterPartition_magnitude_{'_'.join(args.sensor_displace)}_wl_128_"
                                elif 'realdisp' in dataset_name:
                                    outer_dir = f"OuterPartition_magnitude_{'_'.join(args.sensor_displace)}_"
                                else:
                                    outer_dir = 'OuterPartition_magnitude_'
                                save_dir = FOLDER_LOG + 'log_magnitude'
                            else:
                                outer_dir = 'OuterPartition_'
                                if dataset_name == 'unimib_128w':
                                    outer_dir = 'OuterPartition_wl_128_'
                                elif 'realdisp' in dataset_name:
                                    outer_dir = f"OuterPartition_{'_'.join(args.sensor_displace)}_"
                                save_dir = FOLDER_LOG + 'log_no_magnitude'
                            save_dir = 'log_prova'
                            model = Model(dataset_name=dataset_name,
                                          configuration_file=configuration,
                                          multi_task=multitask,
                                          lr='dynamic',
                                          model_type=model_type,
                                          fold_test=fold_test,
                                          save_dir=save_dir,
                                          outer_dir=outer_dir +
                                          str(overlap) + '/',
                                          overlap=overlap,
                                          magnitude=magnitude,
                                          init_lr=args.init_lr,
                                          drop_factor=args.drop_factor,
                                          drop_epoch=args.drop_epoch,
                                          path_best_model=args.path_best_model,
                                          log=args.log)

                            model.create_dataset(
                                args.run_colab, args.colab_path)

                            model.load_data(
                                only_acc=args.only_acc,
                                only_acc_gyro=args.only_acc_gyro,
                                realdisp='realdisp' in dataset_name)

                            if args.unify:
                                model.unify_act(
                                    model.configuration.sbhar_mapping[args.unify_method])

                            # plot original distribution data train and test
                            if args.plot_distribution:
                                model.plot_distribution_data(val_test=True)

                            if args.augmented:
                                model.augment_data(
                                    function_to_apply=args.aug_function,
                                    augmented_par=args.augmented_par,
                                    compose=args.compose,
                                    only_compose=args.only_compose,
                                    plot_augmented=args.plot_augmented,
                                    ratio_random_transformations=args.ratio,
                                    n_func_to_apply=args.n_func_to_apply)
                                #model.plot_distribution_data(val_test=False)

                            model.normalize_data()

                            # tf dataset to weight sample in train set
                            model.tf_dataset(
                                args.weighted_based_on, args.weighted)

                            model.build_model(stride=args.stride)
                            if args.print_model_summary:
                                model.print_model_summary()

                            if args.train:
                                model.loss_opt_metric()
                                model.train_model(args.epochs)
                                if args.plot_pred_base_act_val:
                                    _ = model.plot_pred_based_act(
                                        title='User classification accuracy based on activity on val set',
                                        test=False,
                                        colab_path=colab_path,
                                        file_name=f'val_fold_{fold_test}',
                                        save_plot=args.save_plot,
                                        show_plot=True)
                                performances = model.test_model(
                                    log=args.confusion_matrix)
                                for p, key in zip(
                                    performances, list(
                                        mean_performances.keys())):
                                    mean_performances[key] += p
                                pred_right = model.plot_pred_based_act(
                                    title='User classification accuracy based on activity on test set',
                                    test=True,
                                    colab_path=colab_path,
                                    file_name=f'test_fold_{fold_test}',
                                    save_plot=False,
                                    show_plot=args.plot_pred_based_act_test)
                                performance_for_activity.append(pred_right)
                                if args.save_best_model:
                                    model.best_model.save_weights(
                                        filepath=args.path_best_model, overwrite=True, save_format=None)

                        for k in list(mean_performances.keys()):
                            mean_performances[k] /= len([*args.fold_test])

                        print("Mean accuracy, f1 after cross validation: {} {}".format(mean_performances['acc'], mean_performances['f1']))

                        # mean performance for activity on different fold
                        if args.mean_perfomance_cross_validation and args.train:
                            if 'realdisp' in dataset_name:
                                dataset_name_plot = model.dataset_name_plot + f"_{'_'.join(args.sensor_displace)}_wl_{args.win_len}"
                            else:
                                dataset_name_plot = model.dataset_name_plot
                            plot_pred_based_act(
                                correct_predictions=performance_for_activity,
                                label_act=model.mapping_act_label(),
                                folds=len([*args.fold_test]),
                                title='Mean 10-cross user classification accuracy based on activity',
                                dataset_name=dataset_name_plot,
                                colab_path=colab_path,
                                file_name='mean',
                                save_plot=args.save_plot,
                                save_txt=True)
                                
                            if args.save_mean_perfomance_cross_validation:
                                save_mean_performance_txt(
                                    mean_performances,
                                    dataset_name=dataset_name_plot,
                                    colab_path=colab_path)
