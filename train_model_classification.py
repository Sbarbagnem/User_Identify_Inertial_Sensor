from __future__ import absolute_import, division, print_function
import tensorflow as tf
import argparse
import sys

import configuration
from model.custom_model import Model
from util.utils import str2bool

FOLDER_LOG = 'log/'

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Arguments for train classification model")

    parser.add_argument(
        '-p',
        '--plot',
        type=str2bool,
        default=False,
        help='bool to plot or not distribution of train and test')
    parser.add_argument('-t', '--train', type=str2bool,
                        default=True, help='bool to train or not model')
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
        '-ppba',
        '--plot_pred_base_act',
        type=str2bool,
        default=True,
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
            'realdisp',
            'unimib_sbhar',
            'sbhar_six_adl'],
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
        '-compose_transformations',
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
    parser.add_argument('-fold_val', '--fold_val', type=int,
                        default=[], nargs='+', help='fold for validation', required=True)
    parser.add_argument(
        '-fold_test',
        '--fold_test',
        type=int,
        nargs='+',
        default=[],
        help='list of int represent folds on wich testing model')
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
            'resnet18_lstm_consecutive'],
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
        default=50,
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
    args = parser.parse_args()

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
                        if magnitude:
                            if dataset_name == 'sbhar_six_adl':
                                outer_dir = 'OuterPartition_magnitude_sbhar_six_adl_'
                            else:
                                outer_dir = 'OuterPartition_magnitude_'
                            save_dir = FOLDER_LOG + 'log_magnitude'
                        else:
                            outer_dir = 'OuterPartition_'
                            save_dir = FOLDER_LOG + 'log_no_magnitude'

                        save_dir = 'log'

                        # fold used as validation during training set
                        fold_val = args.fold_val
                        fold_test = args.fold_test

                        model = Model(dataset_name=dataset_name,
                                      configuration_file=configuration,
                                      multi_task=multitask,
                                      lr='dynamic',
                                      model_type=model_type,
                                      fold_val=fold_val,
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
                                      log=True)

                        model.create_dataset(args.run_colab, args.colab_path)

                        model.load_data(
                            only_acc=args.only_acc)

                        if args.unify:
                            model.unify_act(
                                model.configuration.sbhar_mapping[args.unify_method])

                        # plot original distribution data train and test
                        if args.plot:
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
                            model.plot_distribution_data(val_test=False)

                        model.normalize_data()

                        # tf dataset to weight sample in train set
                        model.tf_dataset(args.weighted_based_on, args.weighted)

                        if args.train:
                            model.build_model()
                            model.print_model_summary()
                            model.loss_opt_metric()
                            model.train_model(args.epochs)
                            if args.plot_pred_base_act:
                                model.plot_pred_based_act(title='percentage error in validation best seen')
                            if args.fold_test != []:
                                model.test_model()
                                model.plot_pred_based_act(title='percentage error in test set')
                            if args.save_best_model:
                                model.best_model.save_weights(filepath=args.path_best_model, overwrite=True, save_format=None)
                        # TODO
                        '''
                        if args.load_model:
                            # cose .......
                            model.load_model(path_model)
                            model.test_model()
                        '''
