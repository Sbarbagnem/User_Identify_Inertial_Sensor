import argparse
import sys
import tensorflow as tf

from model.model_gait import ModelGait
from configuration import config
from util.utils import str2bool

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Arguments for train classification model")

    parser.add_argument(
        '-colab_path',
        '--colab_path',
        type=str,
        default='',
        help='colab path to data'
    )
    parser.add_argument(
        '-summary_model',
        '--summary_model',
        type=str2bool, 
        default=False
    )
    parser.add_argument(
        '-init_lr',
        '--init_lr',
        type=float, 
        default=0.001
    )
    parser.add_argument(
        '-log_train',
        '--log_train',
        type=str2bool, 
        default=False
    )    
    parser.add_argument(
        '-stride',
        '--stride',
        default=1,
        type=int
    )
    parser.add_argument(
        '-fc',
        '--fc',
        type=str2bool,
        default=False
    )
    parser.add_argument(
        '-train',
        '--train',
        type=str2bool,
        default=True
    )
    parser.add_argument(
        '-epochs',
        '--epochs',
        default=100,
        type=int
    )
    parser.add_argument(
        '-filter_num_user',
        '--filter_num_user',
        default=None,
        type=int
    )
    parser.add_argument(
        '-model'
        '--model',
        type=str,
        default='our',
        choices=['our', 'paper']
    )
    parser.add_argument(
        '-batch_size',
        '--batch_size',
        type=int,
        default=128
    )
    args = parser.parse_args()

    # GPU settings
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    model = ModelGait(config, args.colab_path)
    model.load_data(filter_num_user=args.filter_num_user)
    model.split_train_test()
    model.normalize_data()
    model.create_tf_dataset(batch_size=args.batch_size)
    model.build_model(stride=args.stride, fc=args.fc, summary=args.summary_model, name=args.model)
    if args.train:
        model.loss_metric(init_lr=args.init_lr)
        model.train_model(log=args.log_train, epochs=args.epochs)
        model.test_model()