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
        deafult=False
    )
    parser.add_argument(
        '-init_lr',
        '--init_lr',
        type=int, 
        deafult=0.001
    )
    parser.add_argument(
        '-log_train',
        '--log_train',
        type=str2bool, 
        deafult=False
    )    
    args = parser.parse_args()

    # GPU settings
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    model = ModelGait(config, args.colab_path)
    model.load_data()
    model.split_train_test()
    model.normalize_data()
    model.create_tf_dataset()
    model.build_model(summary=args.summary)
    model.loss_metric(init_lr=args.init_lr)
    model.train_model(log=args.log_train)
    model.test_model()