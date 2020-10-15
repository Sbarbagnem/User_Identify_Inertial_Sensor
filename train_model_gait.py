import argparse
import sys
import tensorflow as tf

from model.model_gait import ModelGait
from configuration import config

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
    model.build_model(summary=True)
    model.loss_metric(init_lr=0.1)
    model.train_model(log=True)
    model.test_model()