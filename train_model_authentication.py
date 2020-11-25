import argparse
import tensorflow as tf

import sys

from model.model_authentication import ModelAuthentication
from util.utils import str2bool

# date le feature distanza euclidea (output arrray con gallery, probe, distance, same_user(0,1), same_activity(0,1))

parser = argparse.ArgumentParser(description = "Parameter to train and evaluate autentication")

parser.add_argument(
    '-path_data',
    '--path_data',
    type=str
)
parser.add_argument(
    '-name_dataset',
    '--name_dataset',
    type=str
)
parser.add_argument(
    '-name_model',
    '--name_model',
    type=str
)
parser.add_argument(
    '-train_classifier',
    '--train_classifier',
    type=str2bool,
    default=False
)
parser.add_argument(
    '-generate_features',
    '--generate_features',
    type=str2bool,
    default=False
)
parser.add_argument(
    '-compute_distance',
    '--compute_distance',
    type=str2bool,
    default=False
)
parser.add_argument(
    '-compute_eer',
    '--compute_eer',
    type=str2bool,
    default=False
)
parser.add_argument(
    '-action_dependent',
    '--action_dependent',
    type=str2bool,
    default=False
)
parser.add_argument(
    '-preprocess_features',
    '--preprocess_features',
    type=str2bool,
    default=False
)
parser.add_argument(
    '-split_gallery_probe',
    '--split_gallery_probe',
    type=str,
    choices=['random', 'intra_session', 'extra_session']
)

args = parser.parse_args()

# GPU settings
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

path_data = args.path_data
name_dataset = args.name_dataset
name_model = args.name_model
train_classifier = args.train_classifier
generate_features = args.generate_features
compute_distance = args.compute_distance
compute_eer = args.compute_eer
action_dependent = args.action_dependent
split_gallery_probe = args.split_gallery_probe
preprocess_features = args.preprocess_features

model = ModelAuthentication(path_data,name_dataset, name_model)
model.load_data()
model.split_user()
if train_classifier:
    model.create_dataset_classifier()
    model.build_model()
    model.loss_opt_metric()
    model.train_model(log=True)
    model.save_model()
else:
    model.load_model()
if generate_features:
    model.generate_features(split_gallery_probe)
if compute_distance:
    model.compute_distance_gallery_probe(split_gallery_probe, action_dependent, preprocess_features)
if compute_eer:
    model.compute_eer(split_gallery_probe, action_dependent)