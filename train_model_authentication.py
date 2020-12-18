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
    type=str,
    help='Path to read data',
    required=True
)
parser.add_argument(
    '-path_out',
    '--path_out',
    type=str,
    help='Path to store result about authentication: features, distances and eer',
    required=True
)
parser.add_argument(
    '-name_dataset',
    '--name_dataset',
    type=str,
    help='Name used for sub-dir in saved model directory',
)
parser.add_argument(
    '-name_model',
    '--name_model',
    type=str,
    help='Name of the model to save'
)
parser.add_argument(
    '-train_classifier',
    '--train_classifier',
    type=str2bool,
    default=False,
    help='Train or not feature extractor'
)
parser.add_argument(
    '-gyroscope',
    '--gyroscope',
    type=str2bool,
    default=False,
    help='Use or not gyroscope data'
)
parser.add_argument(
    '-generate_features',
    '--generate_features',
    type=str2bool,
    default=False,
    help='Generate feature from subset user of authentication'
)
parser.add_argument(
    '-compute_distance',
    '--compute_distance',
    type=str2bool,
    default=False,
    help='Compute distance between features of gallery and probe'
)
parser.add_argument(
    '-compute_eer',
    '--compute_eer',
    type=str2bool,
    default=False,
    help='Compute eer based on distance files'
)
parser.add_argument(
    '-action_dependent',
    '--action_dependent',
    type=str2bool,
    default=False,
    help='Comupte eer based on action or not'
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
    choices=['random', 'intra_session', 'extra_session'],
    help='intra_session: half of session for gallery and half for probe (time sorted),\nextra_session: session 1 for gallery and session 2 for probe,\nrandom: random between session'
)
parser.add_argument(
    '-colab_path',
    '--colab_path',
    type=str,
    default=''
)
parser.add_argument(
    '-augment_data',
    '--augment_data',
    type=str2bool
)
parser.add_argument(
    '-load_model',
    '--load_model',
    type=str2bool
)
parser.add_argument(
    '-overlap',
    '--overlap',
    type=float
)
parser.add_argument(
    '-method',
    '--method',
    type=str,
    choices=['window_based', 'cycle_based'],
    default='',
    help='Based on preprocessing method delete overlapping between train and test (window_based) or not (cycle_based)'
)
parser.add_argument(
    '-split_method',
    '--split_method',
    type=str,
    default='',
    help='If paper: 8 sample for train, and rest for val; if standard: 70 percentage for train and rest for val'
)

args = parser.parse_args()

# GPU settings
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

path_data = args.path_data
path_out = args.path_out
name_dataset = args.name_dataset
name_model = args.name_model
train_classifier = args.train_classifier
generate_features = args.generate_features
compute_distance = args.compute_distance
compute_eer = args.compute_eer
action_dependent = args.action_dependent
split_gallery_probe = args.split_gallery_probe
preprocess_features = args.preprocess_features
colab_path = args.colab_path
augment_data = args.augment_data
load_model = args.load_model
overlap = args.overlap
method = args.method
split_method = args.split_method
gyroscope = args.gyroscope

if train_classifier:
    if method == '' or split_method == '':
        sys.exit('For train classifier method (cycle or window) and split (stanrd or paper) must be defined')

model = ModelAuthentication(path_data, path_out, name_dataset, name_model, overlap, colab_path)
model.load_data(gyroscope)
model.split_user()
if train_classifier:
    model.split_train_test_classifier(split_method, method)
    if augment_data:
        model.augment_train_data()
    model.normalize_data()
    model.create_dataset_classifier()
    model.build_model()
    model.loss_opt_metric()
    model.train_model(log=True)
    model.save_model()
if load_model:
    model.load_model()
if generate_features:
    model.generate_features(split_gallery_probe)
if compute_distance:
    model.compute_distance_gallery_probe(split_gallery_probe, action_dependent, preprocess_features)
if compute_eer:
    model.compute_eer(split_gallery_probe, action_dependent)