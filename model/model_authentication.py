from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
import pprint
import seaborn as sn
import pandas as pd

from model.resNet182D.resnet18_2D import resnet18 as resnet2D
from util.tf_metrics import custom_metrics
from util.data_augmentation import random_transformation
from util.utils import mapping_act_label, plot_pred_based_act


class ModelAuthentication():
    def __init__(self):
        pass

    def load_data(self):
        """
        Load data and split: 70% of user are using to train feature extraction model,
        30% of user are using to evaluate authentication task.
        """
        pass

    def tf_dataset(self):
        """
        From numpy to tensorflow data to train.
        """
        pass

    def normalize_data(self):
        pass

    def augment_data(self):
        pass

    def build_model(self, stride=1, fc=False):
        pass

    def loss_opt_metric(self):
        """
        Define metric, optimizer and loss function.
        """
        pass

    @tf.function
    def train_step(self):
        pass

    @tf.function
    def valid_step(self):
        pass

    def train_model(self, epochs):
        pass

    def decay_lr_on_plateau(self):
        pass

    def test_model(self, log=False):
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass

    def generate_features(self):
        pass

    def test_authentication(self):
        pass
    
    def evaluate_authentication(self):
        pass
