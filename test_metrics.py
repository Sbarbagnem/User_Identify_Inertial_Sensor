import tensorflow as tf
import numpy as np
from util.tf_metrics import custom_metrics
import pprint

A=tf.constant([0,1,2,3,0,0,1,2,3,3])
B=tf.constant([[0.3,0.2,0.1,0.2],
                [0.1,0.4,0.1,0.2],
                [0.3,0.4,0.1,0.2],
                [0.3,0.2,0.1,0.1],
                [0.4,0.2,0.1,0.2],
                [0.3,0.2,0.9,0.2],
                [0.3,0.5,0.1,0.2],
                [0.3,0.2,0.6,0.2],
                [0.3,0.2,0.1,0.8],
                [0.3,0.9,0.1,0.2]], dtype=tf.float64)

cm = tf.math.confusion_matrix(A, tf.math.argmax(B,axis=1))
metrics = custom_metrics(cm)

pprint.pprint(metrics)
