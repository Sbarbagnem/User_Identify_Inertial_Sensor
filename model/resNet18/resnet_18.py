import tensorflow as tf
from model.resNet18.configuration import config
from model.resNet18.residual_block import make_basic_block_layer

'''
    from https://github.com/calmisential/Basic_CNNs_TensorFlow2
'''


class ResNetTypeI(tf.keras.Model):
    def __init__(self, layer_params, dataset, multi_task):
        super(ResNetTypeI, self).__init__()

        self.multi_task = multi_task

        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(7, 7),
                                            strides=2,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                               strides=2,
                                               padding="same")

        self.layer1 = make_basic_block_layer(filter_num=64,
                                             blocks=layer_params[0],
                                             name='residual_block_1')
        self.layer2 = make_basic_block_layer(filter_num=128,
                                             blocks=layer_params[1],
                                             name='residual_block_2',
                                             stride=2)
        self.layer3 = make_basic_block_layer(filter_num=256,
                                             blocks=layer_params[2],
                                             name='residual_block_3',
                                             stride=2)
        self.layer4 = make_basic_block_layer(filter_num=512,
                                             blocks=layer_params[3],
                                             name='residual_block_4',
                                             stride=2)

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()

        if multi_task:
            # activity classification
            self.fc_activity = tf.keras.layers.Dense(units=config[dataset]['NUM_CLASSES_ACTIVITY'], 
                                                    activation=tf.keras.activations.softmax, 
                                                    name='fc_act')

        # user classification
        self.fc_user = tf.keras.layers.Dense(units=config[dataset]['NUM_CLASSES_USER'], 
                                             activation=tf.keras.activations.softmax, 
                                             name='fc_user')

    def call(self, inputs, training=None):

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)

        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)

        x = self.avgpool(x)

        if self.multi_task:       
            output_activity = self.fc_activity(x)
            output_user = self.fc_user(x)
            return output_activity, output_user
        else:
            output_user = self.fc_user(x)
            return output_user           

def resnet_18(dataset, multi_task):
    return ResNetTypeI(layer_params=[2, 2, 2, 2], dataset=dataset, multi_task=multi_task)