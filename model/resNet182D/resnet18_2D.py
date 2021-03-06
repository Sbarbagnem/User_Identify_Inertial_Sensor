import tensorflow as tf
import sys

'''
    from https://github.com/calmisential/Basic_CNNs_TensorFlow2
'''


class ResNet18SingleBranch(tf.keras.Model):
    def __init__(self, multi_task, num_act, num_user, stride=1, fc=False, flatten=False, feature_generator=False):
        super(ResNet18SingleBranch, self).__init__()

        self.multi_task = multi_task
        if multi_task:
            self.num_act = num_act
        self.num_user = num_user
        self.fc = fc
        self.flatten = flatten
        self.feature_generator = feature_generator

        # features about single axis sensor (kernel len from metier paper)

        self.conv1 = tf.keras.layers.Conv2D(filters=32,
                                            kernel_size=(5,1),
                                            strides=(1,1),
                                            padding="valid",
                                            kernel_regularizer=tf.keras.regularizers.l2,
                                            name='conv2d_input'
                                            )
        self.bn1 = tf.keras.layers.BatchNormalization(name='bn_1')
        
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3,1),
                                               strides=(2,1),
                                               padding="valid",
                                               name='max_pool_1')
        
        # features about interaction between sensor

        self.layer1 = make_basic_block_layer(filter_num=32,
                                             blocks=2,
                                             name='residual_block_1',
                                             kernel=(3,3))
        self.layer2 = make_basic_block_layer(filter_num=64,
                                             blocks=2,
                                             name='residual_block_2',
                                             stride=stride,
                                             kernel=(3,3))

        if flatten:
            self.flatten = tf.keras.layers.Flatten()
        else:
            self.flatten = tf.keras.layers.GlobalAveragePooling2D(name='global_average_pooling')

        if self.fc:
            self.fc1 = tf.keras.layers.Dense(int(self.num_user*1.5), activation='relu')

        if not self.feature_generator:
            if multi_task:
                # activity classification
                self.fc_activity = tf.keras.layers.Dense(units=num_act,
                                                        activation=tf.keras.activations.softmax,
                                                        name='fc_act')
            # user classification
            self.fc_user = tf.keras.layers.Dense(units=num_user,
                                                activation=tf.keras.activations.softmax,
                                                name='fc_user')

    def call(self, inputs, training=None):

        #print('shape input: {}'.format(inputs.shape))

        ### CNN ###
        x = self.conv1(inputs)
        #print('shape conv1: {}'.format(x.shape))
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        #print('shape pool1: {}'.format(x.shape))
        x = self.layer1(x, training=training)
        #print('shape res_1: {}'.format(x.shape))
        x = self.layer2(x, training=training)
        #print('shape res_2: {}'.format(x.shape))
        out_cnn = self.flatten(x)
        #print('shape avg_pool: {}'.format(out_cnn.shape))
        if self.fc:
            out_cnn = self.fc1(out_cnn)
            #print('shape dense: {}'.format(out_cnn.shape))
        if not self.feature_generator:
            if self.multi_task:
                output_activity = self.fc_activity(out_cnn)
                output_user = self.fc_user(out_cnn)
                return output_activity, output_user
            else:
                output_user = self.fc_user(out_cnn)
                return output_user
        else:
            return out_cnn

    def build_graph(self, raw_shape):
        x = tf.keras.layers.Input(shape=raw_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))


class BasicBlock(tf.keras.layers.Layer):

    def __init__(self, filter_num, kernel, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=kernel,
                                            strides=stride,
                                            padding='same',
                                            kernel_regularizer=tf.keras.regularizers.l2,
                                            name='conv1'
                                            )
        self.bn1 = tf.keras.layers.BatchNormalization(name='bn1')
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=kernel,
                                            strides=1,
                                            padding="same",
                                            kernel_regularizer=tf.keras.regularizers.l2,
                                            name='conv2'
                                            )
        self.bn2 = tf.keras.layers.BatchNormalization(name='bn2')
        # doownsample per ristabilire dimensioni residuo tra un blocco e l'altro
        if stride != 1:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(tf.keras.layers.Conv2D(filters=filter_num,
                                                       kernel_size=(1, 1),
                                                       strides=stride,
                                                       name='conv_equal'))
            self.downsample.add(tf.keras.layers.BatchNormalization(name='bn_equal'))
        # all'interno del blocco le dimensioni del residuo in input sono le stesse
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)

        # relu su residuo + output conv
        output = tf.nn.relu(tf.keras.layers.add([residual, x]))

        return output


def make_basic_block_layer(filter_num, blocks, name, kernel, stride=1):
    res_block = tf.keras.Sequential(name=name)
    res_block.add(BasicBlock(filter_num, kernel=kernel, stride=stride))

    for _ in range(1, blocks):
        res_block.add(BasicBlock(filter_num, kernel=kernel, stride=1))

    return res_block


def resnet18(multi_task, num_act, num_user, stride=1, fc=False, flatten=False, feature_generator=False):
    return ResNet18SingleBranch(multi_task=multi_task, num_act=num_act, num_user=num_user, stride=stride, fc=fc, flatten=flatten, feature_generator=feature_generator)