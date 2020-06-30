import tensorflow as tf


'''

    Same structure as resnet18 but with conv1D instead conv2D
    Input shape is (batch, time_step, axes)

'''

class ResNet18SingleBranch(tf.keras.Model):
    def __init__(self, layer_params, multi_task, num_act, num_user, axes):
        super(ResNet18SingleBranch, self).__init__()

        self.multi_task = multi_task
        if multi_task:
            self.num_act = num_act
        self.num_user = num_user
        self.axes= axes

        self.conv1 = tf.keras.layers.Conv1D(filters=32,
                                            kernel_size=5,
                                            strides=1,
                                            padding="valid",
                                            kernel_regularizer=tf.keras.regularizers.l2)
        self.bn1 = tf.keras.layers.BatchNormalization()
        
        self.pool1 = tf.keras.layers.MaxPool1D(pool_size=3,
                                               strides=2,
                                               padding="valid")
        self.layer1 = make_basic_block_layer(filter_num=32,
                                             blocks=2,
                                             name='residual_block_1',
                                             kernel=3)
        self.layer2 = make_basic_block_layer(filter_num=64,
                                             blocks=2,
                                             name='residual_block_2',
                                             kernel=3,
                                             downsample=True,
                                             stride=2)
        self.avgpool = tf.keras.layers.GlobalAveragePooling1D()

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

        print('shape input: {}'.format(inputs.shape))

        # reshape input for conv1d layer
        x = tf.reshape(inputs, [-1, inputs.shape[1], inputs.shape[2]])

        print('shape after reshape and transpose: {}'.format(x.shape))

        ### CNN ###
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        print('shape conv1: {}'.format(x.shape))
        x = self.layer1(x, training=training)
        print('shape res_1: {}'.format(x.shape))
        x = self.layer2(x, training=training)
        print('shape res_2: {}'.format(x.shape))
        x = self.layer3(x, training=training)
        print('shape res_3: {}'.format(x.shape))

        x = self.avgpool(x)
        
        if self.multi_task:
            output_activity = self.fc_activity(x)
            output_user = self.fc_user(x)
            return output_activity, output_user
        else:
            output_user = self.fc_user(x)
            return output_user


class BasicBlock(tf.keras.layers.Layer):

    def __init__(self, filter_num, kernel, stride=1, downsample=False):
        super(BasicBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv1D(filters=filter_num,
                                            kernel_size=kernel,
                                            strides=stride,
                                            padding='same',
                                            kernel_regularizer=tf.keras.regularizers.l2)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv1D(filters=filter_num,
                                            kernel_size=kernel,
                                            strides=1,
                                            padding="same",
                                            kernel_regularizer=tf.keras.regularizers.l2)
        self.bn2 = tf.keras.layers.BatchNormalization()
        # doownsample per ristabilire dimensioni residuo tra un blocco e l'altro
        if downsample:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(tf.keras.layers.Conv1D(filters=filter_num,
                                                       kernel_size=1,
                                                       strides=stride))
            self.downsample.add(tf.keras.layers.BatchNormalization())
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


def make_basic_block_layer(filter_num, blocks, name, kernel, stride=1, downsample=False):
    res_block = tf.keras.Sequential(name=name)
    res_block.add(BasicBlock(filter_num, kernel=kernel, stride=stride, downsample=downsample))

    for _ in range(1, blocks):
        res_block.add(BasicBlock(filter_num, kernel=kernel, stride=1, downsample=False))

    return res_block


def resnet18(multi_task, num_act, num_user, axes):
    return ResNet18SingleBranch(layer_params=[2, 2, 2, 2], multi_task=multi_task, num_act=num_act, num_user=num_user, axes=axes)
