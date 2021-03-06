import tensorflow as tf


class Resne18MultiTask(tf.keras.Model):
    def __init__(self, layer_params, num_act, num_user):
        super(Resne18MultiTask, self).__init__()

        self.num_act = num_act
        self.num_user = num_user

        ### USER NET ###
        self.conv1_user = tf.keras.layers.Conv2D(filters=32,
                                            kernel_size=(5,1),
                                            strides=(1,1),
                                            padding="valid",
                                            kernel_regularizer=tf.keras.regularizers.l2)
        self.bn1_user = tf.keras.layers.BatchNormalization()      
        self.pool1_user = tf.keras.layers.MaxPool2D(pool_size=(3,1),
                                               strides=(2,1),
                                               padding="valid")
        self.layer1_user = make_basic_block_layer(filter_num=32,
                                             blocks=2,
                                             name='residual_block_1',
                                             kernel=(3,3))
        self.layer2_user = make_basic_block_layer(filter_num=64,
                                             blocks=2,
                                             name='residual_block_2',
                                             stride=1,
                                             kernel=(3,3))
        self.avgpool_2d_user = tf.keras.layers.GlobalAveragePooling2D()

        ### ACTIVITY NET ###
        self.conv1_act = tf.keras.layers.Conv2D(filters=32,
                                            kernel_size=(5,1),
                                            strides=(1,1),
                                            padding="valid",
                                            kernel_regularizer=tf.keras.regularizers.l2)
        self.bn1_act = tf.keras.layers.BatchNormalization()
        self.pool1_act = tf.keras.layers.MaxPool2D(pool_size=(3,1),
                                               strides=(2,1),
                                               padding="valid")
        self.layer1_act = make_basic_block_layer(filter_num=32,
                                             blocks=2,
                                             name='residual_block_1',
                                             kernel=(3,3))
        self.layer2_act = make_basic_block_layer(filter_num=64,
                                             blocks=2,
                                             name='residual_block_2',
                                             stride=1,
                                             kernel=(3,3))
        self.avgpool_2d_act = tf.keras.layers.GlobalAveragePooling2D()

        self.fc_activity = tf.keras.layers.Dense(units=num_act,
                                                     activation=tf.keras.activations.softmax,
                                                     name='fc_act')
        self.fc_user = tf.keras.layers.Dense(units=num_user,
                                             activation=tf.keras.activations.softmax,
                                             name='fc_user')

    def call(self, inputs, training=None):

        #print('shape input: {}'.format(inputs.shape))

        ### USER NET ###
        x = self.conv1_user(inputs)
        #print('shape conv1: {}'.format(x.shape))
        x = self.bn1_user(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1_user(x)
        #print('shape pool1: {}'.format(x.shape))
        x = self.layer1_user(x, training=training)
        #print('shape res_1: {}'.format(x.shape))
        x = self.layer2_user(x, training=training)
        #print('shape res_2: {}'.format(x.shape))
        out_cnn_user = self.avgpool_2d_user(x)
        #print('shape avg_pool: {}'.format(out_cnn_user.shape))

        ### ACTIVITY NET ###
        x = self.conv1_act (inputs)
        #print('shape conv1: {}'.format(x.shape))
        x = self.bn1_act(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1_act(x)
        #print('shape pool1: {}'.format(x.shape))
        x = self.layer1_act(x, training=training)
        #print('shape res_1: {}'.format(x.shape))
        x = self.layer2_act(x, training=training)
        #print('shape res_2: {}'.format(x.shape))
        out_cnn_activity = self.avgpool_2d_act(x)
        #print('shape avg_pool: {}'.format(out_cnn.shape))

        out_cnn_user = tf.math.multiply(out_cnn_user, out_cnn_activity)

        output_activity = self.fc_activity(out_cnn_activity)
        output_user = self.fc_user(out_cnn_user)
        return output_activity, output_user


class BasicBlock(tf.keras.layers.Layer):

    def __init__(self, filter_num, kernel, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=kernel,
                                            strides=stride,
                                            padding='same',
                                            kernel_regularizer=tf.keras.regularizers.l2)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=kernel,
                                            strides=1,
                                            padding="same",
                                            kernel_regularizer=tf.keras.regularizers.l2)
        self.bn2 = tf.keras.layers.BatchNormalization()
        # doownsample per ristabilire dimensioni residuo tra un blocco e l'altro
        if stride != 1 or kernel[0]!=1:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(tf.keras.layers.Conv2D(filters=filter_num,
                                                       kernel_size=(1, 1),
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


def make_basic_block_layer(filter_num, blocks, name, kernel, stride=1):
    res_block = tf.keras.Sequential(name=name)
    res_block.add(BasicBlock(filter_num, kernel=kernel, stride=stride))

    for _ in range(1, blocks):
        res_block.add(BasicBlock(filter_num, kernel=kernel, stride=1))

    return res_block


def resne18MultiTask(num_act, num_user):
    return Resne18MultiTask(layer_params=[2, 2, 2, 2], num_act=num_act, num_user=num_user)
