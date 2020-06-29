import tensorflow as tf

class SingleLSTM(tf.keras.Model):
    def __init__(self, num_user):
        super(SingleLSTM, self).__init__()

        self.num_user = num_user

        self.lstm = tf.keras.layers.LSTM(128, return_sequences=True)

        self.conv1 = tf.keras.layers.Conv2D(filters=32,
                                            kernel_size=(5,1),
                                            strides=(1,1),
                                            padding="valid",
                                            kernel_regularizer=tf.keras.regularizers.l2)
        self.bn1 = tf.keras.layers.BatchNormalization()
        
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2,1),
                                               strides=(2,1),
                                               padding="valid")
        # features about interaction between sensor

        self.layer1 = make_basic_block_layer(filter_num=32,
                                             blocks=2,
                                             name='residual_block_1',
                                             kernel=(3,3))
        self.layer2 = make_basic_block_layer(filter_num=64,
                                             blocks=2,
                                             name='residual_block_2',
                                             stride=1,
                                             kernel=(3,3),
                                             downsample=True)

        self.avgpool_2d = tf.keras.layers.GlobalAveragePooling2D()

        self.fc_user = tf.keras.layers.Dense(units=num_user,
                                             activation=tf.keras.activations.softmax,
                                             name='fc_user')

    def call(self, inputs, training=None, **kwargs):
        x = tf.reshape(inputs, [-1, inputs.shape[1], inputs.shape[2]*inputs.shape[3]])
        x = self.lstm(x, training=training)
        print('output lstm: {}'.format(x.shape))

        x = tf.expand_dims(x, 3) # add channel=1 for conv2d

        x = self.conv1(x)
        print('shape conv1: {}'.format(x.shape))
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        print('shape pool1: {}'.format(x.shape))
        x = self.layer1(x, training=training)
        print('shape res_1: {}'.format(x.shape))
        x = self.layer2(x, training=training)
        print('shape res_2: {}'.format(x.shape))

        x = self.avgpool_2d(x)
        print('output avg pool: {}'.format(x.shape))

        x = self.fc_user(x)
        return x

def create_single_lstm(num_user):
    return SingleLSTM(num_user)

class BasicBlock(tf.keras.layers.Layer):

    def __init__(self, filter_num, kernel, stride=1, downsample=False):
        super(BasicBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=kernel,
                                            strides=stride,
                                            padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=kernel,
                                            strides=1,
                                            padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()
        # doownsample per ristabilire dimensioni residuo tra un blocco e l'altro
        if stride != 1 or downsample:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(tf.keras.layers.Conv2D(filters=filter_num,
                                                       kernel_size=(1,1),
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
        res_block.add(BasicBlock(filter_num, kernel=kernel, stride=1))

    return res_block