import tensorflow as tf

'''
    from https://github.com/calmisential/Basic_CNNs_TensorFlow2
'''


class ResNet18SingleBranchLSTM(tf.keras.Model):
    '''
        consecutive simplify Resnet18 and LSTM
    '''
    def __init__(self, layer_params, multi_task, num_act, num_user, axes):
        super(ResNet18SingleBranchLSTM, self).__init__()

        self.multi_task = multi_task
        if multi_task:
            self.num_act = num_act
        self.num_user = num_user
        self.axes= axes

        # CNN for extract features about single axes sonsors

        self.conv1 = tf.keras.layers.Conv2D(filters=32,
                                            kernel_size=(1,5),
                                            strides=(1,1),
                                            padding="valid")
        self.bn1 = tf.keras.layers.BatchNormalization()
        
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(1,2),
                                               strides=(1,2),
                                               padding="valid")

        self.layer1 = make_basic_block_layer(filter_num=32,
                                             blocks=2,
                                             name='residual_block_1',
                                             kernel=(1, 3))
                                       
        self.layer2 = make_basic_block_layer(filter_num=64,
                                             blocks=2,
                                             name='residual_block_2',
                                             stride=1,
                                             kernel=(1,3),
                                             downsample=True)

        # LSTM
        lstm_forward = tf.keras.layers.LSTM(units=128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True, time_major=False) 
        lstm_backward = tf.keras.layers.LSTM(units=128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True, go_backwards=True, time_major=False)
        self.lstm_bidirectional = tf.keras.layers.Bidirectional(layer=lstm_forward, merge_mode='concat', backward_layer=lstm_backward)
        self.avg_pol_1d = tf.keras.layers.GlobalAveragePooling1D()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')

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

        ### CNN ###
        x = self.conv1(inputs)
        print('shape conv1: {}'.format(x.shape))
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        print('shape pool1: {}'.format(x.shape))
        x = self.layer1(x, training=training)
        print('shape res_1: {}'.format(x.shape))
        out_cnn = self.layer2(x, training=training)
        print('shape res_2: {}'.format(out_cnn.shape))

        ### LSTM ###
        input_lstm = tf.transpose(out_cnn, [0,2,1,3])
        input_lstm = tf.reshape(input_lstm, [-1, input_lstm.shape[1], input_lstm.shape[2]*input_lstm.shape[3]])
        print('input LSTM: {}'.format(input_lstm.shape))
        out_lstm = self.lstm_bidirectional(input_lstm, training=training)
        print('output LSTM: {} '.format(out_lstm.shape))

        ### FULLY CONNECTED ###
        out_fully = self.dense1(self.avg_pol_1d(out_lstm), training=training)

        if self.multi_task:
            output_activity = self.fc_activity(out_fully)
            output_user = self.fc_user(out_fully)
            return output_activity, output_user
        else:
            output_user = self.fc_user(out_fully)
            return output_user


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


def make_basic_block_layer(filter_num, blocks, name, kernel, stride=1, downsample=False):
    res_block = tf.keras.Sequential(name=name)
    res_block.add(BasicBlock(filter_num, kernel=kernel, stride=stride, downsample=downsample))

    for _ in range(1, blocks):
        res_block.add(BasicBlock(filter_num, kernel=kernel, stride=1))

    return res_block


def resnet18_lstm(multi_task, num_act, num_user, axes):
    return ResNet18SingleBranchLSTM(layer_params=[2, 2, 2, 2], multi_task=multi_task, num_act=num_act, num_user=num_user, axes=axes)
