import tensorflow as tf

'''
    from https://github.com/calmisential/Basic_CNNs_TensorFlow2
'''


class ResNet18SingleBranchLSTM(tf.keras.Model):
    '''
        parallel simplify Resnet18 and LSTM
    '''

    def __init__(self, layer_params, multi_task, num_act, num_user):
        super(ResNet18SingleBranchLSTM, self).__init__()
  
        self.multi_task = multi_task
        if multi_task:
            self.num_act = num_act
        self.num_user = num_user

        # features about single axis sensor (parameters from metier)

        self.conv1 = tf.keras.layers.Conv2D(filters=32,
                                            kernel_size=(5, 1),
                                            strides=(1, 1),
                                            padding="valid",
                                            kernel_regularizer=tf.keras.regularizers.l2)
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 1),
                                               strides=(2, 1),
                                               padding="valid")
        # features about interaction between sensor

        self.layer1 = make_basic_block_layer(filter_num=32,
                                             blocks=2,
                                             name='residual_block_1',
                                             kernel=(3, 3))
        self.layer2 = make_basic_block_layer(filter_num=64,
                                             blocks=2,
                                             name='residual_block_2',
                                             stride=1,
                                             kernel=(3, 3))

        self.avgpool_2d = tf.keras.layers.GlobalAveragePooling2D()

        # LSTM
        lstm_forward = tf.keras.layers.LSTM(units=128, dropout=0.2, recurrent_dropout=0.0,
                                            return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2)
        lstm_backward = tf.keras.layers.LSTM(units=128, dropout=0.2, recurrent_dropout=0.0, 
                                            return_sequences=True, go_backwards=True, kernel_regularizer=tf.keras.regularizers.l2)
        self.lstm_bidirectional=tf.keras.layers.Bidirectional(
            layer = lstm_forward, merge_mode = 'concat', backward_layer = lstm_backward)
        self.avgpool_1d=tf.keras.layers.GlobalAveragePooling1D()

        if multi_task:
            # activity classification
            self.fc_activity=tf.keras.layers.Dense(units = num_act,
                                                     activation = tf.keras.activations.softmax,
                                                     name = 'fc_act')

        # user classification
        self.fc_user=tf.keras.layers.Dense(units = num_user,
                                             activation=tf.keras.activations.softmax,
                                             name='fc_user')

    def call(self, inputs, training=None):
        
        print('shape input: {}'.format(inputs.shape))

        ### CNN head ###
        x = self.conv1(inputs)
        print('shape conv1: {}'.format(x.shape))
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        print('shape pool1: {}'.format(x.shape))
        x = self.layer1(x, training=training)
        print('shape res_1: {}'.format(x.shape))
        x = self.layer2(x, training=training)
        print('shape res_2: {}'.format(x.shape))
        out_cnn = self.avgpool_2d(x)
        print('shape avg_pool: {}'.format(out_cnn.shape))

        ### LSTM head ###       
        input_lstm = tf.reshape(inputs, [-1, inputs.shape[1], inputs.shape[2]*inputs.shape[3]])
        print('input LSTM: {}'.format(input_lstm.shape))
        out_lstm = self.lstm_bidirectional(input_lstm, training=training)
        print('output LSTM: {} '.format(out_lstm.shape))
        out_lstm = self.avgpool_1d(out_lstm)
        print('output avg lstm: {} '.format(out_lstm.shape))

        # MERGE CNN and LSTM flatten output
        merge = tf.concat([out_cnn,out_lstm], axis=1)
        print('shape merge: {}'.format(merge.shape))

        if self.multi_task:
            output_activity = self.fc_activity(merge)
            output_user = self.fc_user(merge)
            return output_activity, output_user
        else:
            output_user = self.fc_user(merge)
            return output_user


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


def resnet18_lstm(multi_task, num_act, num_user):
    return ResNet18SingleBranchLSTM(layer_params=[2, 2, 2, 2], multi_task=multi_task, num_act=num_act, num_user=num_user)
