import tensorflow as tf

'''
    A multi-head 1D-CNN, one cnn flow for every sensor in input.
    Output of head-CNN are merged and feed in another 1D-CNN.
    The output is feed in LSTM.
'''
class Resnet18MultiBranch(tf.keras.Model):
    def __init__(self, sensor_dict, num_user, magnitude):
        super(Resnet18MultiBranch, self).__init__()

        self.sensor_name = list(sensor_dict.keys())
        if magnitude:
            self.sensor_axes = [x+1 for x in list(sensor_dict.values())]
        else:
            self.sensor_axes = list(sensor_dict.values())
        self.num_user = num_user

        self.branches = []

        ### MULTI-HEAD CNN-1D ###
        for sensor_name in self.sensor_name:
            self.branches.append(new_Head_CNN(sensor_name))    

        ### CNN-1D on merged output heads if there are more than one sensor ###
        if len(self.sensor_name) > 1 :
            self.conv_merge1 = tf.keras.layers.Conv2D(filters=64,
                                                kernel_size=(5,1),
                                                strides=1,
                                                padding="same",
                                                name='conv_merge_1',
                                                kernel_regularizer=tf.keras.regularizers.l2)
            self.bn_merge1 = tf.keras.layers.BatchNormalization(name='bn_merge_1')
            self.conv_merge2 = tf.keras.layers.Conv2D(filters=128,
                                                kernel_size=(5,1),
                                                strides=1,
                                                padding="same",
                                                name='conv_merge_2',
                                                kernel_regularizer=tf.keras.regularizers.l2)
            self.bn_merge2 = tf.keras.layers.BatchNormalization(name='bn_merge_2')

        ### LSTM ###
        self.lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=128, dropout=0.2, return_sequences=True), merge_mode='concat')
        self.global_avg_pool_1d = tf.keras.layers.GlobalAveragePooling1D()

        self.fc_user = tf.keras.layers.Dense(units=num_user,
                                             activation=tf.keras.activations.softmax,
                                             name='fc_user')


    def call(self, inputs, training=None):
        '''
            input [batch_size, samples, axes, 1]
        '''

        #inputs = tf.reshape(inputs, [-1, inputs.shape[1], inputs.shape[2]])  # [batch, time_step, feaures]
        merge_branch = []

        print('inputs: {}'.format(inputs.shape))

        ### Mult-head CNN-1D ###

        if len(self.sensor_name) > 1:
            sensor_split = tf.split(inputs, num_or_size_splits=self.sensor_axes, axis=2)
            for input_sensor, branch in zip(sensor_split, self.branches):
                print('input single head cnn: {}'.format(input_sensor.shape))
                x = branch(input_sensor, training=training)
                print('output head: {}'.format(x.shape))
                merge_branch.append(x)
        else:
            x = self.branches[0](inputs, training=training)
            print('output head: {}'.format(x.shape))
            merge_branch.append(x)


        ### Merge output head-cnn on features axis ###
        if len(merge_branch) > 1:
            merge = tf.concat(merge_branch, axis=2) # [batch_size, time_step, features]
        else:
            merge = merge_branch[0]

        print('shape merged: {}'.format(merge.shape))

        ### CNN-1D on merged output ###
        if len(self.sensor_name) > 1:
            merge = self.conv_merge1(merge, training=training)
            merge = self.bn_merge1(merge)
            merge = tf.nn.relu(merge)
            merge = self.conv_merge2(merge, training=training)
            merge = self.bn_merge2(merge)
            merge = tf.nn.relu(merge)
            print('shape merged after conv : {}'.format(merge.shape))

        merge = tf.reshape(merge, [-1, merge.shape[1], merge.shape[2]*merge.shape[3]])

        lstm  = self.lstm(merge, training=training)
        print('shape after lstm: {}'.format(lstm.shape))

        lstm = self.global_avg_pool_1d(lstm)
        print('shape after global avg pool: {}'.format(lstm.shape))

        output_user = self.fc_user(lstm)
        return output_user

class resnet18Block(tf.keras.layers.Layer):
    def __init__(self):
        super(resnet18Block, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters=32,
                                            kernel_size=(5,1),
                                            strides=1,
                                            padding="valid",
                                            kernel_regularizer=tf.keras.regularizers.l2)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3,1),
                                               strides=(2,1),
                                               padding="valid")

        self.layer1 = make_basic_block_layer(filter_num=32,
                                             blocks=1,
                                             name='residual_block_1',
                                             kernel=(3,3))
        
        self.layer2 = make_basic_block_layer(filter_num=64,
                                             blocks=1,
                                             name='residual_block_2',
                                             stride=2,
                                             kernel=(3,3),
                                             downsample=True)
        

    def call(self, inputs, training=None):

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)

        return x

class BasicBlock(tf.keras.layers.Layer):

    def __init__(self, filter_num, kernel, stride=1, downsample=False):
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
        if downsample:
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


def new_Head_CNN(sensor_name):
    resnet_head = tf.keras.Sequential(name='resnet_branch_{}'.format(sensor_name))
    resnet_head.add(resnet18Block())
    return resnet_head

def resnet18MultiBranch(sensor_dict, num_user, magnitude):
    return Resnet18MultiBranch(sensor_dict, num_user, magnitude)
