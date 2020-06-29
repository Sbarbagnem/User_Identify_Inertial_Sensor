import tensorflow as tf

'''
    A multi-head 1D-CNN, one cnn flow for every sensor in input.
    Output of everu CNN is merge and then feed in a LSTM.
    Before output there is a FC layer of 1000 neurons.

    Ispired by: https://www.researchgate.net/publication/339946359_Human_Activity_Recognition_using_Multi-Head_CNN_followed_by_LSTM
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
            #self.branches.append(resnet18BlockNoClass(sensor_name))
            self.branches.append(new_head(sensor_name))    

        ### LSTM on merged output cnn ###
        self.lstm = tf.keras.layers.LSTM(units=128)

        ### FC before output ###
        self.fc = tf.keras.layers.Dense(256, activation='relu')

        self.fc_user = tf.keras.layers.Dense(units=num_user,
                                             activation=tf.keras.activations.softmax,
                                             name='fc_user')


    def call(self, inputs, training=None):
        '''
            input [batch_size, samples, axes, 1]
        '''

        inputs = tf.reshape(inputs, [-1, inputs.shape[1], inputs.shape[2]*inputs.shape[3]]) # [batch, time_step, feaures]
        merge_branch = []

        if len(self.sensor_name)>1:
            sensor_split = tf.split(inputs, num_or_size_splits=self.sensor_axes, axis=2)
            for input_sensor, branch in zip(sensor_split, self.branches):
                print(input_sensor.shape)
                x = branch(input_sensor, training=training)
                print('output head: {}'.format(x.shape))
                #x = tf.reshape(x, shape=[-1, 1, x.shape[1]*x.shape[2]*x.shape[3], 1]) # flatten
                merge_branch.append(x)
        else:
            x = self.branches[0](inputs, training=training)
            #x = tf.reshape(x, shape=[-1, 1, x.shape[1]*x.shape[2]*x.shape[3], 1]) # flatten
            merge_branch.append(x)


        # merge output
        if len(merge_branch) > 1:
            merge = tf.concat(merge_branch, axis=2) # [batch_size, time_step, features]
        else:
            merge = merge_branch[0]

        print('shape merged: {}'.format(merge.shape))

        x  = self.lstm(merge, training=training)
        print('shape after lstm: {}'.format(x.shape))

        x  = self.fc(x, training=training)

        x = tf.nn.dropout(x, rate=0.3)

        output_user = self.fc_user(x)
        return output_user

class Resnet18BlockNoClass(tf.keras.layers.Layer):
    def __init__(self, sensor_name):
        super(Resnet18BlockNoClass, self).__init__()

        self.conv1 = tf.keras.layers.Conv1D(filters=32,
                                            kernel_size=5,
                                            strides=1,
                                            padding="valid")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool1D(pool_size=2,
                                               strides=2,
                                               padding="valid")

        # features about interaction between sensor's axes

        self.layer1 = make_basic_block_layer(filter_num=32,
                                             blocks=2,
                                             name='residual_block_1',
                                             kernel=3)
        self.layer2 = make_basic_block_layer(filter_num=64,
                                             blocks=2,
                                             name='residual_block_2',
                                             stride=1,
                                             kernel=3,
                                             downsample=True)

    def call(self, inputs, training=None):

        #print('shape input: {}'.format(inputs.shape))
        x = self.conv1(inputs)
        #print('shape conv1: {}'.format(x.shape))
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        #print('shape pool1: {}'.format(x.shape))
        x = self.layer1(x, training=training)
        #print('shape res_1: {}'.format(x.shape))
        x = self.layer2(x, training=training)
        #print('shape avg_pool: {}'.format(x.shape))

        return x

class BasicBlock(tf.keras.layers.Layer):

    def __init__(self, filter_num, kernel, stride=1, downsample=False):
        super(BasicBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv1D(filters=filter_num,
                                            kernel_size=kernel,
                                            strides=stride,
                                            padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv1D(filters=filter_num,
                                            kernel_size=kernel,
                                            strides=1,
                                            padding="same")
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

class HeadCNN(tf.keras.layers.Layer):
    def __init__(self):
        super(HeadCNN, self).__init__()

        self.conv1 = tf.keras.layers.Conv1D(filters=512, kernel_size=3, strides=1)
        self.max1 = tf.keras.layers.MaxPool1D(pool_size=2, strides=1)
        self.conv2 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=1)
        self.max2 = tf.keras.layers.MaxPool1D(pool_size=2, strides=1)
        self.conv3 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=1)
        self.max3 = tf.keras.layers.MaxPool1D(pool_size=2, strides=1)
        self.conv4 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, strides=1)
        self.max4 = tf.keras.layers.MaxPool1D(pool_size=2, strides=1)

    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs, training=training)
        x = self.max1(x)
        x = self.conv2(inputs, training=training)
        x = self.max2(x)
        x = self.conv3(inputs, training=training)
        x = self.max3(x)
        x = self.conv4(inputs, training=training)
        x = self.max4(x)
        return x

def make_basic_block_layer(filter_num, blocks, name, kernel, stride=1, downsample=False):
    res_block = tf.keras.Sequential(name=name)
    res_block.add(BasicBlock(filter_num, kernel=kernel, stride=stride, downsample=downsample))

    for _ in range(1, blocks):
        res_block.add(BasicBlock(filter_num, kernel=kernel, stride=1))

    return res_block


def resnet18BlockNoClass(sensor_name):
    resnet = tf.keras.Sequential(name='resnet_branch_{}'.format(sensor_name))
    resnet.add(Resnet18BlockNoClass(sensor_name))
    return resnet

def resnet18MultiBranch(sensor_dict, num_user, magnitude):
    return Resnet18MultiBranch(sensor_dict, num_user, magnitude)

def new_head(sensor_name):
    head = tf.keras.Sequential(name='branch_{}'.format(sensor_name))
    head.add(HeadCNN())
    return head
