import tensorflow as tf

'''
    from https://github.com/calmisential/Basic_CNNs_TensorFlow2
'''


class ResNet18SingleBranch(tf.keras.Model):
    def __init__(self, layer_params, multi_task, num_act, num_user):
        super(ResNet18SingleBranch, self).__init__()

        self.multi_task = multi_task
        if multi_task:
            self.num_act = num_act
        self.num_user = num_user

        # features about single axis sensor (parameters from metier)

        self.conv1 = tf.keras.layers.Conv2D(#filters=64,
                                            filters=32,
                                            #kernel_size=(7, 7),
                                            kernel_size=(1,5),
                                            #strides=2,
                                            strides=(1,1),
                                            padding="valid")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(#pool_size=(3, 3),
                                               pool_size=(1,2),
                                               strides=(1,2),
                                               padding="valid")

        # features about interaction between sensor

        self.layer1 = make_basic_block_layer(#filter_num=64,
                                             filter_num=32,
                                             blocks=layer_params[0],
                                             name='residual_block_1')
        self.layer2 = make_basic_block_layer(#filter_num=128,
                                             filter_num=64,
                                             blocks=layer_params[1],
                                             name='residual_block_2',
                                             stride=2)
        '''
        self.layer3 = make_basic_block_layer(#filter_num=256,
                                             filter_num=64,
                                             blocks=layer_params[2],
                                             name='residual_block_3',
                                             stride=2)
        '''
        '''
        self.layer4 = make_basic_block_layer(filter_num=512,
                                             blocks=layer_params[3],
                                             name='residual_block_4',
                                             stride=2)
        '''

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()

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

        x = self.avgpool(x)
        #print('shape avg_pool: {}'.format(x.shape))

        if self.multi_task:
            output_activity = self.fc_activity(x)
            output_user = self.fc_user(x)
            return output_activity, output_user
        else:
            output_user = self.fc_user(x)
            return output_user


class BasicBlock(tf.keras.layers.Layer):

    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=stride,
                                            padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=1,
                                            padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()
        # doownsample per ristabilire dimensioni residuo tra un blocco e l'altro
        if stride != 1:
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


def make_basic_block_layer(filter_num, blocks, name, stride=1):
    res_block = tf.keras.Sequential(name=name)
    res_block.add(BasicBlock(filter_num, stride=stride))

    for _ in range(1, blocks):
        res_block.add(BasicBlock(filter_num, stride=1))

    return res_block


def resnet18(multi_task, num_act, num_user):
    return ResNet18SingleBranch(layer_params=[2, 2, 2, 2], multi_task=multi_task, num_act=num_act, num_user=num_user)
