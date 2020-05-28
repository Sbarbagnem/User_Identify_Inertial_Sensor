import tensorflow as tf

'''
    from https://github.com/calmisential/Basic_CNNs_TensorFlow2
'''
class Resnet18MultiBranch(tf.keras.Model):
    def __init__(self, sensor_dict, multi_task, num_act, num_user):
        super(Resnet18MultiBranch, self).__init__()

        self.sensor_name = list(sensor_dict.keys())
        self.sensor_axes = list(sensor_dict.values())
        self.multi_task = multi_task
        if multi_task:
            self.num_act = num_act
        self.num_user = num_user

        self.branches = []

        for sensor_name in self.sensor_name:
            self.branches.append(resnet18BlockNoClass(sensor_name))    

        self.conv1_merge = tf.keras.layers.Conv2D(filters=64,
                                        kernel_size=(3,3),
                                        strides=2,
                                        padding="same",
                                        name='conv1_merge_branch')
        self.bn1_merge = tf.keras.layers.BatchNormalization(name='bn1_merge')

        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()

        if multi_task:
            self.fc_activity = tf.keras.layers.Dense(units=num_act,
                                                     activation=tf.keras.activations.softmax,
                                                     name='fc_act')
        self.fc_user = tf.keras.layers.Dense(units=num_user,
                                             activation=tf.keras.activations.softmax,
                                             name='fc_user')


    def call(self, inputs, training=None):
        '''
            input [batch_size, axes, samples, channel]
        '''
        merge_branch = []

        if len(self.sensor_name)>1:
            sensor_split = tf.split(inputs, num_or_size_splits=self.sensor_axes, axis=1)
            for input_sensor, branch in zip(sensor_split, self.branches):
                #print(input_sensor.shape)
                x = branch(input_sensor, training=training)
                #x = tf.reshape(x, shape=[-1, 1, x.shape[1]*x.shape[2]*x.shape[3], 1]) # flatten
                merge_branch.append(x)
        else:
            x = self.branches[0](inputs, training=training)
            #x = tf.reshape(x, shape=[-1, 1, x.shape[1]*x.shape[2]*x.shape[3], 1]) # flatten
            merge_branch.append(x)


        # merge output
        if len(merge_branch) > 1:
            merge = tf.concat(merge_branch, axis=1) # [batch_size, k_sensor, flatten_conv]
        else:
            merge = merge_branch[0]

        print(merge.shape)

        merge = self.conv1_merge(merge, training=training)
        merge = self.bn1_merge(merge, training=training)
        merge = tf.nn.relu(merge)

        merge = self.avg_pool(merge)

        #print(merge.shape)

        if self.multi_task:
            output_activity = self.fc_activity(merge)
            output_user = self.fc_user(merge)
            return output_activity, output_user
        else:
            output_user = self.fc_user(merge)
            return output_user

class Resnet18BlockNoClass(tf.keras.layers.Layer):
    def __init__(self, sensor_name):
        super(Resnet18BlockNoClass, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters=32,
                                            kernel_size=(1,5),
                                            strides=(1,1),
                                            padding="valid")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(1,2),
                                               strides=(1,2),
                                               padding="valid")

        # features about interaction between sensor's axes

        self.layer1 = make_basic_block_layer(filter_num=32,
                                             blocks=2,
                                             name='residual_block_1_{}'.format(sensor_name))
        self.layer2 = make_basic_block_layer(filter_num=64,
                                             blocks=2,
                                             name='residual_block_2{}'.format(sensor_name),
                                             stride=2)

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

    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3,3),
                                            strides=stride,
                                            padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3,3),
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


def resnet18BlockNoClass(sensor_name):
    resnet = tf.keras.Sequential(name='resnet_branch_{}'.format(sensor_name))
    resnet.add(Resnet18BlockNoClass(sensor_name))
    return resnet

def resnet18MultiBranch(sensor_dict, multi_task, num_act, num_user):
    return Resnet18MultiBranch(sensor_dict, multi_task, num_act, num_user)
