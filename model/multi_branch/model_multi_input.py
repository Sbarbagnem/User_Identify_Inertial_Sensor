import tensorflow as tf

class ModelMultiBranch(tf.keras.Model):
    def __init__(self, sensor_dict, multi_task, num_act, num_user):
        super(ModelMultiBranch, self).__init__()

        self.senor_name = list(sensor_dict.keys())
        self.sensor_axes = list(sensor_dict.values())
        self.multi_task = multi_task
        if multi_task:
            self.num_act = num_act
        self.num_user = num_user

        self.branches = []

        for sensor_name, sensor_axes in zip(self.senor_name, self.sensor_axes):
            self.branches.append(make_basic_branch(sensor_name, sensor_axes))    

        self.conv1_merge = tf.keras.layers.Conv2D(filters=64,
                                        kernel_size=(len(sensor_name), 8),
                                        strides=(1),
                                        padding="same",
                                        name='conv1_merge_branch')
        self.bn1_merge = tf.keras.layers.BatchNormalization(name='bn1_merge')

        self.conv2_merge = tf.keras.layers.Conv2D(filters=64,
                                        kernel_size=(1, 6),
                                        strides=(1),
                                        padding="same",
                                        name='conv2_merge_branch')
        self.bn2_merge = tf.keras.layers.BatchNormalization(name='bn2_merge')

        self.conv3_merge = tf.keras.layers.Conv2D(filters=64,
                                        kernel_size=(1, 4),
                                        strides=(1),
                                        padding="same",
                                        name='conv3_merge_branch')
        self.bn3_merge = tf.keras.layers.BatchNormalization(name='bn3_merge')

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

        sensor_split = tf.split(inputs, num_or_size_splits=self.sensor_axes, axis=1)

        merge_branch = []

        for input_sensor, branch in zip(sensor_split, self.branches):
            x = branch(input_sensor, training=training)
            x = tf.reshape(x, shape=[-1, 1, x.shape[1]*x.shape[2]*x.shape[3], 1]) # flatten
            merge_branch.append(x)

        # merge output
        merge = tf.concat(merge_branch, axis=1) # [batch_size, k_sensor, flatten_conv]

        merge = self.conv1_merge(merge, training=training)
        merge = self.bn1_merge(merge, training=training)
        merge = tf.nn.relu(merge)
        merge = self.conv2_merge(merge, training=training)
        merge = self.bn2_merge(merge, training=training)
        merge = tf.nn.relu(merge)
        merge = self.conv3_merge(merge, training=training)
        merge = self.bn3_merge(merge, training=training)
        merge = tf.nn.relu(merge)

        merge = self.avg_pool(merge)


        if self.multi_task:
            output_activity = self.fc_activity(merge)
            output_user = self.fc_user(merge)
            return output_activity, output_user
        else:
            output_user = self.fc_user(merge)
            return output_user



class BasicBranch(tf.keras.layers.Layer):
    def __init__(self, name_sensor, sensor_axes):
        super(BasicBranch, self).__init__()
        '''
         input [batch_size, axes, samples, channel]
        '''

        self.name_sensor = name_sensor
        self.sensor_axes = sensor_axes


        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(sensor_axes, 6),
                                            strides=(1, 3),
                                            padding="valid",
                                            name='conv1_{}'.format(name_sensor))
        self.bn1 = tf.keras.layers.BatchNormalization(name='bn1_{}'.format(name_sensor))
        self.conv2 = tf.keras.layers.Conv2D(filters=64, 
                                            kernel_size=(1, 4),
                                            strides=(1, 2), 
                                            padding='valid',
                                            name='conv2_{}'.format(name_sensor))
        self.bn2 = tf.keras.layers.BatchNormalization(name='bn2_{}'.format(name_sensor))
        self.conv3 = tf.keras.layers.Conv2D(filters=64, 
                                            kernel_size=(1, 2),
                                            strides=(1), 
                                            padding='valid',
                                            name='conv3_{}'.format(name_sensor))
        self.bn3 = tf.keras.layers.BatchNormalization(name='bn3_{}'.format(name_sensor))

    def call(self, input, training=None, ):

        # residual

        x = self.conv1(input)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = tf.nn.relu(x)
        
        return x


def make_basic_branch(name_sensor, sensor_axes):
    basic_branch = tf.keras.Sequential(name=name_sensor)
    basic_branch.add(BasicBranch(name_sensor, sensor_axes))
    return basic_branch

def model_multi_branch(sensor_dict, multi_task, num_act, num_user):
    return ModelMultiBranch(sensor_dict=sensor_dict, multi_task=multi_task, num_act=num_act, num_user=num_user)
