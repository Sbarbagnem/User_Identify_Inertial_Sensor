import tensorflow as tf
import sys

class ModelPaper(tf.keras.Model):
    def __init__(self, num_user):
        super(ModelPaper, self).__init__()
        
        self.num_user = num_user
        self.conv1 = tf.keras.layers.Conv2D(filters=32,
                                            kernel_size=(6,1),
                                            strides=(1,1),
                                            padding="valid",
                                            kernel_regularizer=tf.keras.regularizers.l2)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pad1 = tf.keras.layers.ZeroPadding2D(((2,2), (1,1)))
        self.conv2 = tf.keras.layers.Conv2D(filters=48,
                                            kernel_size=(6,3),
                                            strides=(1,1),
                                            padding="valid",
                                            kernel_regularizer=tf.keras.regularizers.l2)  
        self.bn2 = tf.keras.layers.BatchNormalization()  
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3,2),
                                               strides=(3,2),
                                               padding="valid") 
        self.conv3 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(6,2),
                                            strides=(1,1),
                                            padding="valid",
                                            kernel_regularizer=tf.keras.regularizers.l2)  
        self.bn3 = tf.keras.layers.BatchNormalization()  
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(3,1),
                                               strides=(3,1),
                                               padding="valid") 
        self.conv4 = tf.keras.layers.Conv2D(filters=96,
                                            kernel_size=(3,1),
                                            strides=(1,1),
                                            padding="valid",
                                            kernel_regularizer=tf.keras.regularizers.l2)  
        self.bn4 = tf.keras.layers.BatchNormalization()
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(int(1.5*self.num_user), activation=tf.keras.activations.tanh)
        self.fc_user = tf.keras.layers.Dense(units=num_user, activation=tf.keras.activations.softmax)

    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        #print(f'conv1 {x.shape}')
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pad1(x)
        #print(f'padding1 {x.shape}')
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        #print(f'pool1 {x.shape}')
        x = self.conv3(x)
        #print(f'conv3 {x.shape}')
        x = self.bn3(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool2(x)
        #print(f'pool2 {x.shape}')
        x = self.conv4(x)
        #print(f'conv4 {x.shape}')
        x = self.bn4(x, training=training)
        x = tf.nn.relu(x)
        x = self.flatten(x)
        #print(f'flatten {x.shape}')
        x = self.fc1(x)
        out = self.fc_user(x)

        return out