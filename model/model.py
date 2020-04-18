import numpy as np
import tensorflow as tf
#import tensorflow.contrib.slim as slim
from util.tensor_toolbox_yyang import TensorProducer

def weight_variable_with_scope(shape, scope):
    with tf.variable_scope( scope ):
        w = tf.get_variable("weights", shape, initializer=tf.random_normal_initializer(stddev=0.1))
    return w
def weight_variable(shape):
    w = tf.get_variable("weights", shape, initializer=tf.random_normal_initializer(stddev=0.1))
    return w

def bias_variable(shape):
    return tf.get_variable("bias", shape, initializer=tf.constant_initializer(0.0))

def activation(x):
    return tf.nn.relu(x)

def conv_unit( input, w, is_training ):
    assert len( w.shape ) == 4
    conv    = tf.nn.conv2d( input, w, strides=[1, 1, 1, 1], padding='VALID' )
    output  = tf.layers.batch_normalization( conv, training=is_training )
    return  activation( output )

def fc_unit( input, filter_shape, is_training, scope ):
    assert len( filter_shape ) == 2
    with tf.variable_scope( scope ):
        w       = weight_variable( filter_shape )
        mul     = tf.matmul( input, w )
        output  = tf.layers.batch_normalization( mul, training=is_training )
    return activation( output )

def pool_unit( input, ksize, stride, padding='VALID' ):
    return tf.nn.max_pool( input, ksize, stride, padding )

def fc_unit_without_activiation( input, filter_shape, is_training, scope ):
    assert len( filter_shape ) == 2
    with tf.variable_scope( scope ):
        w       = weight_variable( filter_shape )
        mul     = tf.matmul( input, w )
        output  = tf.layers.batch_normalization( mul, training=is_training )
    return output   

def lstm_unit( input, rnn_size, keep_prob, scope ):
    with tf.variable_scope( scope ):
        rnn_cell    = tf.contrib.rnn.BasicLSTMCell( rnn_size, forget_bias=1.0, state_is_tuple=True )
        rnn_cell    = tf.contrib.rnn.DropoutWrapper( cell=rnn_cell , input_keep_prob=1.0, output_keep_prob=keep_prob )
        output, _   = tf.nn.dynamic_rnn( rnn_cell, inputs=input, dtype=tf.float32, time_major=False )
    return output

def gru_unit( input, rnn_size, keep_prob, scope ):
    with tf.variable_scope( scope ):
        rnn_cell    = tf.contrib.rnn.GRUCell( rnn_size )
        rnn_cell    = tf.contrib.rnn.DropoutWrapper( cell=rnn_cell , input_keep_prob=1.0, output_keep_prob=keep_prob )
        output, _   = tf.nn.dynamic_rnn( rnn_cell, inputs=input, dtype=tf.float32, time_major=False )
    return output

class MTLMA_pretrain( object ):

    def __init__( self ):
        print( 'MTLMA_pretrain' )

    def __call__( self, inputs, a_labels, u_labels, act_num, user_num, win_len, dname, fold, is_training = True, drop_keep_prob = 0.9 ):

        with tf.variable_scope( 'act_network' ):
            # weights of CNN
            A_W_conv1   = weight_variable_with_scope( shape=[5,  1,  1,  32], scope='a_conv1' )
            A_W_conv2   = weight_variable_with_scope( shape=[5,  1,  32, 32], scope='a_conv2' )
            A_W_conv3   = weight_variable_with_scope( shape=[5,  1,  32, 32], scope='a_conv3' )
            
            # CNN
            A_net       = conv_unit( inputs,    A_W_conv1,      is_training )
            A_net       = conv_unit( A_net,     A_W_conv2,      is_training )
            A_net       = pool_unit( A_net,     [1, 2, 1, 1],   [1, 2, 1, 1] )
            A_net       = conv_unit( A_net,     A_W_conv3,      is_training )

            # bi-lstm
            A_net       = tf.reshape( A_net,    [-1, A_net.get_shape()[1].value, A_net.get_shape()[2].value*A_net.get_shape()[3].value] )
            A_net       = tf.transpose( A_net,  [1, 0, 2] )
            A_lstm_unit = tf.contrib.cudnn_rnn.CudnnLSTM( num_layers=1, num_units=128, input_mode='auto_select', direction='bidirectional', dropout=0.1 )
            A_net, _    = A_lstm_unit( inputs=A_net, scope='lstm' )
            A_net       = tf.transpose( A_net,  [1, 0, 2] )

        with tf.variable_scope( 'user_network' ):
            # weights of CNN
            U_W_conv1   = weight_variable_with_scope( shape=[5,  1,  1,  32], scope='u_conv1' )
            U_W_conv2   = weight_variable_with_scope( shape=[5,  1,  32, 32], scope='u_conv2' )
            U_W_conv3   = weight_variable_with_scope( shape=[5,  1,  32, 32], scope='u_conv3' )
            
            # CNN
            U_net       = conv_unit( inputs,    U_W_conv1,  is_training )
            U_net       = conv_unit( U_net,     U_W_conv2,  is_training )
            U_net       = pool_unit( U_net,     [1, 2, 1, 1], [1, 2, 1, 1] )
            U_net       = conv_unit( U_net,     U_W_conv3,  is_training )

            #bi-lstm
            U_net       = tf.reshape( U_net,    [-1, U_net.get_shape()[1].value, U_net.get_shape()[2].value*U_net.get_shape()[3].value] )
            U_net       = tf.transpose( U_net,  [1, 0, 2] )
            u_lstm_unit = tf.contrib.cudnn_rnn.CudnnLSTM( num_layers=1, num_units=128, input_mode='auto_select', direction='bidirectional', dropout=0.1 )
            U_net, _    = u_lstm_unit( inputs=U_net, scope='lstm' )
            U_net       = tf.transpose( U_net,  [1, 0, 2] )

        with tf.variable_scope( 'act_network' ):
            # attention for AR net
            A_ATT       = tf.reshape( U_net, [-1, U_net.get_shape()[1].value*U_net.get_shape()[2].value] )
            A_ATT       = fc_unit( A_ATT,   [A_ATT.get_shape()[1].value,   128],   is_training, scope='att1' )
            A_ATT       = fc_unit( A_ATT,   [128,   A_net.get_shape()[1].value],   is_training, scope='att2' ) # time axis
            A_ATT       = tf.expand_dims( tf.nn.softmax( A_ATT ), 2 )

        with tf.variable_scope( 'user_network' ):
            # attention for UR net
            U_ATT       = tf.reshape( A_net, [-1, A_net.get_shape()[1].value*A_net.get_shape()[2].value] )
            U_ATT       = fc_unit( U_ATT,   [U_ATT.get_shape()[1].value,   128],    is_training, scope='att1' )
            U_ATT       = fc_unit( U_ATT,   [128,   U_net.get_shape()[1].value],    is_training, scope='att2' )
            U_ATT       = tf.expand_dims( tf.nn.softmax( U_ATT ), 2 )

        with tf.variable_scope( 'act_network' ):
            # output layer
            A_net       = tf.reduce_sum( tf.multiply( A_net, A_ATT ), 1 )
            A_net       = fc_unit_without_activiation( A_net, [A_net.get_shape()[1].value, act_num], is_training, scope='otpt' )       
            
        with tf.variable_scope( 'user_network' ):
            # output layer
            U_net       = tf.reduce_sum( tf.multiply( U_net, U_ATT ), 1 )
            U_net       = fc_unit_without_activiation( U_net, [U_net.get_shape()[1].value, user_num], is_training, scope='otpt' )

        # losses
        A_cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits( labels = a_labels , logits = A_net ) )
        A_penality 	    = sum( tf.nn.l2_loss(tf_var) for tf_var in self.get_act_step_vars() )
        A_loss          = A_cross_entropy + 0.0003*A_penality

        U_cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits( labels = u_labels , logits = U_net ) )
        U_penality 	    = sum( tf.nn.l2_loss(tf_var) for tf_var in self.get_user_step_vars() )
        U_loss          = U_cross_entropy + 0.0003*U_penality

        return A_net, A_loss, U_net, U_loss

    def get_act_step_vars( self ):
        return  tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES, scope='act_network' )

    def get_user_step_vars( self ):
        return  tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES, scope='user_network' )


class MTLMA_train( object ):

    def __init__( self ):
        print( 'MTLMA_train' )

    def __call__( self, inputs, a_labels, u_labels, act_num, user_num, win_len, dname, fold, is_training = True, drop_keep_prob = 0.9 ):

        # weights of CNN
        w_conv1, _  = TensorProducer( np.stack( [ np.load("./data/parameters/{}f{}a1.npy".format( dname, fold)), np.load("./data/parameters/{}f{}u1.npy".format( dname, fold)) ], axis=4 ), 'Tucker', eps_or_k=0.1, return_true_var=True )
        w_conv2, _  = TensorProducer( np.stack( [ np.load("./data/parameters/{}f{}a2.npy".format( dname, fold)), np.load("./data/parameters/{}f{}u2.npy".format( dname, fold)) ], axis=4 ), 'Tucker', eps_or_k=0.1, return_true_var=True )
        w_conv3, _  = TensorProducer( np.stack( [ np.load("./data/parameters/{}f{}a3.npy".format( dname, fold)), np.load("./data/parameters/{}f{}u3.npy".format( dname, fold)) ], axis=4 ), 'Tucker', eps_or_k=0.1, return_true_var=True )

        # CNN
        A_net       = conv_unit( inputs,    w_conv1[:,:,:,:,0], is_training )
        A_net       = conv_unit( A_net,     w_conv2[:,:,:,:,0], is_training )
        A_net       = pool_unit( A_net,     [1, 2, 1, 1],       [1, 2, 1, 1] )
        A_net       = conv_unit( A_net,     w_conv3[:,:,:,:,0], is_training )
        # bi-lstm
        A_net       = tf.reshape( A_net,    [-1, A_net.get_shape()[1].value, A_net.get_shape()[2].value*A_net.get_shape()[3].value] )
        A_net       = tf.transpose( A_net,  [1, 0, 2] )
        A_lstm_unit = tf.contrib.cudnn_rnn.CudnnLSTM( num_layers=1, num_units=128, input_mode='auto_select', direction='bidirectional', dropout=0.1 )
        A_net, _    = A_lstm_unit( inputs=A_net, scope='A_lstm' )
        A_net       = tf.transpose( A_net,  [1, 0, 2] )     

        # CNN
        U_net       = conv_unit( inputs,    w_conv1[:,:,:,:,1], is_training )
        U_net       = conv_unit( U_net,     w_conv2[:,:,:,:,1], is_training )
        U_net       = pool_unit( U_net,     [1, 2, 1, 1],       [1, 2, 1, 1] )
        U_net       = conv_unit( U_net,     w_conv3[:,:,:,:,1], is_training )
        # bi-lstm
        U_net       = tf.reshape( U_net,    [-1, U_net.get_shape()[1].value, U_net.get_shape()[2].value*U_net.get_shape()[3].value] )
        U_net       = tf.transpose( U_net,  [1, 0, 2] )
        u_lstm_unit = tf.contrib.cudnn_rnn.CudnnLSTM( num_layers=1, num_units=128, input_mode='auto_select', direction='bidirectional', dropout=0.1 )
        U_net, _    = u_lstm_unit( inputs=U_net, scope='U_lstm' )
        U_net       = tf.transpose( U_net,  [1, 0, 2] )        

        # attention for ARnet
        A_ATT       = tf.reshape( U_net, [-1, U_net.get_shape()[1].value*U_net.get_shape()[2].value] )
        A_ATT       = fc_unit( A_ATT,   [A_ATT.get_shape()[1].value,   128],   is_training, scope='A_att1' )
        A_ATT       = fc_unit( A_ATT,   [128,   A_net.get_shape()[1].value],   is_training, scope='A_att2' ) # time axis
        A_ATT       = tf.expand_dims( tf.nn.softmax( A_ATT ), 2 )

        # attention for URnet
        U_ATT       = tf.reshape( A_net, [-1, A_net.get_shape()[1].value*A_net.get_shape()[2].value] )
        U_ATT       = fc_unit( U_ATT,   [U_ATT.get_shape()[1].value,   128],    is_training, scope='U_att1' )
        U_ATT       = fc_unit( U_ATT,   [128,   U_net.get_shape()[1].value],    is_training, scope='U_att2' )
        U_ATT       = tf.expand_dims( tf.nn.softmax( U_ATT ), 2 )

        # output layer of ARnet
        A_net       = tf.reduce_sum( tf.multiply( A_net, A_ATT ), 1 )
        A_net       = fc_unit_without_activiation( A_net, [A_net.get_shape()[1].value, act_num], is_training, scope='A_otpt' )       
        
        # output layer of URnet
        U_net       = tf.reduce_sum( tf.multiply( U_net, U_ATT ), 1 )
        U_net       = fc_unit_without_activiation( U_net, [U_net.get_shape()[1].value, user_num], is_training, scope='U_otpt' )


        A_cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits( labels = a_labels , logits = A_net ) )
        U_cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits( labels = u_labels , logits = U_net ) )
        penality 	    = sum( tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables() )
        loss            = A_cross_entropy + U_cross_entropy + 0.0003*penality

        return A_net, loss, U_net, loss

    def get_act_step_vars( self ):
        return  tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES )

    def get_user_step_vars( self ):
        return  tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES )