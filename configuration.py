# global training parameters
EPOCHS = 50
BATCH_SIZE = 128

# parameters based on dataset
config = {
    'unimib': {
        'NUM_CLASSES_ACTIVITY'  : 9,
        'NUM_CLASSES_USER'      : 30,
        'WINDOW_AXES'           : 3,  # number of sample for window
        'WINDOW_SAMPLES'        : 100,    # number of axes' sensor
        'CHANNELS'              : 1,
        'SENSOR_DICT':          {
            'accelerometer' : 3
        }
    },
    'sbhar': {
        'NUM_CLASSES_ACTIVITY'  : 12,
        'NUM_CLASSES_USER'      : 30,
        'WINDOW_AXES'           : 6,  # number of sample for window
        'WINDOW_SAMPLES'        : 100,    # number of axes' sensor
        'CHANNELS'              : 1,       
        'SENSOR_DICT':          {
            'accelerometer' : 3,
            'gyroscope'     : 3
        }
    },
    'realdisp': {
        'NUM_CLASSES_ACTIVITY'  : 33,
        'NUM_CLASSES_USER'      : 17,
        'WINDOW_AXES'           : 9,  # number of sample for window
        'WINDOW_SAMPLES'        : 100,    # number of axes' sensor
        'CHANNELS'              : 1,       
        'SENSOR_DICT':          {
            'accelerometer' : 3,
            'gyroscope'     : 3,
            'magnetometer'  : 3
        }
    }
}

'''
save_model_dir = "saved_model/"
save_every_n_epoch = 10
test_image_dir = ""

dataset_dir = "dataset/"
train_dir = dataset_dir + "train"
valid_dir = dataset_dir + "valid"
test_dir = dataset_dir + "test"
train_tfrecord = dataset_dir + "train.tfrecord"
valid_tfrecord = dataset_dir + "valid.tfrecord"
test_tfrecord = dataset_dir + "test.tfrecord"
# VALID_SET_RATIO = 1 - TRAIN_SET_RATIO - TEST_SET_RATIO
TRAIN_SET_RATIO = 0.6
TEST_SET_RATIO = 0.2
'''