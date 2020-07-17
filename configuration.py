# global training parameters
EPOCHS = 50
BATCH_SIZE = 128

# parameters based on dataset
config = {
    'unimib': {
        'PATH_OUTER_PARTITION': './data/datasets/UNIMIBDataset/',
        'NUM_CLASSES_ACTIVITY': 9,
        'NUM_CLASSES_USER': 30,
        'WINDOW_AXES': 3,  # number of sample for window
        'WINDOW_SAMPLES': 100,    # number of axes' sensor
        'CHANNELS': 1,
        'SENSOR_DICT':          {
            'accelerometer': 3
        }
    },
    'sbhar': {
        'PATH_OUTER_PARTITION': './data/datasets/SBHAR_processed/',
        'NUM_CLASSES_ACTIVITY': 12,
        'NUM_CLASSES_USER': 30,
        'WINDOW_AXES': 6,  # number of sample for window
        'WINDOW_SAMPLES': 100,    # number of axes' sensor
        'CHANNELS': 1,
        'SENSOR_DICT':          {
            'accelerometer': 3,
            'gyroscope': 3
        }
    },
    'realdisp': {
        'PATH_OUTER_PARTITION': './data/datasets/REALDISP_processed/',
        'NUM_CLASSES_ACTIVITY': 33,
        'NUM_CLASSES_USER': 17,
        'WINDOW_AXES': 9,  # number of sample for window
        'WINDOW_SAMPLES': 100,    # number of axes' sensor
        'CHANNELS': 1,
        'SENSOR_DICT':          {
            'accelerometer': 3,
            'gyroscope': 3,
            'magnetometer': 3
        }
    },
    'unimib_sbhar': {
        'PATH_OUTER_PARTITION': './data/datasets/merged_unimib_sbhar/',
        'NUM_CLASSES_ACTIVITY': 21,
        'NUM_CLASSES_USER': 60,
        'WINDOW_AXES': 3,  # number of sample for window
        'WINDOW_SAMPLES': 100,    # number of axes' sensor
        'CHANNELS': 1,
        'SENSOR_DICT':          {
            'accelerometer': 3
        }
    }
}
