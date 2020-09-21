# global training parameters
EPOCHS = 50
EPOCHS_GAN = 20
BATCH_SIZE = 128
LATENT_DIM = 128

# parameters based on dataset
config = {
    'unimib': {
        'PATH_OUTER_PARTITION': './data/datasets/UNIMIB_processed/',
        'NUM_CLASSES_ACTIVITY': 9,
        'NUM_CLASSES_USER': 30,
        'WINDOW_AXES': 3,  # number of sample for window
        'WINDOW_SAMPLES': 100,    # number of axes' sensor
        'AXIS': 3,
        'AXIS_MAGNITUDE': 1,
        'SENSOR_DICT':          {
            'accelerometer': 3
        }
    },
    'unimib_75w': {
        'PATH_OUTER_PARTITION': './data/datasets/UNIMIB_processed/',
        'NUM_CLASSES_ACTIVITY': 9,
        'NUM_CLASSES_USER': 30,
        'WINDOW_AXES': 3,  # number of sample for window
        'WINDOW_SAMPLES': 75,    # number of axes' sensor
        'AXIS': 3,
        'AXIS_MAGNITUDE': 1,
        'SENSOR_DICT':          {
            'accelerometer': 3
        }
    },
    'unimib_128w': {
        'PATH_OUTER_PARTITION': './data/datasets/UNIMIB_processed/',
        'NUM_CLASSES_ACTIVITY': 9,
        'NUM_CLASSES_USER': 30,
        'WINDOW_AXES': 3,  # number of sample for window
        'WINDOW_SAMPLES': 128,    # number of axes' sensor
        'AXIS': 3,
        'AXIS_MAGNITUDE': 1,
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
        'AXIS': 6,
        'AXIS_MAGNITUDE': 2,
        'SENSOR_DICT':          {
            'accelerometer': 3,
            'gyroscope': 3
        } 
    },
    'sbhar_128w': {
        'PATH_OUTER_PARTITION': './data/datasets/SBHAR_processed/',
        'NUM_CLASSES_ACTIVITY': 12,
        'NUM_CLASSES_USER': 30,
        'WINDOW_AXES': 6,  # number of sample for window
        'WINDOW_SAMPLES': 128,    # number of axes' sensor
        'AXIS': 6,
        'AXIS_MAGNITUDE': 2,
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
        'AXIS': 9,
        'AXIS_MAGNITUDE': 3,
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
        'AXIS': 3,
        'AXIS_MAGNITUDE': 1,
        'SENSOR_DICT':          {
            'accelerometer': 3
        }
    },
    'sbhar_six_adl': {
        'PATH_OUTER_PARTITION': './data/datasets/SBHAR_processed/',
        'NUM_CLASSES_ACTIVITY': 6,
        'NUM_CLASSES_USER': 30,
        'WINDOW_AXES': 6,  # number of sample for window
        'WINDOW_SAMPLES': 100,    # number of axes' sensor
        'AXIS': 6,
        'AXIS_MAGNITUDE': 2,
        'SENSOR_DICT':          {
            'accelerometer': 3,
            'gyroscope': 3
        }
    }
}

sbhar_mapping = {
    'sbhar_all_in_one': {
        'mapping':{
            6: [6,7,8,9,10,11]
        },
        'NUM_CLASSES_ACTIVITY': 7
    },
    'sbhar_complementary': {
        'mapping':{
            6: [6,7], # stand-to-sit, sit-to-stand
            7: [8,9], # sit-to-lie, lie-to-sit
            8: [10,11] # stand-to-lie, lie-to-stand
        },
        'NUM_CLASSES_ACTIVITY': 9
    },
    'sbhar_couple': { # 90% accuracy
        'mapping':{
            6: [6,10], # stand-to-sit, stand-to-lie
            7: [7,8], # sit-to-stand, sit-to-lie
            8: [9,11] # lie-to-sit, lie-to-stand
        },
        'NUM_CLASSES_ACTIVITY': 9
    },
    'sbhar_up_down': {
        'mapping': {
            6: [6,10,8], # stand-to-sit, stand-to-lie, sit-to-lie
            7: [7,11,9] # sit-to-stand, lie-to-stand, lie-to-sit
        },
        'NUM_CLASSES_ACTIVITY': 8
    }
}
