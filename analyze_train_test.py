import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random

import configuration
from model.custom_model import Model


dataset_name = 'unimib'
multitask = False
model_type = 'resnet18_2D'
fold = [0]
save_dir = 'log_prove'
outer_dir = 'Outerpartition_magnitude_prova_balance_'
overlap = 5.0
magnitude = True

model = Model(dataset_name=dataset_name,
              configuration_file=configuration,
              multi_task=multitask, lr='dynamic',
              model_type=model_type,
              fold=fold,
              save_dir=save_dir,
              outer_dir=outer_dir +
              str(overlap)+'/',
              overlap=overlap,
              magnitude=magnitude,
              log=True)
model.create_dataset()
model.load_data( only_acc=False, normalize=True, delete='delete')

#### plot some example of train and test of same user and activity
while True:
    i = input()
    if i == 'q':
        print('exit!')
        break
    if i == 'n':
        user = np.random.randint(0,model.num_user,1)
        act = np.random.randint(0,model.num_act,1)
        print(f'user: {user} act: {act}')

        train_act = np.where(model.train_act == act)[0]
        train_user = np.where(model.train_user == user)[0]
        test_act = np.where(model.test_act == act)[0]
        test_user = np.where(model.test_user == user)[0]

        idx_train = set(train_act).intersection(set(train_user))
        idx_test = set(test_act).intersection(set(test_user))

        train_sample = model.train[random.choice(tuple(idx_train)), :, :, 0]
        test_sample = model.test[random.choice(tuple(idx_test)), :, :, 0]

        step = np.arange(0,train_sample.shape[0])

        plt.figure()
        plt.suptitle(f'user {user} act {act}')
        plt.subplot(1,2,1)
        plt.title('train')
        plt.plot(step, train_sample[:,0], label='x')
        plt.plot(step, train_sample[:,1], label='y')
        plt.plot(step, train_sample[:,2], label='z')
        plt.subplot(1,2,2)
        plt.title('test')
        plt.plot(step, test_sample[:,0], label='x')
        plt.plot(step, test_sample[:,1], label='y')
        plt.plot(step, test_sample[:,2], label='z')     
        plt.show()  

