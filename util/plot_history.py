import matplotlib.pyplot as plt 

def plot_performance(ActivityAccuracy, UserAccuracy, fold, path_to_save, save=False):
  
  plt.plot(ActivityAccuracy)
  plt.plot(UserAccuracy)
  plt.title('Fold {} for test'.format(fold))
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['Activity_accuracy', 'User_accuracy'], loc='lower right')

  if save:
    plt.savefig(path_to_save + 'plot_{}.png'.format(fold))

  plt.show()
