'''
Created Date: Wednesday, July 27th 2022, 5:15:31 pm
Author: Rutuja Gurav (rutuja.gurav@email.ucr.edu)
Copyright (c) 2022 M.A.D. Lab @ UCR (https://madlab.cs.ucr.edu)

'''

import matplotlib.pyplot as plt
def plot_training_metrics(history=None, title=None, SAVE_PATH=None):
    
    metrics = [metric for metric in history.keys() if 'val' not in metric]
    for metric in metrics:
        plt.plot(history[metric])
        if metric != 'lr':
            plt.plot(history['val_'+metric])
        if 'loss' in metric:
            yl = -0.1*max(max(history[metric]), max(history['val_'+metric]))
            yh = 1.5*max(max(history[metric]), max(history['val_'+metric]))
            print(yl,yh)
            plt.ylim(yl, yh)
        plt.ylabel(metric)
        plt.xlabel('epoch')
        plt.title(title)
        plt.legend(['train', 'validation'], loc='upper right')
        plt.savefig(SAVE_PATH+'{}.png'.format(metric))
        plt.close()