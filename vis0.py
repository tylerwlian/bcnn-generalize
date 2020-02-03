from keras import backend as K

import numpy as np
import pickle
import sklearn.calibration as sklc

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_cal(pp, tp, filename):
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(pp, tp)

    plt.xlabel('Predicted probabilities')
    plt.ylabel('True probabilities')
    plt.savefig(filename)
    plt.close()


# var is a dict of var arrays
def plot_var(var, filename):
    for vis_str in var.keys():
        plt.hist(var[vis_str], bins=20, alpha=0.5, label=vis_str)

    plt.xlabel('Predictive variance')
    plt.legend(loc='upper right')
    plt.savefig(filename)
    plt.close()


# entr is a dict of entropy arrays
def plot_entr(entr, filename):
    for vis_str in entr.keys():
        plt.hist(entr[vis_str], bins=20, alpha=0.5, label=vis_str)

    plt.xlabel('Predictive uncertainty')
    plt.legend(loc='upper right')
    plt.savefig(filename)
    plt.close()


def visualize_stats(eval_stats=None, filename_id=None, saved_eval=None):
    if eval_stats is None and saved_eval is None:
        raise ValueError('Not enough parameters!')

    if saved_eval is not None:
        with open(saved_eval, 'rb') as f:
            eval_stats = pickle.load(f)

    databases = eval_stats.keys()
    var, entr = {}, {}
    for vis_str in databases:
        y_true, y_pred, y_var, _ = eval_stats[vis_str]
        p_true, p_pred, p_var = y_true[:, 5], y_pred[:, 5], None  # PNEUMONIA_IDX is 5

        # compute uncertainty-related quantities, if relevant
        if y_var is not None:
            p_var = y_var[:, 5]
            var[vis_str] = p_var
            entr[vis_str] = -1 * (p_pred * np.log(p_pred) + (1 - p_pred) * np.log(1 - p_pred))

        # plot calibration curve
        pp, tp = sklc.calibration_curve(p_true, p_pred, n_bins=10)
        plot_cal(pp, tp, 'cal_' + filename_id + '.png')

    if not any([None is v for v in var.values()]):  # var is None for non-BCNNs
        plot_var(var, 'var_' + filename_id + '.png')  # predictive variance
        plot_entr(entr, 'entr_' + filename_id + '.png')  # entropy
