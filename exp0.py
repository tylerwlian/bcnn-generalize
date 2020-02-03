"""

"""
import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

import numpy as np
import pandas as pd
import sklearn.metrics as sklm

import time
import pickle
import argparse

import utils0 as utils
import vis0 as vis


# Define experimental parameters from command line
def get_flags():
    parser = argparse.ArgumentParser(description='Define experimental parameters')
    parser.add_argument('--model_type', type=str, help='Model class (e.g., dn, mcdo, mcbn)')
    parser.add_argument('--data_str', type=str, help='Train data source (nih, stf, mit)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--saved_model', type=str, help='Specify path to saved model')
    parser.add_argument('--train_sample', type=float, help='Sample from train data')
    parser.add_argument('--eval_sample', type=float, help='Sample from test data')
    return parser.parse_args()


# Train the model
def train(model_type, data_str, epochs, saved_model=None, sample=None, target_size=(224, 224)):
    # create model
    model = utils.get_model_from_string(model_type=model_type,
                                        target_size=target_size,
                                        saved_model=saved_model)

    if saved_model is not None:
        return model, None  # loaded from checkpoint, no training needed

    # create data generators
    data = utils.get_data_from_string(data_str=data_str, target_size=target_size)
    train_gen, valid_gen = data.get_train_val_gen(sample=sample)

    # create Callbacks
    lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.1)
    early_stop = EarlyStopping(monitor='val_loss', patience=3)
    model_checkpoint = ModelCheckpoint('check_%s_%s.hdf5' % (model_type, data_str),
                                       monitor='val_loss', save_best_only=True)

    # train model
    start = time.time()
    print('Training on', data_str)
    train_history = model.fit_generator(train_gen,
                                        epochs=epochs,
                                        callbacks=[lr_schedule, early_stop, model_checkpoint],
                                        workers=0,
                                        validation_data=valid_gen)
    end = time.time()
    print('Finished training in', end - start, 's\n')

    return model, train_history


def evaluate(model, sample=None, target_size=(224, 224)):
    eval_stats = {}
    for test_str in ['nih', 'stf']:
        print('Evaluating on', test_str)

        database = utils.get_data_from_string(test_str, target_size)
        test_gen = database.get_test_gen(sample=sample)

        y_true = test_gen._data
        if model.name in ['dn', 'densenet121', 'model_1']:  # temp
            y_pred = model.predict_generator(test_gen, workers=0)
            y_pred = y_pred[np.argsort(test_gen.index_array)]  # undo shuffle
            y_var = None
        else:
            y_pred, y_var = model.predict_post_generator(test_gen, t=50, workers=0)
        eval_stats.update({test_str: (y_true, y_pred, y_var, test_gen.filenames)})

        compute_stats(y_true, y_pred, class_idx=database.PNEUMONIA_IDX)

    return eval_stats


def compute_stats(y_true, y_pred, class_idx):
    # retrieve predictions for Pneumonia label only
    true = y_true[:, class_idx]
    pred = y_pred[:, class_idx]

    fpr, tpr, thr = sklm.roc_curve(true, pred)
    t_idx = np.where(tpr > 0.95)[0][0]  # find the first threshold s.t. tpr > 0.95

    print('t:', thr[t_idx])
    print('Accuracy:', np.mean((pred > thr[t_idx]) == true))
    print('Sensitivity:', tpr[t_idx])
    print('Specificity:', 1 - fpr[t_idx])
    print('ROC AUC:', sklm.roc_auc_score(true, pred))
    print('Precision (PPV):', np.sum((pred > thr[t_idx])[np.where(true == 1)] == 1) / np.sum(pred > thr[t_idx]))
    print('NPV:', np.sum((pred > thr[t_idx])[np.where(true == 0)] == 0) / np.sum(pred < thr[t_idx]))

    # pcn, rcl, _ = sklm.precision_recall_curve(true, pred)
    # print('PRC AUC:', sklm.auc(rcl, pcn))


def main():
    FLAGS = get_flags()
    print('User input parameters:', FLAGS)

    model, history = train(model_type=FLAGS.model_type,
                           data_str=FLAGS.data_str,
                           epochs=FLAGS.epochs,
                           saved_model=FLAGS.saved_model,
                           sample=FLAGS.train_sample)

    if history is not None:  # if model loaded from checkpoint
        with open('train_%s_%s.pickle' % (FLAGS.model_type, FLAGS.data_str), 'wb') as f:
            pickle.dump(history, f)

    eval_stats = evaluate(model, sample=FLAGS.eval_sample)

    with open('eval_%s_%s.pickle' % (model.name, FLAGS.data_str), 'wb') as f:
        pickle.dump(eval_stats, f)

    vis.visualize_stats(eval_stats, filename_id='%s_%s' % (model.name, FLAGS.data_str))

    print('Done.')


if __name__ == '__main__':
    main()
