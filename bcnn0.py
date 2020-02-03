from __future__ import print_function

from keras.models import Model
from keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, Input
from keras.optimizers import Adam
from keras.applications import VGG19

import numpy as np


class BayesianCNN:
    def __init__(self, target_size=(224, 224), model=None):
        if model is not None:
            self.model = model
        else:
            self.model = self.get_model(input_shape=target_size + (3,))
        # model should be compiled
        self.name = self.model.name

    def get_model(self):
        raise NotImplementedError

    # wrapper function
    def fit_generator(self, generator, **kwargs):
        return self.model.fit_generator(generator, **kwargs)

    # Returns predictive mean and predictive uncertainty (variance)
    def predict_post_generator(self, generator, t, **kwargs):
        preds_t = None
        for _ in range(t):
            preds = self.model.predict_generator(generator, **kwargs)
            preds = preds[np.argsort(generator.index_array)]  # undo shuffle
            if preds_t is None:
                preds_t = np.expand_dims(preds, axis=2)
            else:
                preds = np.expand_dims(preds, axis=2)
                preds_t = np.concatenate((preds_t, preds), axis=2)
        return np.mean(preds_t, axis=2), np.var(preds_t, axis=2)

    # wrapper function
    def save(self, filename):
        self.model.save(filename)


class MCDropout(BayesianCNN):
    def __init__(self, **kwargs):
        super(MCDropout, self).__init__(**kwargs)

        self.opt = Adam(lr=0.0001, decay=0.0001)
        self.model.compile(optimizer=self.opt,
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

    def get_model(self, input_shape):
        base = VGG19(weights='imagenet',
                     include_top=False,
                     input_shape=input_shape)

        inputs = Input(shape=input_shape)
        x = inputs
        for i, layer in enumerate(base.layers):
            if i != 0:  # skip Input layer
                x = layer(x)
            if isinstance(layer, Conv2D):  # insert Dropout after every Conv2D
                x = Dropout(0.5)(x, training=True)  # Dropout during testing
        x = Flatten()(x)
        x = Dense(512, activation='relu', name='fc1')(x)
        x = Dropout(0.5)(x, training=True)
        x = Dense(256, activation='relu', name='fc2')(x)
        x = Dropout(0.5)(x, training=True)
        pred_layer = Dense(7, activation='sigmoid', name='pred_layer')(x)

        return Model(inputs=inputs, outputs=pred_layer, name='mcdo')

    def predict_post(self, x, t, **kwargs):
        preds_t = [self.model.predict(x, **kwargs) for _ in range(t)]
        return np.mean(preds_t, axis=0), np.var(preds_t, axis=0)


class MCBatchNorm(BayesianCNN):
    def __init__(self, **kwargs):
        super(MCBatchNorm, self).__init__(**kwargs)

        # compile model - double check uncertainty validity
        self.opt = Adam(lr=0.001, decay=0.0001)
        self.model.compile(optimizer=self.opt,
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

    def get_model(self, input_shape):
        base = VGG19(weights='imagenet',
                     include_top=False,
                     input_shape=input_shape)

        inputs = Input(shape=input_shape)
        x = inputs
        for i, layer in enumerate(base.layers):
            if i != 0:  # skip Input layer
                x = layer(x)
            if isinstance(layer, Conv2D):  # insert BatchNorm after every Conv2D
                x = BatchNormalization()(x, training=True)  # BatchNorm during testing
        x = Flatten()(x)
        x = Dense(512, activation='relu', name='fc1')(x)
        x = BatchNormalization()(x, training=True)
        x = Dense(256, activation='relu', name='fc2')(x)
        x = BatchNormalization()(x, training=True)
        pred_layer = Dense(7, activation='sigmoid', name='pred_layer')(x)

        return Model(inputs=inputs, outputs=pred_layer, name='mcbn')
