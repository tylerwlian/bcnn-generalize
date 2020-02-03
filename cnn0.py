"""

"""

from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications import DenseNet121, VGG19
from keras.optimizers import Adam


# Return compiled DenseNet-121 multiclass classier
def DenseNetMultiClass(weights='imagenet', target_size=(224, 224), model=None):
    if model is not None:
        return model  # return model loaded from checkpoint

    base = DenseNet121(weights=weights,
                       include_top=False,
                       pooling='avg',
                       input_shape=target_size + (3,))
    pred_layer = Dense(7, activation='sigmoid', name='pred_layer')(base.output)

    model = Model(inputs=base.input, outputs=pred_layer, name='dn')

    # opt = SGD(lr=0.01, momentum=0.9, decay=1e-4)
    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['acc'])

    return model


# Return compiled DenseNet-121 binary classifier
def DenseNetBinClass(weights='imagenet', target_size=(224, 224)):
    base = DenseNet121(weights=weights,
                       include_top=False,
                       input_shape=target_size + (3,))
    x = GlobalAveragePooling2D()(base.output)
    pred_layer = Dense(1, activation='sigmoid', name='pred_layer')(x)

    model = Model(inputs=base.input, outputs=pred_layer)

    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['acc'])

    return model


# Return compiled VGG-19 model
def VGG19BinClass(weights='imagenet', target_size=(224, 224)):
    base = VGG19(weights=weights,
                 include_top=False,
                 input_shape=target_size + (3,))
    x = GlobalAveragePooling2D()(base.output)
    pred_layer = Dense(1, activation='sigmoid', name='pred_layer')(x)

    model = Model(inputs=base.input, outputs=pred_layer)

    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['acc'])

    return model
