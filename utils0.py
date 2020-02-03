import keras

from cnn0 import DenseNetMultiClass
from bcnn0 import MCDropout, MCBatchNorm
from data0 import GuangzhouDatabase, NIHDatabase, StanfordDatabase, BethIsrDatabase


def get_data_from_string(data_str, target_size):
    if data_str == 'gzh':
        database = GuangzhouDatabase(target_size=target_size)
    elif data_str == 'nih':
        database = NIHDatabase(target_size=target_size)
    elif data_str == 'stf':
        database = StanfordDatabase(target_size=target_size)
    elif data_str == 'mit':
        database = BethIsrDatabase(target_size=target_size)
    else:
        raise ValueError

    return database


# Return a compiled model
def get_model_from_string(model_type, target_size, saved_model):
    if saved_model is not None:
        model = keras.models.load_model(saved_model)
        print('Model loaded from', saved_model)
        if model.name != model_type:
            print('Warning: Loaded model (name %s) may not be of %s architecture'
                  % (model.name, model_type))
    else:
        model = None

    if model_type == 'dn':
        return DenseNetMultiClass(target_size=target_size, model=model)
    elif model_type == 'mcdo':
        return MCDropout(target_size=target_size, model=model)
    elif model_type == 'mcbn':
        return MCBatchNorm(target_size=target_size, model=model)
    else:
        raise ValueError
