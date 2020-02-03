from __future__ import print_function

from keras.preprocessing.image import ImageDataGenerator
import pandas as pd

# TODO: Figure out class weights, add comments with database info


class CXRDatabase:
    def __init__(self):
        self.CLASS_LABELS = ['Atelectasis',
                             'Cardiomegaly',
                             'Consolidation',
                             'Edema',
                             'Effusion',
                             'Pneumonia',
                             'Pneumothorax']
        self.PNEUMONIA_IDX = 5

    # TODO: have a setting to normalize by self, instead of by ImageNet
    def get_train_val_datagen(self, val_split):
        train_val_datagen = ImageDataGenerator(
            horizontal_flip=True,
            rescale=1 / 255,
            # Normalizes each channel w.r.t. ImageNet
            preprocessing_function=self.preprocess_input,
            validation_split=val_split
        )
        return train_val_datagen

    def get_test_datagen(self):
        test_datagen = ImageDataGenerator(
            rescale=1 / 255,
            # Normalizes each channel w.r.t. ImageNet
            preprocessing_function=self.preprocess_input
        )
        return test_datagen

    def preprocess_input(self, x):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        return (x - mean) / std

    def get_train_valid_gen(self):
        raise NotImplementedError

    def get_test_gen(self):
        raise NotImplementedError


class StanfordDatabase(CXRDatabase):

    def __init__(self, pol=1, target_size=(224, 224)):
        super(StanfordDatabase, self).__init__()
        self.PATH_TO_TRAIN_VAL_DF = './data/Stanford/train_labels_edit.csv'
        self.PATH_TO_TEST_DF = './data/Stanford/valid_labels_edit.csv'
        self.PATH_TO_DIRECTORY = './data/Stanford/'

        # load dataframes
        self.train_val_df = pd.read_csv(self.PATH_TO_TRAIN_VAL_DF)
        self.test_df = pd.read_csv(self.PATH_TO_TEST_DF)

        # Apply uncertainty policy to -1 labels
        if pol:  # 'all-ones' uncertainty policy
            replace_dict = {-1: 1}
        else:  # 'all-zeros' uncertainty policy
            replace_dict = {-1: 0}
        self.train_val_df = self.train_val_df.replace(replace_dict)
        self.test_df = self.test_df.replace(replace_dict)

        self.target_size = target_size
        self.class_mode = 'other'

    def get_train_val_gen(self, val_split=0.15, batch_size=16, sample=None):  # 191027 (orig), 165630 (edit)
        train_val_datagen = self.get_train_val_datagen(val_split)
        gen_params = {
            'batch_size': batch_size,
            'dataframe': self.train_val_df,
            'x_col': 'filename',
            'y_col': self.CLASS_LABELS,
            'directory': self.PATH_TO_DIRECTORY,
            'class_mode': self.class_mode,
            'target_size': self.target_size,
        }

        if sample is not None:
            gen_params.update({'dataframe': self.train_val_df.sample(frac=sample)})

        train_gen = train_val_datagen.flow_from_dataframe(
            subset='training', **gen_params
        )

        val_gen = train_val_datagen.flow_from_dataframe(
            subset='validation', **gen_params
        )

        return train_gen, val_gen

    def get_test_gen(self, batch_size=159, sample=None):  # 202 (orig), 25599 (edit)
        test_datagen = self.get_test_datagen()
        gen_params = {
            'batch_size': batch_size,
            'dataframe': self.test_df,
            'x_col': 'filename',
            'y_col': self.CLASS_LABELS,
            'directory': self.PATH_TO_DIRECTORY,
            'class_mode': self.class_mode,
            'target_size': self.target_size,
        }

        if sample is not None:
            gen_params.update({'dataframe': self.test_df.sample(frac=sample)})

        test_gen = test_datagen.flow_from_dataframe(**gen_params)

        return test_gen


class NIHDatabase(CXRDatabase):
    def __init__(self, target_size=(224, 224)):
        super(NIHDatabase, self).__init__()
        self.PATH_TO_DF = './data/NIH/nih_labels_path_orig_split.csv'
        self.PATH_TO_DIRECTORY = './data/NIH/images/'

        self.df = pd.read_csv(self.PATH_TO_DF)

        self.target_size = target_size
        self.class_mode = 'other'

    def get_train_val_gen(self, val_split=0.15, batch_size=16, sample=None):  # 86524 total
        train_val_datagen = self.get_train_val_datagen(val_split)
        gen_params = {
            'batch_size': batch_size,
            'dataframe': self.df[self.df['fold'] == 'train'],
            'x_col': 'Image Index',
            'y_col': self.CLASS_LABELS,
            'directory': self.PATH_TO_DIRECTORY,
            'class_mode': self.class_mode,
            'target_size': self.target_size,
        }

        if sample is not None:
            gen_params.update({'dataframe': self.df[self.df['fold'] == 'train'].sample(frac=sample)})

        train_gen = train_val_datagen.flow_from_dataframe(
            subset='training', **gen_params
        )

        val_gen = train_val_datagen.flow_from_dataframe(
            subset='validation', **gen_params
        )

        return (train_gen, val_gen)

    def get_test_gen(self, batch_size=36, sample=None):  # 25596 total
        test_datagen = self.get_test_datagen()
        gen_params = {
            'batch_size': batch_size,
            'dataframe': self.df[self.df['fold'] == 'test'],
            'x_col': 'Image Index',
            'y_col': self.CLASS_LABELS,
            'directory': self.PATH_TO_DIRECTORY,
            'class_mode': self.class_mode,
            'target_size': self.target_size,
        }

        if sample is not None:
            gen_params.update({'dataframe': self.df[self.df['fold'] == 'test'].sample(frac=sample)})

        test_gen = test_datagen.flow_from_dataframe(**gen_params)

        return test_gen


class GuangzhouDatabase(CXRDatabase):
    def __init__(self, target_size=(224, 224), class_mode='binary'):
        self.PATH_TO_TRAIN_VAL_DIRECTORY = './data/Guangzhou/chest_xray/train/'
        self.PATH_TO_TEST_DIRECTORY = './data/Guangzhou/chest_xray/test/'

        self.target_size = target_size
        self.class_mode = class_mode
        # weight for '0' class: (num. 0) / total train_val
        # TODO: CHANGE
        # w0 = (1349 + 3884) / (2 * 1349)
        # w1 = (1349 + 3884) / (2 * 3884)
        w0 = 3884 / (1349 + 3884)
        w1 = 1349 / (1349 + 3884)
        self.class_weight = {0: w0, 1: w1}

    def get_train_val_gen(self, val_split=0.15, batch_size=16):
        train_val_datagen = self.get_train_val_datagen(val_split)
        gen_params = {
            'batch_size': batch_size,
            'directory': self.PATH_TO_TRAIN_VAL_DIRECTORY,
            'class_mode': self.class_mode,
            'target_size': self.target_size,
        }

        train_gen = train_val_datagen.flow_from_directory(
            subset='training', **gen_params
        )

        val_gen = train_val_datagen.flow_from_directory(
            subset='validation', **gen_params
        )

        return (train_gen, val_gen)

    def get_test_gen(self, batch_size=39):  # 624 total
        test_datagen = self.get_test_datagen()
        gen_params = {
            'batch_size': batch_size,
            'directory': self.PATH_TO_TEST_DIRECTORY,
            'class_mode': self.class_mode,
            'target_size': self.target_size,
        }

        test_gen = test_datagen.flow_from_directory(**gen_params)

        return test_gen


class BethIsrDatabase(CXRDatabase):

    def __init__(self, pol=1, target_size=(224, 224)):
        super(BethIsrDatabase, self).__init__()
        self.PATH_TO_TRAIN_VAL_DF = './../../../work/twl16/train_edit.csv'
        self.PATH_TO_TEST_DF = './../../../work/twl16/valid_edit.csv'
        self.PATH_TO_DIRECTORY = './../../../work/twl16/'

        # load dataframes
        self.train_val_df = pd.read_csv(self.PATH_TO_TRAIN_VAL_DF)
        self.test_df = pd.read_csv(self.PATH_TO_TEST_DF)

        # Apply uncertainty policy to -1 labels
        if pol:  # 'all-ones' uncertainty policy
            replace_dict = {-1: 1}
        else:  # 'all-zeros' uncertainty policy
            replace_dict = {-1: 0}
        self.train_val_df = self.train_val_df.replace(replace_dict)
        self.test_df = self.test_df.replace(replace_dict)

        self.target_size = target_size
        self.class_mode = 'other'

    def get_train_val_gen(self, val_split=0.15, batch_size=16, sample=None):  # 244397 (edit)
        train_val_datagen = self.get_train_val_datagen(val_split)
        gen_params = {
            'batch_size': batch_size,
            'dataframe': self.train_val_df,
            'x_col': 'path',
            'y_col': self.CLASS_LABELS,
            'directory': self.PATH_TO_DIRECTORY,
            'class_mode': self.class_mode,
            'target_size': self.target_size,
        }

        if sample is not None:
            gen_params.update({'dataframe': self.train_val_df.sample(frac=sample)})

        train_gen = train_val_datagen.flow_from_dataframe(
            subset='training', **gen_params
        )

        val_gen = train_val_datagen.flow_from_dataframe(
            subset='validation', **gen_params
        )

        return train_gen, val_gen

    def get_test_gen(self, batch_size=159, sample=None):  # 25598 (edit)
        test_datagen = self.get_test_datagen()
        gen_params = {
            'batch_size': batch_size,
            'dataframe': self.test_df,
            'x_col': 'path',
            'y_col': self.CLASS_LABELS,
            'directory': self.PATH_TO_DIRECTORY,
            'class_mode': self.class_mode,
            'target_size': self.target_size,
        }

        if sample is not None:
            gen_params.update({'dataframe': self.test_df.sample(frac=sample)})

        test_gen = test_datagen.flow_from_dataframe(**gen_params)

        return test_gen
