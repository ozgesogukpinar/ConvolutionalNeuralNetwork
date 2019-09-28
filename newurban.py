#!/usr/bin/env python
# coding: utf-8

import _pickle as cPickle  # python 3 change
# In[ ]:
import os
import random
import warnings

import librosa
import librosa.display
import numpy as np
import pandas as pd
import keras
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from keras.utils import to_categorical
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import keras
from keras.layers import Activation, Dense, Dropout, Conv2D, \
                         Flatten, MaxPooling2D
from keras.models import Sequential
from keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint,TensorBoard,ProgbarLogger
from sklearn.model_selection import train_test_split
import librosa
import librosa.display
import numpy as np
import pandas as pd
import random
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical
from numpy import argmax

# object serialization

# get_ipython().run_line_magic('matplotlib', 'inline')

# In[ ]:


# In[3]:


# enable memory profiler for memory management usage %%memit
# from memory_profiler import memory_usage
#
# get_ipython().run_line_magic('load_ext', 'memory_profiler')

# enable garbage collection control
import gc

gc.enable()

# In[4]:


# progress tracker

# In[5]:


SKIP_AUDIO_RELOAD = False

# In[7]:


# location of the sound files
INPUT_PATH = '/media/kdtm/kdtm_backup/research/audio-classification/'

TRAIN_INPUT = INPUT_PATH + 'data/Train'
TRAIN_AUDIO_DIR = TRAIN_INPUT + 'data/Train'

TEST_INPUT = INPUT_PATH + 'data/Test'
TEST_AUDIO_DIR = TEST_INPUT + 'data/Test'


# In[8]:


def load_input_data(pd, filepath):
    # Read Data
    data = pd.read_csv(filepath)
    return data


# In[10]:

#
# # training file
# TRAIN_FILE = '/media/kdtl/3t/research/audio-classification/data/train.csv'
#
# # show info
# train_input = load_input_data(pd, TRAIN_FILE)
# train_input.head()
#
# # In[11]:
#
#
# # training file
# TEST_FILE = '/media/kdtl/3t/research/audio-classification/data/test.csv'
#
# # show info
# test_input = load_input_data(pd, TEST_FILE)
# test_input.head()
#
# # In[12]:
#
#
# # labels
# valid_train_label = train_input[['Class']]
# # x=data['label'].unique()
# valid_train_label.count()
#
# # unique classes
# x = train_input.groupby('Class')['Class'].count()
# x
#
# # In[13]:
#
#
# # # train data size
# valid_train_data = train_input[['ID', 'Class']]
# valid_train_data.count()
# #
# # # In[14]:
# #
# #
# # # test data size
# valid_test_data = test_input[['ID']]
# valid_test_data.count()

# In[16]:


# # sample-1 load
# sample1 = '/media/kdtl/3t/research/audio-classification/data/Train/943.wav'
# duration = 2.97
# sr = 22050
#
# y, sr = librosa.load(sample1, duration=duration, sr=sr)
# ps = librosa.feature.melspectrogram(y=y, sr=sr)
#
# input_length = sr * duration
# offset = len(y) - round(input_length)
# print("input:", round(input_length), " load:", len(y), " offset:", offset)
# print("y shape:", y.shape, " melspec shape:", ps.shape)
#
# # In[17]:
#
#
# # sample-1 waveplot
# librosa.display.waveplot(y, sr)
#
# # In[18]:
#
#
# # sample-1: audio
# import IPython.display as ipd
#
# ipd.Audio(sample1)
#
# # In[19]:
#
#
# # sample-1: spectrogram
# librosa.display.specshow(ps, y_axis='mel', x_axis='time')

# ## Prepare data file loading

# In[21]:


# # training audio files
# valid_train_data['path'] = '/media/kdtl/3t/research/audio-classification/data/Train' + '/' + train_input['ID'].astype(
#     'str') + ".wav"
# print("sample", valid_train_data.path[1])
# valid_train_data.head(5)
#
# # In[22]:
#
#
# # test audio files
# valid_test_data['path'] = '/media/kdtl/3t/research/audio-classification/data/Test' + '/' + test_input['ID'].astype(
#     'str') + ".wav"
# print("sample", valid_test_data.path[1])
#
# valid_test_data.head(5)


# Loading audio file and features

# In[23]:


#
# set duration on audio loading to make audio content to ensure each training data have same size
# 
# for instance, 3 seconds audio will have 128*128 which will be use on this notebook
#
def audio_norm(data):
    max_data = np.max(data)
    min_data = np.min(data)
    data = (data - min_data) / (max_data - min_data + 0.0001)
    return data - 0.5


# fix the load audio file size
# audio_play_duration = 2.97


def load_audio_file(file_path, duration=2.97, sr=22050):
    # load 5 seconds audio file, default 22 KHz default sr=22050
    # sr=resample to 16 KHz = 11025
    # sr=resample to 11 KHz = 16000
    # To preserve the native sampling rate of the file, use sr=None
    input_length = sr * duration
    # Load an audio file as a floating point time series.
    # y : np.ndarray [shape=(n,) or (2, n)] - audio time series
    # sr : number > 0 [scalar] - sampling rate of y
    y, sr = librosa.load(file_path, sr=sr, duration=duration)
    dur = librosa.get_duration(y=y)
    # pad output if audio file less than duration
    # Use edge-padding instead of zeros
    # librosa.util.fix_length(y, 10, mode='edge')
    if (round(dur) < duration):
        offset = len(y) - round(input_length)
        print("fixing audio length :", file_path)
        print("input:", round(input_length), " load:", len(y), " offset:", offset)
        y = librosa.util.fix_length(y, round(input_length))
        # y = audio_norm(y)
    # using a pre-computed power spectrogram
    # Short-time Fourier transform (STFT)
    # D = np.abs(librosa.stft(y))**2
    # ps = librosa.feature.melspectrogram(S=D)
    ps = librosa.feature.melspectrogram(y=y, sr=sr)
    return ps


# In[25]:

def prepare_data(SKIP_AUDIO_RELOAD):
    train_audio_data = []
    train_object_file = '/media/kdtm/kdtm_backup/research/audio-classification/data/saved_train_audio_data_0.20.p'

    # override the reload flag if serized file doesn't exist
    if not os.path.isfile(train_object_file):
        SKIP_AUDIO_RELOAD = False

    # load training data
    if SKIP_AUDIO_RELOAD is True:
        print("skip re-loading TRAINING data from audio files")
    else:
        print("loading train audio data, may take more than 15 minutes. please wait!")
        for row in tqdm(valid_train_data.itertuples()):
            ps = load_audio_file(file_path=row.path, duration=2.97)
            if ps.shape != (128, 128):
                continue
            train_audio_data.append((ps, row.Class))

    print("Number of train samples: ", len(train_audio_data))
    if SKIP_AUDIO_RELOAD is True:
        train_audio_data = cPickle.load(open(train_object_file, 'rb'))
        print("loaded train data [%s] records from object file" % len(train_audio_data))
    else:
        cPickle.dump(train_audio_data, open(train_object_file, 'wb'))
        print("saved loaded train data :", len(train_audio_data))

    test_audio_data = []
    test_object_file = '/media/kdtm/kdtm_backup/research/audio-classification/data/saved_test_audio_data_0.20.p'

    # override the reload flag if serized file doesn't exist
    if not os.path.isfile(test_object_file):
        SKIP_AUDIO_RELOAD = False

    if SKIP_AUDIO_RELOAD is True:
        print("skip re-loading TEST data from audio files")
    else:
        print("loading test audio data, may take more than 15 minutes. please wait!")
        for row in tqdm(valid_test_data.itertuples()):
            ps = load_audio_file(file_path=row.path, duration=2.97)
            if ps.shape != (128, 128):
                print("***data shape is wrong, replace it with zeros ", ps.shape, row.path)
                ps = np.zeros([128, 128])

            test_audio_data.append((ps, row.ID))
        print("Number of test samples: ", len(test_audio_data))

    if SKIP_AUDIO_RELOAD is True:
        test_audio_data = cPickle.load(open(test_object_file, 'rb'))
        print("loaded train data [%s] records from object file" % len(train_audio_data))
    else:
        cPickle.dump(test_audio_data, open(test_object_file, 'wb'))
        print("saved loaded test data :", len(test_audio_data))

    list_labels = encode_labels()

    #split_data_set(train_audio_data)
    X_train, X_test, y_train, y_test =split_data_set(train_audio_data)

    encoding_to_integer(y_train, y_test)
    label_encoder = LabelEncoder()
    y_train = np.array(keras.utils.to_categorical(label_encoder.fit_transform(y_train), len(list_labels)))
    y_test = np.array(keras.utils.to_categorical(label_encoder.fit_transform(y_test), len(list_labels)))

    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.20, random_state=42)

    print("test ", X_test.shape, len(y_test))
    print("valid ", X_val.shape, len(y_val))

    #print("list labels ", list_labels, len(list_labels))
    #print("X_train ", X_train, len(X_train))
    return X_train, X_test, y_train, y_test, X_val, y_val, list_labels


def encode_labels():
    # get a set of unique text labels
    list_labels = sorted(list(set(valid_train_data.Class.values)))
    print("unique text labels count: ", len(list_labels))
    print("labels: ", list_labels)

    # integer encode
    label_encoder = LabelEncoder()
    label_integer_encoded = label_encoder.fit_transform(list_labels)
    print("encoded labelint values", label_integer_encoded)

    # one hot encode
    encoded_test = to_categorical(label_integer_encoded)
    inverted_test = argmax(encoded_test[0])
    # print(encoded_test, inverted_test)

    # map filename to label
    file_to_label = {k: v for k, v in zip(valid_train_data.path.values, valid_train_data.ID.values)}
    # Map integer value to text labels
    label_to_int = {k: v for v, k in enumerate(list_labels)}
    # print ("test label to int ",label_to_int["Applause"])

    # map integer to text labels
    int_to_label = {v: k for k, v in label_to_int.items()}
    return list_labels

def split_data_set(train_audio_data=None):
    # full dataset
    dataset = train_audio_data
    random.shuffle(dataset)

    RATIO = 0.9
    train_cutoff = round(len(dataset) * RATIO)
    train = dataset[:train_cutoff]
    test = dataset[train_cutoff:]

    X_train, y_train = zip(*train)
    X_test, y_test = zip(*test)

    # Reshape for CNN input
    X_train = np.array([x.reshape((128, 128, 1)) for x in X_train])
    X_test = np.array([x.reshape((128, 128, 1)) for x in X_test])

    print("train ", X_train.shape, len(y_train))
    print("test ", X_test.shape, len(y_test))

    return X_train, X_test, y_train, y_test



def encoding_to_integer(y_train=None, y_test=None):
    label_encoder = LabelEncoder()
    y_train_integer_encoded = label_encoder.fit_transform(y_train)
    y_test_integer_encoded = label_encoder.fit_transform(y_test)


def create_models(X_train , list_labels):
    # build convolution model
    # input shape = (128, 128, 1)
    model = Sequential()
    input_shape = X_train.shape[1:]

    model.add(Conv2D(24, (5, 5), strides=(1, 1), input_shape=input_shape))
    model.add(MaxPooling2D((4, 2), strides=(4, 2)))
    model.add(Activation('relu'))

    model.add(Conv2D(48, (5, 5), padding="valid"))
    model.add(MaxPooling2D((4, 2), strides=(4, 2)))
    model.add(Activation('relu'))

    model.add(Conv2D(48, (5, 5), padding="valid"))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dropout(rate=0.5))

    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(rate=0.5))

    model.add(Dense(len(list_labels)))
    model.add(Activation('softmax'))
    model.summary()

    MAX_EPOCHS = 50
    MAX_BATCH_SIZE = 20
    # learning rate reduction rate
    MAX_PATIENT = 2

    # saved model checkpoint file
    best_model_file = "./best_model_trained.hdf5"

    # callbacks
    # removed EarlyStopping(patience=MAX_PATIENT)
    callback = [ReduceLROnPlateau(patience=MAX_PATIENT, verbose=1), ModelCheckpoint(filepath=best_model_file, monitor='loss', verbose=1, save_best_only=True)]

    # compile
    model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=['accuracy'])

    # train
    print('training started.... please wait!')

    checkpoint = ModelCheckpoint(best_model_file, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]



    history = model.fit(x=X_train, y=y_train,
                            epochs=MAX_EPOCHS,
                        batch_size=MAX_BATCH_SIZE,
                        verbose=1,
                        validation_data=(X_val, y_val),
                        callbacks=callbacks_list
                        )
    history = model.fit(X_train, y_train, batch_size=MAX_BATCH_SIZE, nb_epoch=MAX_EPOCHS, verbose=1)

    print('training finished')

    # quick evaludate model
    print('Evaluate model with test data')
    score = model.evaluate(x=X_test, y=y_test)

    print('test loss:', score[0])
    print('test accuracy:', score[1])





if __name__ == '__main__':
    # training file
    TRAIN_FILE = '/media/kdtm/kdtm_backup/research/audio-classification/data/train.csv'

    # show info
    train_input = load_input_data(pd, TRAIN_FILE)
    train_input.head()

    # In[11]:

    # training file
    TEST_FILE = '/media/kdtm/kdtm_backup/research/audio-classification/data/test.csv'

    # show info
    test_input = load_input_data(pd, TEST_FILE)
    test_input.head()

    # # train data size
    valid_train_data = train_input[['ID', 'Class']]
    valid_train_data.count()
    #
    # # In[14]:
    #
    #
    # # test data size
    valid_test_data = test_input[['ID']]
    valid_test_data.count()

    # training audio files
    valid_train_data['path'] = '/media/kdtm/kdtm_backup/research/audio-classification/data/Train' + '/' + train_input[
        'ID'].astype(
        'str') + ".wav"
    print("sample", valid_train_data.path[1])
    valid_train_data.head(5)

    # In[22]:

    # test audio files
    valid_test_data['path'] = '/media/kdtm/kdtm_backup/research/audio-classification/data/Test' + '/' + test_input['ID'].astype(
        'str') + ".wav"
    print("sample", valid_test_data.path[1])

    valid_test_data.head(5)

    audio_play_duration = 2.97

    X_train, X_test, y_train, y_test, X_val, y_val, list_labels = prepare_data(True)
    model = create_models(X_train, list_labels)








