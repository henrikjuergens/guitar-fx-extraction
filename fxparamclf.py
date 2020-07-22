"""Classifies the parameters of the used Audio Effect"""

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pickle
import librosa
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras import models
from keras import layers
from keras import optimizers
from wavtoarray import DATA_PATH


def read_data():
    os.chdir(DATA_PATH)
    os.chdir('Gitarre monophon/Samples/Distortion')
    train_data = []
    train_labels = []
    for file_name in os.listdir(os.getcwd()):
        if file_name.endswith(".wav"):
            print(file_name)
            # Labeling the sample with one hot encoding
            label_no = int(file_name[13])  # Effect setting is the label
            label = np.zeros([3])
            label[label_no-1] = 1
            train_labels.append(label)

            # Loading the audio
            y, sr = librosa.load(file_name, sr=44100)
            # Onset Detection
            y = np.insert(y, 0, np.zeros(1023))
            y = librosa.util.normalize(y)

            onset_frame = librosa.onset.onset_detect(y=y, sr=sr, units='frames',
                                                     pre_max=20000, post_max=20000,
                                                     pre_avg=20000, post_avg=20000, delta=0, wait=1000)
            offset_frame = librosa.samples_to_frames(samples=y.shape[0])
            onset_sample = librosa.core.frames_to_samples(onset_frame[0])
            offset_sample = librosa.core.frames_to_samples(offset_frame)
            y_cut = y[onset_sample:offset_sample]

            mfcc = librosa.feature.mfcc(y=y_cut, sr=sr, n_mfcc=2)
            mfcc_delta = librosa.feature.delta(mfcc)
            m_features = np.concatenate((mfcc, mfcc_delta))
            v_features = []
            for feat in m_features:
                lin_coeff, lin_residual, _, _, _ = np.polyfit(np.arange(len(feat)), feat, 1, full=True)
                v_features.extend(lin_coeff)
                # v_features.append(lin_residual)
            train_data.append(np.hstack(v_features))
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    return train_data, train_labels


def create_model(input_dim):
    model = models.Sequential()
    model.add(layers.Dense(64, input_dim=input_dim, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(3, activation='softmax', input_dim=input_dim))
    model.summary()
    model.compile(optimizer=optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    return model


def train_model(model, train_data, train_labels):
    tf.keras.utils.normalize(train_data)
    history = model.fit(train_data, train_labels, epochs=50, verbose=1, validation_split=0.2)
    # test_loss, test_acc = model.evaluate(X_test, y_test)
    # print(history)
    # print('Test Loss: %f Test Accuracy: %f', test_loss, test_acc)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


os.chdir(DATA_PATH)
if not Path('ParamEstData.pickle').exists():
    train_data, train_labels = read_data()
    with open('ParamEstData.pickle', 'wb') as handle:
        pickle.dump(train_data, handle)
    with open('ParamEstLabels.pickle', 'wb') as handle:
        pickle.dump(train_labels, handle)
    print('Data Saved')
else:
    with open('ParamEstData.pickle', 'rb') as handle:
        train_data = pickle.load(handle)
    with open('ParamEstLabels.pickle', 'rb') as handle:
        train_labels = pickle.load(handle)


train_model(create_model(train_data.shape[1]), train_data, train_labels)
train_labels_sk = [np.argmax(label) + 1 for label in train_labels]
print(train_labels_sk)
clf = SVC(kernel='linear')
train_data, test_data, train_labels_sk, test_labels_sk = train_test_split(train_data,
                                                                          train_labels_sk)
clf.fit(train_data, train_labels_sk)
pred = clf.predict(test_data)
print('SVM Accuracy: ' + str(accuracy_score(test_labels_sk, pred)))

