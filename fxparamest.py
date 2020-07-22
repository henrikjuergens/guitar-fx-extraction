"""Estimates the parameters of the used Audio Effect"""

import os
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import joblib
from keras import models, layers, optimizers, utils
from wavtoarray import DATA_PATH
import plots


def get_dist_feat(y_cut, sr):
    """Extracts features for Distortion parameter estimation"""
    v_features = []
    mfcc = librosa.feature.mfcc(y=y_cut, sr=sr, n_mfcc=3)
    mfcc_delta = librosa.feature.delta(mfcc)
    m_features_mfcc = np.concatenate((mfcc, mfcc_delta))

    for feat in m_features_mfcc:
        lin_coeff, lin_residual, _, _, _ = np.polyfit(np.arange(len(feat)), feat, 1, full=True)
        v_features.extend(lin_coeff)
        # v_features.append(lin_residual)

    return v_features


def get_trem_feat(y_cut, sr):
    """Extracts features for Tremolo parameter estimation"""
    rms = librosa.feature.rms(S=librosa.core.stft(y_cut))
    rms_delta = librosa.feature.delta(rms)
    m_features_rms = np.concatenate((rms, rms_delta))

    v_features = []

    for feat in m_features_rms:

        feat_cut = feat - np.average(feat)
        # feat[5:round(0.66*len(feat))]  # Cut the Fadeout at two thirds of the file
        # plots.rms_lin_reg(feat_cut)

        feat_windowed = feat_cut * np.hanning(len(feat_cut))
        feat_int = np.pad(feat_windowed, (0, 1024 - len(feat_windowed) % 1024), 'constant')

        rfft = np.fft.rfft(feat_int)
        rfft_norm = np.abs(rfft) * 4 / 1024
        # plots.rms_fft(rfft_norm)
        rfft_max = np.max(rfft_norm)
        rfft_max_ind = np.argmax(rfft_norm)
        low_limit = rfft_max_ind - 32 if rfft_max_ind - 32 >= 0 else 0
        high_limit = rfft_max_ind + 32 if rfft_max_ind + 32 <= len(rfft_norm) else len(rfft_norm)
        rfft_norm[low_limit:high_limit] = np.zeros(high_limit - low_limit)

        rfft_max2_ind = np.argmax(rfft_norm)
        if rfft_max_ind < rfft_max2_ind:
            v_features.extend([rfft_max, rfft_max_ind,
                               np.max(rfft_norm), rfft_max2_ind])
        else:
            v_features.extend([np.max(rfft_norm), rfft_max2_ind,
                               rfft_max, rfft_max_ind])

    return v_features


def get_dly_feat(y_cut, sr, y):
    """Extracts features for Delay parameter estimation"""
    # uncut_onset_strength = librosa.onset.onset_strength(y=y_cut, sr=sr)
    onset_strength = librosa.onset.onset_strength(y=y_cut, sr=sr)
    onset_strength = np.reshape(onset_strength, [1, len(onset_strength)])
    v_features = []

    dly_onsets = librosa.onset.onset_detect(y=y_cut, sr=sr, units='frames', backtrack=False)

    dtype = [('onset_strength', float), ('onset', int)]
    all_onsets_strength = [(onset_strength[0, onset], onset) for onset in dly_onsets]
    all_onsets_strength_np = np.array(all_onsets_strength, dtype=dtype)
    onsets_sorted = np.sort(all_onsets_strength_np, order='onset_strength')
    strongest_onset = onsets_sorted[-1]
    if len(onsets_sorted) > 1:
        print('More than one onset candidate found')
        strongest_onset_2 = onsets_sorted[-2]
    else:
        strongest_onset_2 = np.array((0, 0), dtype=dtype)

    mfcc_delta = librosa.feature.delta(librosa.feature.mfcc(y_cut, sr=sr, n_mfcc=1)
                                       )[:, strongest_onset['onset']-5:strongest_onset['onset']+3]
    if len(onsets_sorted) > 1:
        mfcc_delta_2 = librosa.feature.delta(librosa.feature.mfcc(y_cut, sr=sr, n_mfcc=1)
                                             )[:, strongest_onset_2['onset']-5:strongest_onset_2['onset']+3]
    else:
        mfcc_delta_2 = np.zeros((1, 8))
    mfcc_delta_sum = np.sum(mfcc_delta, axis=1)
    mfcc_delta_sum_2 = np.sum(mfcc_delta_2, axis=1)
    rms = librosa.amplitude_to_db(librosa.feature.rms(y_cut)).T
    v_features.extend([mfcc_delta_sum, strongest_onset['onset'], rms[strongest_onset['onset']],
                       mfcc_delta_sum_2, strongest_onset_2['onset'], rms[strongest_onset_2['onset']]])

    # plots.onsets_and_strength(all_onsets_strength, onsets_sorted, dly_onsets, strongest_onset,
    #                         strongest_onset_2, y_cut, onset_strength)

    return v_features


def read_data(path_folder):
    """Reads sample data from files and extracts features"""
    os.chdir(DATA_PATH)
    sample_paths = ['Gitarre monophon/Samples/NoFX', 'Gitarre polyphon/Samples/NoFX']
    train_data = []
    train_labels = []
    for path in sample_paths:
        sample_path = os.path.join(path_folder, path)
        os.chdir(sample_path)
        for file_name in os.listdir(os.getcwd()):
            if file_name.endswith(".wav"):
                print(file_name)
                os.chdir(Path('../../Labels'))
                # Label names are: Edge, Gain, Tone
                label_file = file_name[:-4] + '.pickle'
                # label = [0.0, 0.0, 0.0]
                with open(label_file, 'rb') as handle:
                    label = pickle.load(handle)
                    print(label)
                    if path_folder == 'DlyRandomSamples':  # Fix limited delay plugin range
                        label[0] = label[0]*4.0
                        label[1] = label[1]*10.0

                os.chdir('../Samples/NoFX')
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

                v_features = []
                if path_folder == 'DistRandomSamples':
                    v_features = get_dist_feat(y_cut=y_cut, sr=sr)
                elif path_folder == 'TremRandomSamples':
                    v_features = get_trem_feat(y_cut=y_cut, sr=sr)
                elif path_folder == 'DlyRandomSamples':
                    v_features = get_dly_feat(y_cut=y_cut, sr=sr, y=y)
                else:
                    print('Sample folder for feature extraction not found')

                train_data.append(np.hstack(v_features))
        os.chdir(DATA_PATH)

    train_data = np.array(train_data)
    print(train_data.shape)
    scaler = preprocessing.StandardScaler()
    train_data = scaler.fit_transform(train_data)

    train_labels = np.array(train_labels)
    os.chdir(DATA_PATH)
    return train_data, train_labels


def create_model(input_dim, output_dim):
    """Creates the Neural Network for the estimation"""
    model = models.Sequential()
    model.add(layers.Dense(32, input_dim=input_dim))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    # model.add(layers.Dense(32, activation='relu'))
    # model.add(layers.Dense(16, activation='relu'))

    model.add(layers.Dense(output_dim, activation='linear'))
    model.summary()
    model.compile(optimizer=optimizers.Adam(),
                  loss='mean_squared_error',
                  metrics=['mse'])
    return model


def train_model(model, train_data, train_labels):
    """Trains the model for the estimation"""
    utils.normalize(train_data)
    history = model.fit(train_data, train_labels, epochs=1000, verbose=1, validation_split=0.2)
    # plots.learning_curve(history)


def estimate(folder_path):
    """Reads the data from folder path, trains the model, and estimates on test data"""
    os.chdir(DATA_PATH)
    os.chdir(folder_path)
    if not Path('ParamEstData.pickle').exists():
        train_data, train_labels = read_data(path_folder=folder_path)
        # plots.fx_par_data(train_data, train_labels)
        os.chdir(DATA_PATH)
        os.chdir(folder_path)
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
        print('Data Loaded')
        # plots.fx_par_data(train_data, train_labels)

    os.chdir(DATA_PATH)
    os.chdir(folder_path)
    if not Path('ParamEstModel.pickle').exists():
        my_model = create_model(train_data.shape[1], train_labels.shape[1])
        train_data, test_data, train_labels, test_labels = train_test_split(train_data, train_labels,
                                                                            test_size=0.3, random_state=42)
        train_model(my_model, train_data, train_labels)
        os.chdir(DATA_PATH)
        os.chdir(folder_path)
        with open('ParamEstModel.pickle', 'wb') as handle:
            joblib.dump(my_model, handle)
        print('Model Saved')
    else:
        with open('ParamEstModel.pickle', 'rb') as handle:
            my_model = joblib.load(handle)
        print('Model Loaded')
        # train_data, test_data, train_labels, test_labels = train_test_split(train_data, train_labels,
        #                                                                     test_size=0.3, random_state=42)
        test_data = train_data
        test_labels = train_labels
    pred = my_model.predict(test_data)

    print(pred)
    excl_threshold = 0.0  # Excludes data from evaluation, where true labels are below set threshold;
    # threshold = 0.0 does not exclude any samples

    pred_excl = pred[np.where(test_labels[:, 0] > excl_threshold)
                     and np.where(test_labels[:, 1] > excl_threshold)]
    test_labels_excl = test_labels[np.where(test_labels[:, 0] > excl_threshold)
                                   and np.where(test_labels[:, 1] > excl_threshold)]

    error = np.abs(pred_excl - test_labels_excl)
    random_error = np.reshape(np.abs(np.random.random(len(test_labels_excl))-test_labels_excl[:, 0]),
                              [len(test_labels_excl), 1])
    data_frames = []
    for (param_label, param_pred, param_error) in zip(test_labels.T.tolist(), pred.T.tolist(),
                                                      error.T.tolist()):
        data_frames.append(pd.DataFrame({'Test Label': param_label, 'Prediction': param_pred,
                                         'Error': param_error}))
    plots.param_est_error_over_params(data_frames, test_labels, folder_path)

    # error = np.concatenate((error, random_error), axis=1)
    print(folder_path + ' Absolute Error Evaluation')
    print('Mean Error:')
    print(np.mean(error, axis=0))
    print('Standard Deviation:')
    print(np.std(error, axis=0))
    # print('Median Error:')
    # print(np.median(error, axis=0))
    with open('NNAbsoluteErrorRelu.pickle', 'wb') as handle:
        joblib.dump(error, handle)

    print('Prediction Error Saved')
    plots.param_est_error_boxplot(error, folder_path)


if __name__ == '__main__':
    fx = ['DistRandomSamples', 'TremRandomSamples', 'DlyRandomSamples']
    estimate(fx[1])
