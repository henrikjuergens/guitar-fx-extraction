# C:\Users\User\Documents\Masterarbeit\Python\FeatureExtraction.py
"""Extracts features from database, saves to Data.npz and trains SVC using autosklearn.
Autosklearn detemines the optimal classfier for the problem, based on the same data.
In first experiments (10h runtime 1h per estimator) autosklearn
did not outperform a hyperparameter optimisation of the SVM.
Currently installing autosklearn is not possible (installation crashes), so the results could not be
reproduced or explored sufficiently. This is why this code is not part of the thesis,
but still included in the codebase, in case autosklearn can be properly installed again in the future
"""

from pathlib import Path
import os
from multiprocessing import Pool, cpu_count
import csv
import pickle
import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from joblib import dump, load
import wavtoarray
from wavtoarray import Sample
from wavtoarray import DATA_PATH
import plots
import parselmouth
from parselmouth.praat import call
import autosklearn.classification

#@profile
def functionals(m_features):
    """Applys functionals to time series data"""
    v_features = []
    for feat in m_features:
        v_features.append(feat.max())
        v_features.append(feat.min())
        v_features.append(feat.mean())
        v_features.append(feat.std())

        lin_coeff, lin_residual, _, _, _ = np.polyfit(np.arange(len(feat)), feat, 1, full=True)
        v_features.extend(lin_coeff)
        v_features.append(lin_residual)

        quad_coeff, quad_residual, _, _, _ = np.polyfit(np.arange(len(feat)), feat, 2, full=True)
        v_features.extend(quad_coeff)
        v_features.append(quad_residual)

        rfft = np.fft.rfft(feat - np.average(feat))
        avg_abs_feat = np.average(np.abs(feat)) if np.average(np.abs(feat)) > 0.01 else 1
        rfft_abs = np.abs(rfft)/avg_abs_feat

        v_features.append(rfft_abs.max())
        # v_features.append(np.random.random([1, 1])[0, 0])

    v_features = np.hstack(v_features)
    return v_features


def phase_fmax(sig):
    """Analyses phase error of frequency bin with maximal amplitude
        compared to pure sine wave"""
    D = librosa.stft(y=sig, hop_length=256)[20:256]
    S, P = librosa.core.magphase(D)
    phase = np.angle(P)
    spec_sum = S.sum(axis=1)
    max_bin = spec_sum.argmax()
    phase_freq_max = phase[max_bin]

    S_max_bin_mask = S[max_bin]
    thresh = S[max_bin].max()/8
    phase_freq_max = np.where(S_max_bin_mask > thresh, phase_freq_max, 0)
    phase_freq_max_t = np.trim_zeros(phase_freq_max)  # Using only phase with strong signal

    phase_fmax_straight_t = np.copy(phase_freq_max_t)
    diff_mean_sign = np.mean(np.sign(np.diff(phase_freq_max_t)))
    if diff_mean_sign > 0:
        for i in range(1, len(phase_fmax_straight_t)):
            if np.sign(phase_freq_max_t[i-1]) > np.sign(phase_freq_max_t[i]):
                phase_fmax_straight_t[i:] += 2*np.pi
    else:
        for i in range(1, len(phase_fmax_straight_t)):
            if np.sign(phase_freq_max_t[i - 1]) < np.sign(phase_freq_max_t[i]):
                phase_fmax_straight_t[i:] -= 2 * np.pi

    x_axis_t = np.arange(0, len(phase_fmax_straight_t))
    coeff = np.polyfit(x_axis_t, phase_fmax_straight_t, 1)
    linregerr_t = np.copy(phase_fmax_straight_t)
    linregerr_t -= (coeff[0] * x_axis_t + coeff[1])
    linregerr_t = np.reshape(linregerr_t, (1, len(linregerr_t)))

    return linregerr_t


def extract_features(sample):
    """Extracts features from sample"""
    X_list = []
    y_list = []

    print(sample.label)
    print(sample.file_name)
    snd = parselmouth.Sound(os.path.join(sample.path, sample.file_name))

    # Onset Detection
    sample.sig = np.insert(sample.sig, 0, np.zeros(1023))
    sample.sig = librosa.util.normalize(sample.sig)

    onset_frame = librosa.onset.onset_detect(y=sample.sig, sr=sample.fs, units='frames',
                                             pre_max=20000, post_max=20000,
                                             pre_avg=20000, post_avg=20000, delta=0, wait=1000)
    offset_frame = librosa.samples_to_frames(samples=sample.sig.shape[0])
    onset_sample = librosa.core.frames_to_samples(onset_frame[0])
    offset_sample = librosa.core.frames_to_samples(offset_frame)
    smp_cut = sample.sig[onset_sample:offset_sample]

    # Time series features from librosa
    mfcc = librosa.feature.mfcc(y=smp_cut, sr=sample.fs)
    mfcc_pos = (mfcc - np.min(mfcc))
    mfcc_norm = mfcc_pos / np.max(mfcc_pos) - np.mean(mfcc_pos)
    mfcc_delta = librosa.feature.delta(mfcc_norm)
    spec_contr = librosa.feature.spectral_contrast(y=smp_cut, sr=sample.fs)

    phase_res = phase_fmax(smp_cut)
    zero_cr = librosa.feature.zero_crossing_rate(y=smp_cut)
    # zero_cr -= zero_cr.mean()
    zero_cr_delta = librosa.feature.delta(zero_cr)
    rms = librosa.feature.rms(y=smp_cut)
    rms *= 1/rms.max()
    rms_delta = librosa.feature.delta(rms)

    # Time series features from praat
    pitch = snd.to_pitch().to_array()
    pitch_curve, voice_prob = zip(*pitch[0][:])
    pitch_curve = np.array(pitch_curve)
    voice_prob = np.array(voice_prob)
    pitch_onset = int((onset_sample/sample.sig.shape[0]) * pitch_curve.shape[0])
    pitch_curve = pitch_curve[pitch_onset:]
    voice_prob = voice_prob[pitch_onset:]

    pitch_curve = np.reshape(pitch_curve, [1, pitch_curve.shape[0]])
    voice_prob = np.reshape(voice_prob, [1, voice_prob.shape[0]])

    harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    m_features = np.concatenate((mfcc_norm, mfcc_delta, spec_contr,
                                 zero_cr, zero_cr_delta, rms, rms_delta))
    v_features = functionals(m_features)

    # phase_res and pitch curve have different lenghts from m_features, so functionals
    # need to be analysed individually
    v_features = np.append(v_features, functionals(phase_res))
    v_features = np.append(v_features, functionals(pitch_curve))
    v_features = np.append(v_features, functionals(voice_prob))
    v_features = np.append(v_features, hnr)

    X_list.append(v_features)
    y_list.append(sample.label)
    return X_list, y_list


def get_feat_data():
    """Extracts feature data or loads it, if already existent"""
    os.chdir(DATA_PATH)

    if not Path('Data.npz').exists():
        num_processes = cpu_count() // 16 if cpu_count() // 16 > 2 else 2
        pool = Pool(processes=num_processes)
        X_list = []
        y_list = []
        print('Loading data from pickle...')
        with open('SamplesUnproc.pickle', 'rb') as handle:
            all_smp = pickle.load(handle)
        print('Data successfully loaded!')
        print('Extracting features...')

        result = pool.map(extract_features, all_smp)
        plt.show()
        for i in range(0, len(result)):
            X_list.append(result[i][0][0])
            y_list.append(result[i][1][0])

        X = np.array(X_list)
        y = np.array(y_list)

        os.chdir(Path(DATA_PATH))
        np.savez('Data.npz', X=X, y=y)
    else:
        print('Loading feature data...')
        data = np.load('Data.npz')
        X = data['X']
        y = data['y']
        print('done')
    return X, y


def get_classifier(X_train=None, y_train=None):
    """Trains the classifier or loads it, if already existent"""
    os.chdir(DATA_PATH)
    if not Path('SVC.joblib').exists():
        print('Training')
        clf = SVC(gamma='auto', probability=True)
        clf.fit(X_train, y_train)
        print(X_train.shape)
    else:
        clf = load('SVC.joblib')
    return clf


def test_training_excl_feat(feature):
    """Tests accurancy difference when training without feature"""
    index_sum = feature[2]
    X_train_cut = feature[3].copy()
    y_train = feature[4]
    X_test_cut = feature[5].copy()
    y_test = feature[6]
    y_pred = feature[7]

    remove_range = range(index_sum, (index_sum + feature[1]))
    X_train_cut = np.delete(X_train_cut, remove_range, 1)
    X_test_cut = np.delete(X_test_cut, remove_range, 1)
    index_sum += feature[1]

    print('Training without ' + feature[0])
    clf_cut = SVC(gamma='auto')
    clf_cut.fit(X_train_cut, y_train)

    print('Testing without ' + feature[0])
    y_pred_cut = clf_cut.predict(X_test_cut)
    acc_diff = accuracy_score(y_test, y_pred_cut) - accuracy_score(y_test, y_pred)
    # print('Accuracy difference: ' + str(acc_diff))
    # m_cf = confusion_matrix(y_test, y_pred_cut)
    # m_cf_norm = m_cf.astype('float') / m_cf.sum(axis=1)[:, np.newaxis]
    return feature[0], acc_diff


def get_feature_list(X_train, y_train, X_test, y_test, y_pred):
    """Creates a list to annotate feature data"""
    feature_list = [['mfcc', 20], ['mfcc_delta', 20], ['spec_contr', 7],
                    ['zero_cr', 1], ['zero_cr_delta', 1], ['rms', 1], ['rms_delta', 1],
                    ['phase_res', 1], ['pitch_curve', 1], ['voice_prob', 1]]
    for feature in feature_list:
        feature[1] *= 12
    feature_list.append(['hnr', 1])

    index_sum = 0
    for feature in feature_list:
        feature.append(index_sum)
        index_sum += feature[1]
        feature.extend([X_train, y_train, X_test, y_test, y_pred])
    return feature_list


def test_excluding_feature(feature_list):
    labels = []
    accs = []
    for feat in feature_list:
        label, acc = test_training_excl_feat(feat)
        labels.append(label)
        accs.append(acc)
    print(labels)
    print(accs)
    training_without_df = pd.DataFrame(accs)
    training_without_df.index = pd.Index(labels)
    print(training_without_df)

    training_without_df.to_csv('training_res.csv', sep=';')


def test_excluding_functionals(X_train, y_train, X_test, y_test, y_pred):
    functionals_list = ['max', 'min', 'mean', 'std', 'lin_coeff_1', 'lin_coeff_2', 'lin_reg_residual'
                        'quad_coeff_1', 'quad_coeff_2', 'quad_coeff_3', 'quad_reg_residual', 'fft_max', 'rnd']
    acc_diff = []
    for index, functional in enumerate(functionals_list):
        X_train_cut = X_train.copy()
        X_test_cut = X_test.copy()
        remove_range = range(index, X_train.shape[1], len(functionals_list))
        X_train_cut = np.delete(X_train_cut, remove_range, axis=1)
        X_test_cut = np.delete(X_test_cut, remove_range, axis=1)
        print('Training without ' + functional)
        clf_cut = SVC(gamma='auto')
        clf_cut.fit(X_train_cut, y_train)

        print('Testing without ' + functional)
        y_pred_cut = clf_cut.predict(X_test_cut)
        acc_diff.append(accuracy_score(y_test, y_pred_cut) - accuracy_score(y_test, y_pred))
    acc_diff_df = pd.DataFrame(acc_diff, functionals_list)
    print(acc_diff_df)
    acc_diff_df.to_csv('test_excluding_functionals.csv', sep=';')


def train_svc():
    """Extracts/loads feature data and trains Support Vector Machine Classifier"""
    os.chdir(DATA_PATH)
    # wavtoarray.read_data() # for profiling
    X, y = get_feat_data()

    X = np.nan_to_num(X, 0.0)
    print('Splitting data')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print('Standardizing data')
    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    dump(scaler, 'Scaler.joblib')

    # clf = get_classifier(X_train=X_train, y_train=y_train)
    automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=36000,
        per_run_time_limit=7200, ensemble_size=1, ensemble_nbest=1, ensemble_memory_limit=3072, ml_memory_limit=65536, n_jobs=2)
    automl.fit(X_train, y_train)

    y_pred = automl.predict(X_test)
    print('Accuracy: ' + str(accuracy_score(y_test, y_pred)))
    print(automl.show_models())
    print(automl.sprint_statistics())

    # clf_feat_importance = ExtraTreesClassifier(n_estimators=100)
    # clf_feat_importance = clf_feat_importance.fit(X_train, y_train)
    # y_pred_feat_importance = clf_feat_importance.predict(X_test)
    # print('Accuracy Feature Importance: ' + str(accuracy_score(y_test, y_pred_feat_importance)))
    # feat_importances = clf_feat_importance.feature_importances_
    # indices = np.argsort(feat_importances)
    # plt.bar(range(X_train.shape[1]), feat_importances[indices])
    # feat_importances_df = pd.DataFrame(feat_importances)
    # feat_importances_df.to_csv('FeatureImportance')
    # plt.show()

    m_cf = confusion_matrix(y_test, y_pred)
    m_cf_norm = m_cf.astype('float') / m_cf.sum(axis=1)[:, np.newaxis]
    print(m_cf_norm)
    print(np.diagonal(m_cf_norm))
    print('Chorus, Distortion, Feedback Delay, Flanger, NoFX, Overdrive')
    print('Phaser, Reverb, SlapbackDelay, Tremolo, Vibrato')

    m_cf_df = pd.DataFrame(m_cf_norm)
    m_cf_df.index = pd.Index(['Chorus', 'Distortion', 'Feedback Delay', 'Flanger',
                              'NoFX', 'Overdrive', 'Phaser', 'Reverb', 'SlapbackDelay',
                              'Tremolo', 'Vibrato'])
    m_cf_df.columns = ['Chorus', 'Distortion', 'Feedback Delay',
                       'Flanger', 'NoFX', 'Overdrive', 'Phaser',
                       'Reverb', 'SlapbackDelay', 'Tremolo', 'Vibrato']

    m_cf_df.to_csv('Confusion_Matrix.csv', sep=';')
    dump(automl, 'AutomlClf.joblib')
    print('SVC saved')

    # test_excluding_feature(get_feature_list(X_train, y_train, X_test, y_test, y_pred))
    # test_excluding_functionals(X_train, y_train, X_test, y_test, y_pred)

    print('done')


if __name__ == '__main__':
    train_svc()
