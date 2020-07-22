"""Plotting graphs of features etc"""
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm
from scipy.stats.stats import pearsonr
import joblib
from wavtoarray import DATA_PATH

matplotlib.rcParams.update({'font.size': 18})  # Bigger font on all plots, full screen before exporting


def lin_regression(feat, lin_coeff):
    plt.title('Linear Regression of Feature')
    plt.xlabel('Frames')
    plt.ylabel('Feature')
    plt.plot(feat, 's', label='Feature')
    x_axis = range(0, feat.shape[0])
    lin_reg = lin_coeff[0]*x_axis + lin_coeff[1]
    plt.plot(lin_reg, label='Linear Regression')
    plt.legend()
    plt.show()


def phase_fmax_old(phase_freq_max, phase_fmax_straight_t, coeff, x_axis_t, linregerr_t):
    plt.subplot(4, 1, 1)
    plt.plot(phase_freq_max)
    plt.subplot(4, 1, 2)
    plt.plot(phase_fmax_straight_t)
    plt.subplot(4, 1, 3)
    plt.plot(coeff[0]*x_axis_t + coeff[1])
    plt.subplot(4, 1, 4)
    plt.plot(linregerr_t[0])
    plt.show()


def pitch_voiced_curve(pitch_curve, voice_prob):
    fig, ax1 = plt.subplots()
    plt.title('F0-Estimation and Voiced Probability Curve')
    ax1.set_xlabel('Frames')
    ax1.set_ylabel('Frequency in Hz', color='blue')
    ax1.set_ylim(-20, 330)
    ax1.plot(pitch_curve, zorder=2, color='blue')
    ax2 = ax1.twinx()
    ax2.set_ylim(-2/30, 1.1)
    ax2.set_ylabel('Voiced Probability', color='red')
    ax2.plot(voice_prob, color='red', zorder=1)
    plt.show()


def waveform(sample, onset_sample):
    sample_axis = range(0, sample.sig.shape[0])
    time_axis = librosa.core.samples_to_time(sample_axis, sr=sample.fs)
    plt.plot(time_axis, sample.sig, zorder=1)
    plt.vlines(librosa.core.samples_to_time(onset_sample, sr=44100), 1.0, -1.0, zorder=2)
    plt.xlabel('Time in Seconds')
    plt.ylabel('Floating Point Amplitude')
    plt.title('Waveform and Onset of a Guitar Sample')
    plt.show()


def spec_contrast(spec_contr):
    librosa.display.specshow(spec_contr, x_axis='time')
    plt.colorbar()
    plt.ylabel('Frequency Bands')
    plt.title('Spectral Contrast')
    plt.show()


def create_feature_list():
    feature_list = [['mfcc', 20], ['mfcc_delta', 20], ['spec_contr', 7],
                    ['zero_cr', 1], ['zero_cr_delta', 1], ['rms', 1], ['rms_delta', 1],
                    ['phase_res', 1], ['pitch_curve', 1], ['voice_prob', 1]]
    functionals_list = ['max', 'min', 'mean', 'std', 'lin_coeff_1', 'lin_coeff_2',
                        'lin_reg_residual', 'quad_coeff_1', 'quad_coeff_2', 'quad_coeff_3',
                        'quad_reg_residual', 'fft_max']
    ind_feat_list = []
    for feature in feature_list:
        ind_feat_list.extend([str(feature[0]) + str(i) for i in range(feature[1])])
    all_feat_list = []
    for f in ind_feat_list:
        all_feat_list.extend([f + ' ' + str(functionals_list[i])
                              for i in range(len(functionals_list))])
    all_feat_list.append('hnr')
    all_feat_list_df = pd.DataFrame(all_feat_list)
    os.chdir(DATA_PATH)
    all_feat_list_df.to_csv('AllFeaturesListed.csv', sep=';')


def feat_importance(X_train, y_train, X_test, y_test):
    print('Training Extra Trees for feature importance')
    clf_feat_importance = ExtraTreesClassifier(n_estimators=1000, random_state=42, n_jobs=-1)
    clf_feat_importance = clf_feat_importance.fit(X_train, y_train)
    y_pred_feat_importance = clf_feat_importance.predict(X_test)
    print('Accuracy Feature Importance: ' + str(accuracy_score(y_test, y_pred_feat_importance)))
    feat_importances = clf_feat_importance.feature_importances_
    indices = np.argsort(feat_importances)[::-1]
    plt.bar(range(X_train.shape[1]), feat_importances[indices])
    plt.xticks(range(X_train.shape[1]), indices)
    plt.ylabel('Feature Importance')
    plt.xlabel('Features')
    plt.xlim([-1, X_train.shape[1]])
    plt.show()

    feat_importances_df = pd.DataFrame(feat_importances)
    feat_importances_df.to_csv('FeatureImportance.csv', sep=';')


def git_spec():
    os.chdir(DATA_PATH)
    os.chdir('Gitarre monophon\\Samples\\NoFX')
    sample = librosa.load('G61-40100-1111-20593.wav', sr=44100)[0]
    D = np.abs(librosa.core.stft(sample))
    # fft_freq = librosa.core.fft_frequencies(sr=44100)
    librosa.display.specshow(librosa.amplitude_to_db(D,
                                                     ref=np.max),
                             y_axis='log', x_axis='time')
    plt.title('Spectrogram of a Guitar Sample')
    plt.colorbar(format='%+2.0f dB')
    # plt.tight_layout()
    plt.show()


def mel_filter_bank():
    melfb = librosa.filters.mel(sr=44100, n_fft=2048, n_mels=16, norm=None)
    plt.figure()
    print(melfb)
    librosa.display.specshow(melfb, x_axis='linear')
    plt.ylabel('Mel Filter')
    plt.xlabel('Frequency in Hz')
    plt.title('Mel filter bank')
    plt.colorbar()
    # plt.tight_layout()
    plt.show()


def mfcc_git():
    os.chdir(DATA_PATH)
    os.chdir('Gitarre monophon\\Samples\\NoFX')
    sample = librosa.load('G61-40100-1111-20593.wav', sr=44100)[0]
    mfcc = librosa.feature.mfcc(sample, sr=44100, n_mfcc=3)
    librosa.display.specshow(mfcc, sr=44100,
                             x_axis='time')
    plt.title('MFCCs of a Guitar Sample')
    plt.xlabel('Time in Seconds')
    plt.ylabel('MFCCs')
    plt.colorbar()
    plt.show()


def rms_git():
    os.chdir(DATA_PATH)
    os.chdir('Gitarre monophon\\Samples\\NoFX')
    sample = librosa.load('G61-40100-1111-20593.wav', sr=44100)[0]
    sample = librosa.util.normalize(sample)
    rms = librosa.feature.rms(y=sample)[0]
    print(rms)
    rms_db = 10*np.log(np.where(rms < 0.005, 0.005, rms))
    plt.ylabel('RMS Energy in dB')
    plt.xlabel('Time in Seconds')
    x_axis = librosa.frames_to_time(range(0, len(rms_db)), sr=44100)
    plt.title('RMS Energy of a Guitar Sample')
    plt.plot(x_axis, rms_db)
    plt.show()


def dist_baseline_comp():
    """Plots Comparison between baseline from distsetbaseline and neural network from fxparamest"""
    os.chdir(DATA_PATH)
    baseline_abs_error = pd.read_csv('BaselineErrorDistRandom.csv', sep=';')
    baseline_abs_error_quant = baseline_abs_error[1::2]
    print(baseline_abs_error_quant)
    baseline_abs_error_float = baseline_abs_error[::2]
    print(baseline_abs_error_float)
    os.chdir('DistRandomSamples')
    with open('NNAbsoluteErrorRelu.pickle', 'rb') as handle:
        nn_abs_error_relu = joblib.load(handle).T
    print(nn_abs_error_relu.shape)
    baseline_abs_error = [e[1] for e in baseline_abs_error.values]

    errors = [nn_abs_error_relu[3], baseline_abs_error_quant.values[:, 1],
              baseline_abs_error_float.values[:, 1], nn_abs_error_relu[1]]
    plt.boxplot(errors, whis=[0, 100])
    plt.title('Absolute Error Comparison (Gain Parameter)')
    plt.xticks([1, 2, 3, 4], ['Random Error', 'SVM Clf\nQuantized',
                              'SVM Clf\nContinuous', 'Neural Network\n(ours)'])
    plt.ylabel('Absolute FP-Error')
    plt.show()


def kernel_example():
    # Our dataset and targets
    X = np.c_[(.4, -.7),
              (-1.5, -1),
              (-1.4, -.9),
              (-1.3, -1.2),
              (-1.1, -.2),
              (-1.2, -.4),
              (-.5, 1.2),
              (-1.5, 2.1),
              (1, 1),
              # --
              (1.3, .8),
              (1.2, .5),
              (.2, -2),
              (.5, -2.4),
              (.2, -2.3),
              (0, -2.7),
              (1.3, 2.1)].T
    Y = [0] * 8 + [1] * 8

    # figure number
    fignum = 1

    # fit the model
    for kernel in ('linear', 'rbf'):
        clf = svm.SVC(kernel=kernel, gamma=2)
        clf.fit(X, Y)

        # plot the line, the points, and the nearest vectors to the plane
        plt.figure(fignum, figsize=(4, 3))
        plt.clf()

        plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
                    facecolors='none', zorder=10, edgecolors='k')
        plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired,
                    edgecolors='k')

        plt.axis('tight')
        x_min = -3
        x_max = 3
        y_min = -3
        y_max = 3

        XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
        Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(XX.shape)
        plt.figure(fignum, figsize=(4, 3))
        plt.contourf(XX, YY, Z > 0, cmap=plt.cm.Paired)
        plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                    levels=[-.5, 0, .5])

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        plt.xticks(())
        plt.yticks(())
        if kernel == 'linear':
            plt.title('SVM-Classfication with Linear Kernel')
        if kernel == 'rbf':
            plt.title('SVM-Classfication with RBF Kernel')

        fignum = fignum + 1
    plt.show()


def plot_dist_par_data(X, y):  # Legacy code replaced by fx_par_data
    print(X.shape)
    print(y.shape)
    plt.plot(y[:, 0], X[:, 2], marker='.', linestyle='None', label='Steigung 2. MFCC')
    plt.plot(y[:, 1], X[:, 3], marker='.', linestyle='None', label='Offset 2. MFCC')
    for i in [1]:
        for j in range(0, len(y[0])):
            x_label = ['1. MFCC Steigung', '1. MFCC Offset', '2. MFCC Steigung', '2. MFCC Offset',
                       '3. MFCC Steigung', '3. MFCC Offset',
                       '1. MFCC Delta Steigung', '1. MFCC Delta Offset', '2. Delta MFCC Steigung',
                       '2. MFCC Delta Offset', '3. MFCC Delta Steigung', '3. MFCC Delta Offset']
            y_label = ['Edge', 'Gain', 'Tone']
            print(y_label[i], x_label[j])
            print(pearsonr(y[:, i], X[:, j])[0])
    plt.xlabel('Effekteinstellung des Gain-Parameters')
    plt.ylabel('Standardisierte Regressionsdaten')
    plt.legend()

    plt.show()


def fx_par_data(X, y):
    """Plots all features over all labels and calculates pearson_r for correlation"""
    print(X.shape)
    print(y.shape)
    for i in range(X.shape[1]):
        plt.plot(X[:, i], y[:, 0], marker='.', linestyle='None', label='FX Param 1')
        # plt.plot(X[:, i], y[:, 1], marker='.', linestyle='None', label='FX Param 2')
        plt.xlabel('Standardisierte Input-Daten des NN')
        plt.ylabel('Effekteinstellung')
        plt.legend()
        plt.show()

    for i in range(X.shape[1]):
        for j in range(len(y[0])):
            print(pearsonr(y[:, j], X[:, i])[0])


def rfft(rfft_norm):
    f = np.fft.rfftfreq(1024, 441 / 44100)
    plt.plot(f, rfft_norm)
    plt.title('FFT of the Feature Curve')
    plt.ylabel('Feature Magnitude')
    plt.xlabel('Frequency in Hz')
    plt.show()


def phase_spectrogram(phase):
    librosa.display.specshow(phase, y_axis='linear', x_axis='frames')
    plt.title('Phase Spectrogram of a Guitar Sample')
    plt.ylabel('Frequency in Hz')
    plt.xlabel('Frames')
    plt.colorbar(format='%2.2f rad')
    plt.show()


def phase_fmax(phase_freq_max):
    plt.plot(phase_freq_max)
    plt.xlabel('Frames')
    plt.ylabel('Phase in rad')
    plt.title('Phase of the Maximum Frequency Bin')
    plt.show()


def phase_error_unwrapped(phase_fmax_straight_t, coeff, x_axis_t):
    plt.plot(phase_fmax_straight_t, marker='.')
    plt.plot(coeff[0] * x_axis_t + coeff[1])
    plt.xlabel('Frames')
    plt.ylabel('Phase (unwrapped) in rad')
    plt.title('Phase of the Maximum Frequency Bin and Regression Line')
    plt.legend(['Phase', 'Regressionsgerade'])
    plt.show()


def spectrogram(smp_cut):
    S = librosa.stft(y=smp_cut)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(S), ref=np.max), x_axis='time', y_axis='log')
    plt.title('Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.ylabel('Frequency in Hz')
    plt.xlabel('Time in Seconds')
    plt.show()


def phase_reg_line_deviation(phase_res):
    plt.plot(phase_res[0])
    plt.xlabel('Frames')
    plt.ylabel('Deviation from Regression Line in rad')
    plt.title('Deviation from Regression Line')
    plt.legend(['NoFX Sample', 'Phaser Sample'])
    # plt.show  # to compare multiple samples show plot after reading of all samples in test.py


def pitch(pitch_curve):
    plt.plot(pitch_curve[0])
    plt.title('Pitch Curve')
    plt.ylabel('Frequency in Hz')
    plt.xlabel('Frames')
    plt.show()


def param_est_error_over_params(data_frames, test_labels, folder_path):
    df_tails = []
    for df in data_frames:
        for i in range(test_labels.shape[1]):
            label = 'Label Param ' + str(i)
            df[label] = test_labels[:, i]

    for df in data_frames:
        df_sorted = df.sort_values(by=['Error'])
        df_tails.append(df_sorted.tail(df_sorted.shape[0]))

    # Parameter order: (Distortion) Edge, Gain, Tone; (Tremolo) Depth, Frequency; (Delay) Wet, Length
    params = []
    if folder_path == 'DistRandomSamples':
        params = ['Edge', 'Gain', 'Tone']
    elif folder_path == 'TremRandomSamples':
        params = ['Depth', 'Frequency']
    elif folder_path == 'DlyRandomSamples':
        params = ['Wet', 'Length']

    for param_index, param in enumerate(params):
        error_column_name = param + ' Absolute Error'
        df_tails[param_index].rename(columns={'Error': error_column_name}, inplace=True)
        print(df_tails[param_index])
        other_params = params.copy()
        other_params.pop(param_index)
        for param2_index, param2 in enumerate(other_params):
            df_tails[param_index].plot.scatter(x='Label Param ' + str(param_index),
                                               y='Label Param ' +
                                                 str(params.index(param2)),
                                               s=80, c=error_column_name, colormap='magma_r')
            plt.xlabel('True ' + param + ' Setting')
            plt.ylabel('True ' + param2 + ' Setting')
            plt.title(param + ' Parameter Error in Relation to Parameter Settings')
            plt.show()


def rms_lin_reg(feat_cut):
    x_axis = np.arange(0, len(feat_cut))
    coeff = np.polyfit(x_axis, feat_cut, 1)
    feat_lin_reg = np.copy(feat_cut)
    feat_lin_reg -= (coeff[0] * x_axis + coeff[1])
    plt.plot(feat_lin_reg)
    plt.show()


def rms_fft(rfft_norm):
    f = np.fft.rfftfreq(1024, 512 / 44100)
    plt.plot(f, rfft_norm)

    plt.title('FFT of the RMS-Energy')
    plt.xlabel('Frequency in Hz')
    plt.ylabel('Amplitude (Linear RMS-Energy)')
    plt.show()


def onsets_and_strength(all_onsets_strength, onsets_sorted, dly_onsets, strongest_onset,
                        strongest_onset_2, y_cut, onset_strength):
    print(all_onsets_strength)
    print(onsets_sorted)
    plt.subplot(211)
    plt.vlines(librosa.frames_to_samples(dly_onsets), -1.0, 1.0, zorder=2)
    plt.vlines(librosa.frames_to_samples(strongest_onset['onset']), -1.0, 1.0, colors='red', zorder=3)
    plt.vlines(librosa.frames_to_samples(strongest_onset_2['onset']), -1.0, 1.0, colors='green', zorder=3)
    plt.plot(y_cut, zorder=1)
    plt.ylabel('Amplitude in Floating Point')
    plt.xlabel('Samples')
    plt.title('Onset Detection with Delay Effect')
    plt.subplot(212)
    plt.plot(onset_strength[0])
    plt.ylabel('Onset Stength')
    plt.xlabel('Frames')
    plt.show()


def param_est_error_boxplot(error, folder_path):
    plt.boxplot(error, whis=[5, 95])
    # Parameter order: (Distortion) Edge, Gain, Tone; (Tremolo) Depth, Frequency; (Delay) Wet, Length
    plt.title('FX-Parameter Estimation Error')
    if folder_path == 'DistRandomSamples':
        plt.xticks([1, 2, 3], ['Edge', 'Gain', 'Tone', 'Random Settings'])
        plt.hlines(y=0.05, xmin=0.5, xmax=3.5, colors='red', linestyles='--')

    elif folder_path == 'TremRandomSamples':
        plt.xticks([1, 2, 3], ['Depth', 'Frequency', 'Random Settings'])
        plt.hlines(y=0.05, xmin=0.5, xmax=2.5, colors='red', linestyles='--')

    elif folder_path == 'DlyRandomSamples':
        plt.xticks([1, 2, 3], ['Wet', 'Length', 'Random Settings'])
        plt.hlines(y=0.05, xmin=0.5, xmax=2.5, colors='red', linestyles='--')

    plt.ylabel('Absolute FP-Error')
    plt.text(1.5, 0.06, 'Estimated Human\nSetup Error', fontsize=14, horizontalalignment='center')
    plt.show()


def learning_curve(history):
    # Plot training & validation loss values
    plt.plot(history.history['loss'])  # Loss is mean squared error (=/= mean absolute error)
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.ylim(0.0, 0.2)
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.title('Learning Curve with 32-Neuron Hidden Layer')
    plt.show()

# if __name__ == '__main__':
#     dist_baseline_comp()
