"""Predict effects of custom .wav files with the saved classifier from featextr.py"""

import os
import numpy as np
from joblib import load
import matplotlib.pyplot as plt

import featextr
from wavtoarray import DATA_PATH
import wavtoarray

TEST_SAMPLES_PATH = 'MySamples/Current'

if __name__ == '__main__':
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    os.chdir(DATA_PATH)
    os.chdir(TEST_SAMPLES_PATH)
    samples = []
    for file in os.listdir(os.getcwd()):
        if file.endswith(".wav"):
            samples.append(wavtoarray.read_custom_wav(os.getcwd(), file, file))

    X_list = []
    for smp in samples:
        X_new, y_new = featextr.extract_features(smp, training=False)
        X_list.append(X_new[0])
    # plt.show()  # uncomment this and iterate over for smp in samples[0:2] to plot multiple samples

    os.chdir(DATA_PATH)
    print(os.getcwd())
    clf = featextr.get_classifier()
    X = np.array(X_list)

    scaler = load('Scaler.joblib')
    X_scal = scaler.transform(X)

    pred = clf.predict(X_scal)
    prob = clf.predict_proba(X_scal)
    print(X.shape)
    for index, smp in enumerate(samples):
        print(str(smp.file_name) + ' Prediction: ' + str(pred[index]) + '\nProbabilities: '
              + str(prob[index]))
        print()
    print('Done')
