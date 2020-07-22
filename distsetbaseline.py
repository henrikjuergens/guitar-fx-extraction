import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pickle
import librosa
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.externals.joblib import parallel_backend
from wavtoarray import DATA_PATH, Sample
import featextr
import pandas as pd


def read_data():
    os.chdir(DATA_PATH)
    dirs = ['DistRandomSamples/Gitarre monophon/Samples/NoFX',
            'DistRandomSamples/Gitarre polyphon/Samples/NoFX']

    all_samples = []
    for dr in dirs:
        os.chdir(dr)
        for file_name in os.listdir(os.getcwd()):
            if file_name.endswith(".wav"):
                print(file_name)
                os.chdir(Path('../../Labels'))
                # Label names are: Edge, Gain, Tone
                label_file = file_name[:-12] + '.wav-DistRnd.pickle'
                with open(label_file, 'rb') as handle:
                    label = pickle.load(handle)[1]

                os.chdir('../Samples/NoFX')
                librosa.load(file_name)
                all_samples.append(Sample(sig=librosa.load(file_name, sr=44100)[0], fs=44100,
                                          label=label, path=os.getcwd(), file_name=file_name))

        os.chdir(DATA_PATH)
    # train_labels = np.array(train_labels)
    return all_samples


all_samples = read_data()
X_list = []
y_list = []
for sample in all_samples:
    X_new, y_new = featextr.extract_features(sample, training=False)
    X_list.extend(X_new)
    y_list.extend(y_new)


X = np.array(X_list)
y = np.array(y_list)
X = np.nan_to_num(X, 0.0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
y_train = np.round(y_train*np.array(5))
y_test_float = np.copy(y_test)/5
y_test = np.round(y_test*np.array(5))

y_train = y_train.astype(int)
y_test = y_test.astype(int)


print('Standardizing data')
scaler = preprocessing.StandardScaler()
print(X_train.shape)
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
print('Training')
# gamma_list = [2**exp for exp in range(-14, -1)]
# C_list = [2**exp for exp in range(-2, 5)]
#
# tuned_parameters = [{'gamma': gamma_list, 'C': C_list}]
# clf = GridSearchCV(SVC(), tuned_parameters, cv=3, verbose=2)
#
# with parallel_backend('threading', n_jobs=-1):
#     clf.fit(X_train, y_train)

clf = SVC(gamma=0.0009765625, C=2)  # Best Hyperparameters (66,56% Accuracy)
clf.fit(X_train, y_train)
# print("Best parameters set found on development set:")
# print()
# # print(clf.best_params_)  # gamma = 2**(-10), C = 16
# print()

print(X_train.shape)

y_pred = clf.predict(X_test)
print('Accuracy: ' + str(accuracy_score(y_test, y_pred)))
y_error = []
for p, t, t_f in zip(y_pred, y_test, y_test_float):
    y_error.extend([np.abs(np.float(p)/5 - t_f), np.abs(np.float(p)/5-np.float(t)/5)])

y_error_df = pd.DataFrame(y_error)

os.chdir(DATA_PATH)
y_error_df.to_csv('BaselineErrorDistRandom.csv', sep=';')
