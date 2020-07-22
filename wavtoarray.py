"""Reads .wav files and saves them as Sample objects to be processed quickly"""

from pathlib import Path
import os
import platform
import pickle
import librosa

DATA_PATH = Path('C:/Users/User/Documents/GitSamples') \
    if platform.system() == 'Windows' else os.path.join(os.path.expanduser('~'), 'tmp')


class Sample:
    """Contains all information for different processing modules"""
    def __init__(self, sig, fs, label, path, file_name):
        self.sig = sig
        self.fs = fs
        self.label = label
        self.path = path
        self.file_name = file_name


def read_data():
    """Reads .wav files from disk and pickles them as Sample objects"""
    all_samples = []
    directories = ['Gitarre monophon/Samples', 'Gitarre monophon2/Samples',
                   'Gitarre polyphon/Samples', 'Gitarre polyphon2/Samples']
    for dr in directories:
        os.chdir(DATA_PATH)
        os.chdir(Path(dr))
        print(dr)
        for effect_folder in os.listdir(os.getcwd()):
            os.chdir(effect_folder)
            print(effect_folder)
            if effect_folder == 'EQ':
                effect_folder = 'NoFX'
            for file_name in os.listdir(os.getcwd()):
                if file_name.endswith(".wav"):
                    sig, fs = librosa.load(file_name, sr=44100)

                    all_samples.append(Sample(sig, fs, label=effect_folder, path=os.getcwd(),
                                              file_name=file_name))
            os.chdir('..')
    os.chdir(DATA_PATH)
    with open('SamplesUnproc.pickle', 'wb') as handle:
        pickle.dump(all_samples, handle)


def read_custom_wav(file_path, file_name, label):
    """Returns Sample object from the specified .wav file"""
    os.chdir(file_path)
    sig, fs = librosa.load(file_name, sr=22050)
    smp = Sample(sig, fs, label=label, path=os.getcwd(),
                 file_name=file_name)
    return smp


if __name__ == "__main__":
    read_data()
