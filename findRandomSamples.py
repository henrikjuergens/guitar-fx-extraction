import os
import pickle
from wavtoarray import DATA_PATH

os.chdir(DATA_PATH)
fx = ['DistRandomSamples', 'TremRandomSamples', 'DlyRandomSamples']
path_folder = fx[3]
sample_paths = ['Gitarre monophon/Samples/NoFX', 'Gitarre polyphon/Samples/NoFX']
for path in sample_paths:
    sample_path = path_folder#os.path.join(path_folder, path)
    os.chdir(sample_path)
    train_data = []
    train_labels = []
    for file_name in os.listdir(os.getcwd()):
        if file_name.endswith(".wav"):
            print(file_name)
            # os.chdir(Path('../../Labels'))
            # Label names are: Edge, Gain, Tone
            label_file = file_name[:-4] + '.pickle'
            label = [0.0, 0.0, 0.0]
            with open(label_file, 'rb') as handle:
                label = pickle.load(handle)
                print(label)
                if path_folder == 'DlyRandomSamples':  # Fix limited delay plugin range
                    label[0] = label[0]*4.0
                    label[1] = label[1]*10.0

            # os.chdir('../Samples/NoFX')
            train_labels.append(label)
