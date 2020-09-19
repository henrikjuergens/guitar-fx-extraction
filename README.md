# Recognizing Guitar Effects and Their Parameter Settings
This repostory contains the source code to the paper "Recognizing Guitar Effects and Their Parameter Settings".
The paper presents a method to extract guitar effects and their parameter settings from audio.

The paper can be found here: https://dafx2020.mdw.ac.at/proceedings/papers/DAFx2020_paper_2.pdf

The master thesis for which this code was developed is available in German at: https://seafile.cloud.uni-hannover.de/f/4170d1de392b4e03baf4/?dl=1

The code is tested on OpenSUSE and Windows 10, different operating systems might require small changes.

## Installation
Download and extract the IDMT-SMT-AUDIO-EFFECTS Dataset from https://www.idmt.fraunhofer.de/en/business_units/m2d/smt/guitar.html
Download and extract the FX-Estimation Samples from https://seafile.cloud.uni-hannover.de/f/2157452034354e6aa989/?dl=1
(Alternatively these samples can be generated from the IDMT Dataset, see To Generate Data below.)

If you don't want to train the network yourself, you can download our models here: https://seafile.cloud.uni-hannover.de/f/e2f59129188c49adb58e/?dl=1

1. Install anaconda, if not already installed
2. Create new environment with .yml file in the repository: ```conda env create -f environment.yml```
3. Activate the environment: ```conda activate gitfx```
3. Install librosa into gitfx environment: ```conda install -c conda-forge librosa```
4. Install praat-parselmouth into gitfx environment: ```pip install praat-parselmouth```
5. Change the DATA_PATH directory in wav_to_array.py to the directory, where the guitar samples are stored.
6. Change the TEST_SAMPLES_PATH to the relative path from DATA_PATH, where your samples for testing the effect classification are stored.

## Get Started
#### For effect classification run:
1. wav_to_array.py
2. featextr.py
3. tests.py (For testing with your own .wav guitar samples)

#### For effect parameter estimation run:
1. fxparamest.py
2. distsetbaseline.py for training an SVM on the same problem

#### Plots
For plots uncomment specific plot in featextr.py or fxparamest.py. You can modify plots.py to change the plots.

### To Generate Data:
**Note**: The used distortion plugin is only available on Windows - it was chosen because it is similar to the one used for the the IDMT-SMT-AUDIO-EFFECTS Dataset.

The folder reaperscripts contains scripts for sample generation with the integrated scripting terminal of reaper (DAW)
1. Download the distortion and tremolo plugins from the following links: 
    - https://vst4free.com/plugin/582/
    - https://pechenegfx.blogspot.com/2014/11/the-plugin-pecheneg-tremolo.html
2. Create a new reaper project with one track and insert the plugin in the fx slot (as delay we use the ReaDelay)
3. Change the audio system of reaper to Dummy Audio (sample generation will be twice as fast)
4. Make sure all the paths for loading the samples are correct for your system
5. Open the script terminal with Actions -> Show Action List -> ReaScript: Load
6. Press Ctrl + S to run the script
(Will process ~1000 Samples per hour on an average machine)

## Citation
Please cite our paper in your publications if the paper/our code or our database data helps your research:
        
    @inproceedings{Juergens20,
        title = {Recognizing Guitar Effects and Their Parameter Settings},
        author = {J{\"u}rgens, Henrik and Hinrichs, Reemt and Ostermann, J{\"o}rn},
        crossref = {dafx20},
        pages = {310--316}
    }

    @proceedings{dafx20,
        booktitle = {Proceedings of the 23rd International Conference on Digital Audio Effects},
        year = {2020},
        month = {09},
        issn = {2413-6700},
        venue = {Vienna, Austria},
        editor = {Evangelista, Gianpaolo}
    }
