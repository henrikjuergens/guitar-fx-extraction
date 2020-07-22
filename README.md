# Recognizing Guitar Effects and Their Parameter Settings
This repostory contains the source code to the paper "Recognizing Guitar Effects and Their Parameter Settings".
The paper presents a method to extract guitar effects and their parameter settings from audio.
The master thesis for which this code was developed is available in German at: https://seafile.cloud.uni-hannover.de/f/4170d1de392b4e03baf4/?dl=1

The code is tested on OpenSUSE and Windows 10, different operating systems might require small changes.

## Installation
Download and extract the IDMT-SMT-AUDIO-EFFECTS Dataset from https://www.idmt.fraunhofer.de/en/business_units/m2d/smt/guitar.html
Download and extract the FX-Estimation Samples from https://seafile.cloud.uni-hannover.de/f/2157452034354e6aa989/?dl=1
(Alternatively these samples can be generated from the IDMT Dataset, see To Generate Data below.)

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
The folder reaperscripts contains scripts for sample generation with the integrated scripting terminal of reaper (DAW)
1. Create a new reaper project with one track or open one of the provided projects
2. Change the audio system of reaper to Dummy Audio (sample generation will be twice as fast)
3. Make sure all the paths for loading the samples are correct for your system
4. Open the script terminal with Actions -> Show Action List -> ReaScript: Load
5. Press Ctrl + S to run the script
(Will process ~1000 Samples per hour on an average machine)

## Citation
Please cite our paper in your publications if the paper/our code or our database data helps your research:

*we will update the reference as soon as the full paper is published*
