#!/usr/bin/python
#
# This is an example of script for source separation of 3 tracks (drums, voice
# and piano) from an instantaneous mixture.
#
# Parameters used to initialize FASST in this example:
# * Mixture type : instantaneous.
# * Time-Frequency representation : STFT with 1024 frequency bins.
# * Source paramater Wex : Normally distributed random matrix (default init).
# * Source paramater Hex : Normally distributed random matrix (default init).
# * Source paramater A : balanced gains (left, middle, right)
# * Source paramater adaptability : free, all previous parameters are
#   updated during the iterative EM process.
# * Number of EM iterations : 200
#
###########################################################################
# Copyright 2018 Ewen Camberlein (INRIA), Romain Lebarbenchon (INRIA)
# This software is distributed under the terms of the GNU Public License
# version 3 (http://www.gnu.org/licenses/gpl.txt)
###########################################################################

from __future__ import division
import numpy as np
import os, sys
import wave
import shutil

# ------------------------------------------------------------------------
#                      Paths management
# ------------------------------------------------------------------------

# Path of current file
script_path = os.chdir('C:/Users/User/Music/FASSTExamples/FASST_2.2.2_examples/example1')
# os.path.dirname(os.path.abspath(__file__))

# Import the fasst package
fasst_python_dir = 'C:/Program Files/FASST_2.2.2/scripts/python'
if fasst_python_dir not in sys.path:
    sys.path.insert(0, fasst_python_dir)
import fasst

# Create temp/ and result/ directory if it does not exist
tmp_dir = 'C:/Users/User/Music/FASSTExamples/FASST_2.2.2_examples/example1/temp'#os.path.join(script_path, 'temp/');
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

results_dir = 'C:/Users/User/Music/FASSTExamples/FASST_2.2.2_examples/example1/results' #os.path.join(script_path, 'results/')
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# ------------------------------------------------------------------------
#                   Mixture and audio scene information
# ------------------------------------------------------------------------

mixture_wavname = 'C:/Users/User/Documents/GitSamples/MySamples/Current/LateralusDly.wav' #os.path.join(script_path, 'Shannon_Hurley__Sunrise__inst__mix.wav')
fid = wave.open(mixture_wavname)

# Number of sources
J = 3

# Number of channels
I = fid.getnchannels()

# Number of mixture samples
nbSamples_Mix = fid.getnframes()

# ------------------------------------------------------------------------
#               FASST Initialization (compute input xml file)
# ------------------------------------------------------------------------

print
'> FASST initialization'

# --- FASST general configuration (direct inputs for FASST)

transformType = 'STFT'  # Time-frequency transform
wlen = 1024  # Window length in samples (frame length in time domain) - should be multiple of 4 for STFT
Niteration_EM = 200  # Number of iteration of EM algorithm for sources models estimation

# --- Initialization of models and FASST specific parameters for each source

# Define necessary parameters
N = int(np.ceil(nbSamples_Mix / wlen * 2))  # Number of frames
nbin = int(wlen / 2 + 1)  # Number of frequency bins for STFT
K = 4  # NFM rank (number of spectral patterns in the dictionary)

sources = []
# Set common parameters for sources models
for j in range(J):
    source = {}

    # Name of output audio file for source j
    source['name'] = {}
    source['name'] = 'EstimatedSource_' + str(j) + '.wav'

    # Spatial model
    source['A'] = {}
    source['A']['mixingType'] = 'inst'  # Instantaneous mixture
    source['A']['adaptability'] = 'free'  # Will be adapted by FASST

    # Spectral patterns (Wex) and time activation patterns applied to spectral patterns (Hex)
    source['Wex'] = {}
    source['Wex']['adaptability'] = 'free'  # Will be adapted by FASST
    source['Wex']['data'] = 0.75 * abs(np.random.randn(nbin, K)) + 0.25 * np.ones((nbin, K))
    source['Hex'] = {}
    source['Hex']['data'] = 0.75 * abs(np.random.randn(K, N)) + 0.25 * np.ones((K, N))
    source['Hex']['adaptability'] = 'free'  # Will be adapted by FASST

    # Wiener filter parameters
    source['wiener'] = {}
    source['wiener']['a'] = 0  # a  : Over-substraction parameter (in dB) - Default value = 0
    source['wiener']['b'] = 0  # b  : Phase ponderation parameter (between [0,1]) - Default value = 0
    source['wiener'][
        'c1'] = 0  # c1 : Half-time width of the covariance smoothing window ( c1 >= 0) - Default value = 0
    source['wiener'][
        'c2'] = 0  # c2 : Half-frequency width of the covariance smoothing window ( c2 >= 0) - Default value = 0
    source['wiener']['d'] = float("-inf")  # d  : Thresholding parameter ( in dB, d <= 0) - Default value = -Inf

    sources.append(source)

# Set specific initialization parameters for sources models (spatial initialization ~ init gain balance for each source for instantaneous mixture)
sources[0]['A']['data'] = np.array(([np.sin(np.pi / 8)], [np.cos(np.pi / 8)]))
sources[1]['A']['data'] = np.array(([np.sin(np.pi / 4)], [np.cos(np.pi / 4)]))
sources[2]['A']['data'] = np.array(([np.sin(3 * np.pi / 8)], [np.cos(3 * np.pi / 8)]))

# --- Write FASST_data structure in FASST input xml file

# Define FASST_data structure
FASST_data = {}
FASST_data['tfr_type'] = transformType
FASST_data['wlen'] = wlen
FASST_data['iterations'] = Niteration_EM
FASST_data['sources'] = sources

# Write to XML
xml_fname = os.path.join(tmp_dir, 'sources.xml');
fasst.writeXML(xml_fname, FASST_data)

# ------------------------------------------------------------------------
#                        Call FASST binaries
# ------------------------------------------------------------------------
print
'> FASST execution'

print
'>> Input time-frequency representation'
fasst.compute_mixture_covariance_matrix(mixture_wavname, xml_fname, tmp_dir)

print
'>> Refinement of sources models (EM algorithm)'
fasst.estimate_source_parameters(xml_fname, tmp_dir, xml_fname + '.new')

print
'>> Computation of estimated sources'
fasst.estimate_sources(mixture_wavname, xml_fname + '.new', tmp_dir, results_dir)

# Delete temporary folder
shutil.rmtree(tmp_dir)
