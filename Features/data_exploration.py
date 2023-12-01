import pandas as pd
import numpy as np
import librosa, librosa.display
'''
librosa/__init__ requires lazy_loader module
to install:
pip install -U lazy_loader
arch users: 
yay -S python-lazy-loader
'''
import matplotlib.pyplot as plt

file = "Data/genres_original/blues/blues.00000.wav"

#waveform

'''
librosa/core/audio requires soxr module
yay -S python-soxr 
or pip install soxr
'''

signal, sr = librosa.load(file, sr=22050) # signal is 1D array with sr * duration entries -> 22050 * 30

pd.Series(signal).plot(figsize=(10,5), title='Raw Audio Example')
plt.show()

# FFT to get into frequency domain

fft = np.fft.fft(signal)

magnitude = np.abs(fft) # fft is complex valued, by taking magnitude we get real values

frequency = np.linspace(0, sr, len(magnitude))


""" 
#Plotting the frequency vs magnitude 

left_frequency = frequency[:int(len(frequency)/2)]
left_magnitude = magnitude[:int(len(magnitude)/2)]

# Above is done due to symmetry

plt.plot(left_frequency, left_magnitude)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.show()  
"""

# STFT to get spectrogram

n_fft = 2048 #number of samples for fft (window size)
hop_length = 512 #amount we're shifting to the right for each time

stft = librosa.core.stft(signal, hop_length = hop_length, n_fft = n_fft)
spectrogram = np.abs(stft)

log_spectrogram = librosa.amplitude_to_db(spectrogram)

""" librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar()
plt.show()  """


# MFCC for deep learning
MFCCs = librosa.feature.mfcc(y=signal, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
#in the mfcc function above y=signal is needed only writing signal does not work



librosa.display.specshow(MFCCs, sr=sr, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("MFCC")
plt.colorbar()
plt.show() 