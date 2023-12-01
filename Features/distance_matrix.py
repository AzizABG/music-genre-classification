import numpy as np
import pandas as pd
import librosa
import math
import os

all_raw_data = []
dataset_path = "../datasets/raw_data/GTZAN_Dataset/genres_original"
for dirpath, dirnames, filenames in os.walk(dataset_path):
    if dirpath is not dataset_path:
        for f in filenames: #f = 'blues.00000.wav'
            file_path = os.path.join(dirpath, f) # file_path = '/desktop/genre/blues/blues.00000.wav'
            signal, sr = librosa.load(file_path, sr=22050)
            all_raw_data.append(signal)
            
        


#def normalize(X):
#    X_min = np.min(X)
#    X_max = np.max(X)
#    X_norm = ((X - X_min) / (X_max - X_min))*2 -1
#    return X_norm

def intrinsic_distances(arr):
# we use l_infinity instead of euclidean as we have sample rate 22050

    pairwise_distances = [0]
    for i in range(len(arr)-1):
        pairwise_distances.append(abs(arr[i]-arr[i+1]))
    
    partial = 0
    sum = []
    for num in pairwise_distances:
        partial += num
        sum.append(partial)
    
    return sum

minim = 700000
cumulative_distances = []
for i in range(len(all_raw_data)):
    print('computing distances for:{}'.format(i))
    cumulative_distances.append(np.array(intrinsic_distances(all_raw_data[i])))
    
    if (len(cumulative_distances[-1]) <= minim):
        minim = len(cumulative_distances[-1])

    print('lenght = {} and min ={}'.format(len(cumulative_distances[-1]), minim))







def wass(v1,v2):
    v1 = v1[:minim]
    v2 = v2[:minim]
    diff = v1 - v2
    sqr = np.square(diff)
    sum = np.sum(sqr) / len(v1)
    result = np.sqrt(sum)
    return result

dist = np.zeros((990,990))

for i in range(990):
    for j in range(i+1, 990):
        print('computing the distance bw:{} and {}'.format(i,j))
        dist[i][j] = wass(cumulative_distances[i], cumulative_distances[j])

np.save('../datasets/processed_data/distance_matrix', dist)