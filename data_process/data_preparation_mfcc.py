import os
import librosa
import math
import json

#DON'T FORGET TO OBTAIN THE DATA SET FIRST!!!
DATASET_PATH = "../datasets/raw_data/GTZAN_Dataset/genres_original"
JSON_PATH = "../datasets/processed_data/data.json"

SAMPLE_RATE = 22050
DURATION = 30 #each song is 30 seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

#num_segments divides each track into that many chunks to increase the number of input points!
#That is, for every 30 sec track, we will have 5 many 6-sec tracks 
def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    
    #dict to store data
    data ={
        "mapping": [], #["classical", "blues",...]
        "mfcc": [], # [[...],[...], ...] features for each instance
        "labels": [] #[0,1,1,...] label of each instance
    }

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length) # 11.3 -> 12

    #loop through each genre
    #dirpath = the folder we're currently in
    #dirnames = subfolders of dirpath
    #filenames = file names in dirpath
    #i is for labels: classical = 0, blues=1, rock=2 etc.
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        
        #we start at the root dirpath, so we make sure dirpath =! dataset_path
        if dirpath is not dataset_path:

            #save the name label
            dirpath_components = dirpath.split("/") # desktop/genre/blues -> ["desktop", "genre", "blues"]
            name_label = dirpath_components[-1]
            data["mapping"].append(name_label)
            print("\nProcessing {}".format(name_label))

            # process files for a specific genre

            for f in filenames: #f = 'blues.00000.wav'

                #load the audio file
                file_path = os.path.join(dirpath, f) # file_path = '/desktop/genre/blues/blues.00000.wav'
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                #process each segment by extracting mfccs and storing data

                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s
                    finish_sample = start_sample + num_samples_per_segment

                    mfcc = librosa.feature.mfcc(y=signal[start_sample:finish_sample],
                                                sr=sr,
                                                n_mfcc=n_mfcc,
                                                n_fft=n_fft,
                                                hop_length=hop_length)
                    mfcc = mfcc.T #transpose

                    #store mfcc for the segment if it has ethe expected length
                    if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i-1)
                        print("{}, segment{}".format(file_path, s))

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH)
                    




        

