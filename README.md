# music-genre-classification

## Overview
Our project started with the initial idea to use a variation of the Gromov-Wasserstein (GW) distance that is designed for time series (introduced in the paper: [Scalable Gromov–Wasserstein Based Comparison of Biological Time Series](https://link.springer.com/article/10.1007/s11538-023-01175-y)) in order to classify different music genres. Our goal was to leverage this distance notion to develop an interpretable model for music genre classification.

We implemented the algorithm from the paper in order to compute this GW distance. We used the distance matrix with k-nn algorithm to train our model. Moreover, we used other notions of distances such as Mahalanobis distance and L1 distance. We also trained a k-nn with the distance matrices we obtained from Mahalanobis distance and L1 distance. To have a baseline model, we trained two neural networks: CNN and RNN.

## GTZAN data set. 
This data set has different songs in 10 different genres. They are 
- blues 
- classical 
- country 
- disco
- hiphop 
- jazz
- metal 
- pop
- reggae 
- rock 

## Methods 
- 



---


## Notebooks and Python files

### 1. `Data_exploration.ipynb`
- **Description**: This notebook is dedicated to the initial exploration of the audio dataset. It includes data visualization, basic statistics, and preliminary data processing steps.

### 2. `Feature_selection_data_exploration.ipynb`
- **Description**: Focuses on selecting the most relevant features from the dataset. Techniques like correlation analysis and principal component analysis (PCA) are used here.

### 3. `CNN_all_features_3_sec.ipynb` and `CNN_all_features_30_sec.ipynb`
- **Description**: These notebooks implement Convolutional Neural Networks (CNN) for audio classification. The former analyzes 3-second audio clips, while the latter deals with 30-second clips.

### 4. `KNN_all_features_Lp_metric_3_sec.ipynb` and `KNN_all_features_Lp_metric_30_sec.ipynb`
- **Description**: K-Nearest Neighbors (KNN) models are implemented in these notebooks using L1 or Euclidean distance. They differ in the length of audio clips analyzed.

### 5. `KNN_GW_Metric_MFCCs.ipynb` and `KNN_Mahalanobis_Metric_MFCCs.ipynb`
- **Description**: These notebooks explore KNN models with different metrics (Gromov-Wasserstein and Mahalanobis) focusing on Mel-frequency cepstral coefficients (MFCCs) as features.

## Installation

- Python 3.x
- Libraries: numpy, pandas, matplotlib, scikit-learn, tensorflow, librosa (Install using `pip install numpy pandas matplotlib scikit-learn tensorflow librosa`)

## Usage

1. Start by running the `Data_exploration.ipynb` to understand the dataset.
2. Proceed to `Feature_selection_data_exploration.ipynb` for feature analysis and selection.
3. Each of the remaining notebooks and Python files can be run independently based on the model and data segment length of interest.

## Additional Information

- **Data Source**: [GTZAN Dataset - Music Genre Classification](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
<!-- - **Limitations**: [Mention any limitations or considerations in the data or methods used]
- **Contact**: [Your contact information] -->

---

<!-- ## Obtaining GTNAZ data set 
You can obtain the dataset from the following Kaggle link:
[GTZAN Dataset - Music Genre Classification](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) -->
