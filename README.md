# BrainMM: Self-expression Property Theory Guided Multi-modal Brain Graph Learning

# Implementation
## Experimental Dataset Introduction
### Human Immunodeficiency Virus Infection (HIV)
HIV dataset is collected from Early HIV Infection Study in the University of Chicago. The clinical cohort of HIV disorder incudes fMRI and DTI imaging, and each imaging modality have 70 subjects, where 35 of which are early HIV patients and the other 35 are seronegative controls for the sake of exposition. The demographic characteristics of these two sets of participants, such as age, gender, racial makeup, and educational level, are identical. The preprocessing includes realignment to the first volume, followed by slice timing correction, normalization, and spatial smoothness, band-pass filtering, and linear trend removal of the time series before constructing brain connection networks. We preprocess the fMRI data using the DPARSF1 toolbox. We focus on 116 anatomical volumes of interest (AVOI), where a sequence of responds is extracted from them. The functional brain networks are constructed with 90 cerebral regions where each node represents a brain region and the edges are calculated as the pairwise correlation. For the DTI data, we use the FSL toolbox2 for the preprocessing. Each subject is parcellated into 90 regions via the propagation of the automated
anatomical labeling (AAL). 
### Bipolar Disorder (BP)
BP dataset is collected from 52 bipolar I individuals and 45 healthy controls at the University of Chicago, including both fMRI and DTI modalities. The resting-state fMRI and DTI data were acquired on a Siemens 3T Trio scanner using a T2 echo planar imaging (EPI) gradient-echo pulse sequence with integrated parallel acquisition technique (IPAT). For the fMRI data, the brain networks are constructed using the toolbox CONN3, where pairwise BOLD signal correlations are calculated between the 82 labeled Freesurfer-generated cortical/subcortical gray matter regions. For DTI, same as fMRI, we constructed the DTI image into 82 regions.
### Parkinson’s Progression Markers Initiative (PPMI)
This is a restrictively public available dataset to speed breakthroughs and support validation on Parkinson’s Progression research. In PPMI dataset, we consider a total of 718 subjects, where 569 subjects are Parkinson’s disease patients and 149 are healthy controls. We preprocess the raw imaging using the FSL and ANT. 84 ROIs are parcellated from T1-weighted structural MRI using Freesurfer. Based on these 84 ROIs, we construct three views of brain networks using three different whole brain tractography algorithms, namely the Probabilistic Index of Connectivity (PICo), Hough voting (Hough), and FSL. Please refer to Zhan et al. [1] for the detailed brain tractography algorithms.
[1] Liang Zhan, Jiayu Zhou, Yalin Wang, Yan Jin, Neda Jahanshad, Gautam Prasad, Talia M Nir, Cassandra D Leonardo, Jieping Ye, Paul M Thompson, et al. 2015. Comparison of nine tractography algorithms for detecting abnormal structural brain networks in Alzheimer’s disease. Front. Aging Neurosci.
# Running
## Requirement
The framework needs the following dependencies:
```
torch~=1.10.2
numpy~=1.22.2
scikit-learn~=1.0.2
scipy~=1.7.3
pandas~=1.4.1
tqdm~=4.62.3
torch-geometric~=2.0.3
torch-cluster 1.5.9
faiss-cpu 1.7.2
```
## Run
To run our model on any of the datasets in our paper, simply run:
```
python main.py data/<dataset-path>
```
Please place the dataset files in the `data/` folder under the root folder.
All parameter settings are stored in config.py.
