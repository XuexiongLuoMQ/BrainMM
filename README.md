# BrainMM: Self-expression Property Theory Guided Multi-modal Brain Graph Learning

# Implementation
## Experimental Dataset Introduction
### Human Immunodeficiency Virus Infection (HIV)
HIV dataset is collected from the Chicago Early HIV Infection Study at Northwestern University. The clinical cohort of HIV disorder incudes fMRI and DTI imaging, and each imaging modality have 70 subjects, where 35 of which are early HIV patients and the other 35 are seronegative controls. The preprocessing includes realignment to the first volume, followed by slice timing correction, normalization, and spatial smoothness, band-pass filtering, and linear trend removal of the time series before constructing brain connection networks. Then, we use the brain atlas, i.e., automated anatomical labeling (AAL), to divide the brain into the 116 anatomical ROIs and extract a sequence of time courses from them. Finally, brain networks with 90 cerebral regions are constructed, with links representing the correlations between ROIs.
### Bipolar Disorder (BP)

### Parkinson’s Progression Markers Initiative (PPMI)

