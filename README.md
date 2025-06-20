# CircularInference-based_fMRI

This repository provides code for performing model-based regression analysis of fMRI data in schizophrenia patients, using the Circular Inference framework.

## Repository Structure

### behavior/

Scripts and notebooks for behavioral analysis, including simulations and model fitting:

- **simulation/** – Simulates behavior to explore how the Circular Inference model behaves under different parameters.  
- **model-free/** – Analyzes subject performance without applying the Circular Inference model.  
- **parameter_recovery/** – Tests how accurately the model can recover known parameters from simulated data.  
- **model_recovery/** – Assesses whether model fitting can distinguish between different computational models.  
- **model_fits_subjects/** – Fits the Circular Inference model to actual behavioral data to investigate top-down and bottom-up processes in schizophrenia.

### fMRI/

Scripts for fMRI analysis using outputs from the Circular Inference model:

- **model_based_regression.py** – Performs voxel-wise regression to identify brain regions encoding model-derived variables.  
- **submit_subject_level.sh** – Bash script for running `model_based_regression.py` in parallel, subject-by-subject.

### online_task/

Scripts for running the online behavioral experiment associated with this project.



data
--------------------------------------
