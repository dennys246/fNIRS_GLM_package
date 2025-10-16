
# Import dependencies
import mne, os, nilearn

import numpy as np
import pandas as pd
import fNIRS_pipeline as pipeline

from glob import glob

from mne_nirs.statistics import run_glm

"""
This python script is used to run a generic GLM analysis on Flanker 
fNIRS data collected apart of the P-CAT study to assess the impact of
neural activity deconvolution on analysis outcomes. This script was 
originally used to compare the GLM analysis outcomes of a Flanker
task between traditionally preprocessed hemoglobin and deconvolved
neural activity.
"""

# Define runtime variables for analzying data

# Directory deconvolved fNIRS data is stored
data_dir = '/path/to/nirs/data/'

# fNIRS sampling frequency
sfreq = 7.81 

# Duration of events
event_duration = 3.0 

# Note subjects to exclude assessed from QCing
excluded = []

# Load in our data through our pipeline script
# Note this passes back string subject IDs, raw fNIRS scans,
# traditionally preprocessed data, and scan impulse event series
print(f"Loading data...")
subject_ids, raw_scans, preproc_scans, scan_events = pipeline.load_pcat(data_dir)

# Create hold variable for subject level contrasts
standard_scan = None
deconv_contrasts = []
standard_contrasts = []

# Define function for finding channels
def get_channels_with_positions(info):
    """Return indices of channels with valid position information."""
    return [idx for idx, ch in enumerate(info['chs']) if np.any(ch['loc'][:3])]

# Iterate through each subject
for ind, subject_id in enumerate(subject_ids):
    if str(subject_id) in excluded:
        print(f"Subject {subject_id} excluded, skipping...")
        continue

    print(f"Calculating subject {subject_id} congruent-incongruent contrasts")
    # Generate filename where deconvolved file is stored
    deconvolved_filename = f"{data_dir}{subject_id}/{subject_id}_Flanker/{subject_id}_Flanker_Deconvolved.fif"
    
    # Check if is exists
    if os.path.exists(deconvolved_filename) == False: 
        print(f"Missing deconvolved file for {subject_id}, skipping...")
        continue

    # Read in subjects deconv scan
    deconv_scan = mne.io.read_raw_fif(deconvolved_filename)

    # Grab the subject non-deconv scan
    preproc_scan = preproc_scans[ind]
    if standard_scan is None:
        standard_scan = preproc_scan.copy()

    # Grab subject events for just congruent vs. incongruent
    events, event_id = pipeline.process_congruency(scan_events[ind])
    print(f"Events...\n{events}")

    # Format events for design matrix
    pandas_events = pd.DataFrame([[event[0], event_duration, event[2], 1] for event in events], columns = ['onset', 'duration', 'trial_type', 'modulation'])
    frame_times = np.array([sample / sfreq for sample in range(deconv_scan.n_times)])
    print(pandas_events)

    # Create first design matrix
    deconv_design_matrix = nilearn.glm.first_level.make_first_level_design_matrix(
        frame_times,
        pandas_events, 
        hrf_model = None, 
        drift_model = 'cosine', 
        high_pass = 0.01, 
        drift_order = 1)
    
    # Create second design matrix
    standard_design_matrix = nilearn.glm.first_level.make_first_level_design_matrix(
        frame_times,
        pandas_events, 
        hrf_model = 'spm', 
        #hrf_model = "spm + derivative + dispersion", 
        drift_model = 'cosine', 
        high_pass = 0.01, 
        drift_order = 1)

    # For both first and second GLM analysis - 
    for scan, design_matrix, contrast_store, preprocessing in zip([deconv_scan, preproc_scan], [deconv_design_matrix, standard_design_matrix], [deconv_contrasts, standard_contrasts], ['Deconvolved', 'Standard']):
        # Define variable to store data in
        individual_contrasts = []

        # Calculcate subject congruency-incongruency contrast
        glm_results = run_glm(scan, design_matrix)

        # Create you're contrast vector
        # NOTE: Most likely you'll need to change this to you're particular design
        contrast_vec = np.array([1, -1] + [0] * (len(design_matrix.columns) - 2))

        # Compute contrasts from GLM
        contrasts_obj = glm_results.compute_contrast(contrast_vec, 'F')

        # Grab results
        contrasts = contrasts_obj.data
        contrast_effect = contrasts.effect
        contrast_variance = contrasts.variance

        print(f"Contrast effects: {contrast_effect}")
        for channel_contrast in contrast_effect:  # Each channel
            print(channel_contrast)
            individual_contrasts.append(channel_contrast)

        # Figure out channel length
        n_channels = len(raw_scans[ind].info['chs'])

        # Create a numpy array for storing results
        subject_contrast = np.full(n_channels, np.nan)

        # Grab channels
        valid_idxs = get_channels_with_positions(deconv_scan.info) 

        # Store outcome of results
        subject_contrast[valid_idxs] = individual_contrasts  # fill in where you have data
        
        # Print out outcomes
        print(f"Subject {subject_id}: number of channel contrasts = {len(individual_contrasts)}")
        print(f"Expected number of valid channels = {len(valid_idxs)}")
        print(f"Valid channel indices for {subject_id}: {valid_idxs}")

        # Store the contrasts
        contrast_store.append(subject_contrast)

# Stack results into a numpy array
deconv_contrasts = np.vstack(deconv_contrasts)
standard_contrasts = np.vstack(standard_contrasts)

# Save a numpy compressed file of contrasts
np.savez_compressed(
    "group_level_contrasts.npz",
    deconv=deconv_contrasts,
    standard=standard_contrasts,
    subjects=np.array(subject_ids)
)

