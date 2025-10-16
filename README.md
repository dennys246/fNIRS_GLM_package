

----------------- Laboratory for Child Brain Development --------------
-------------------- Washing University in St. Louis ----------------

                        - GLM Analysis Package -

Within this script package you will find a series of scripts used to conduct a 2-stage GLM analysis on a Flanker task. The scripts are geared towards analyzing hemoglobin vs. neural activity, estimated using the HRfunc tool, recording in fNIRS data collected apart of the P-CAT study. The scripts could easily be geared towards other analysis by modifying the different GLM analysis (i.e. only use hemoglobin or neural activity) and modifying what data is being used.

The package contains 4 primary scripts each with their own specific purpose.

1. fNIRS_pipeline.py

This is an example script of how you can prepare you're data for fNIRS GLM analysis. Apart of this script is an example preprocessing script, alongside different functions for loading fNIRS data in both sNIRF file and NIRX folder fNIRS formats. Ultimately, you'll just need fNIRS data loaded into MNE objects.

2. Flanker_labeler.py

An example script of how you can prepare .evt files for the GLM analysis. Most likely if you are analyzing a different task, you will need to modify how the behavioral files are being loaded and read. Ultimately the goal is to use modified .evt files with behavioral data stored within, while preserving the original .evt file with the new post-fix _old.evt.

3. GLM_2-stage.py

Within this script we analyzed the difference between hemoglobin and neural activity data by looking at the differences of two GLM's conducted on the same Flanker task in either modality.

4. GLM_group_inference.py

Conduct a mixed effects model and t-test to assess the significant between both GLMs conducted in the GLM_2-stage.py script. The script will print out the results to your terminal and save them to a csv.

5. GLM_visualization.py

This script shows how you can take contrasts outputted by GLM's and visualize them in MNI space using surface meshes. This script visualizes the output of both the hemoglobin and neural activity GLM contrasts. Note you must have coregistered the same subjects fNIRS to an MRI anatomical scan they completed and generated a transformation.fif matrix for this to work.


____________________________________________________________________________
