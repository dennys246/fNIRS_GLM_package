
# Import dependencies
import mne
from glob import glob

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from nibabel.affines import apply_affine
from nilearn import plotting, datasets, surface
from mne.transforms import invert_transform
from nilearn import plotting, surface, datasets
from scipy.ndimage import gaussian_filter
from nibabel.affines import apply_affine



"""
 --- Create 3D scalp visualization using standard montage ---
 This requires a transformation.fif generated through FreeSurfer
 through a specific subjects fsaverage data generated from 
 anatomical-BOLD coregistration.

 NOTE: If your visualization comes out strange, considering
 using a different subject to generate these transformation files
"""

# Directory to store plots
plot_dir = "/path/to/save/plots/"

# load results
data = np.load("group_level_contrasts.npz", allow_pickle=True)
deconv_contrasts = data["deconv"]    # shape: (n_subjects, n_all_channels)
standard_contrasts = data["standard"]
subjects = data["subjects"]



preproc_scan = mne.io.read_raw_nirx("path/to/a/nirx/scan/")
# get list of channel names from a canonical raw/info (choose one subject you used)
ref_info = preproc_scan.info   # make sure preproc_scans exists in this scope
chan_names = [ch['ch_name'] for ch in ref_info['chs']]

# compute mean across subjects (use nanmean in case some subjects have missing channels)
deconv_mean = np.nanmean(deconv_contrasts, axis=0)
standard_mean = np.nanmean(standard_contrasts, axis=0)

# now create a vector of values ordered by the same channel list as info picks
picks = mne.pick_types(ref_info, fnirs=True)
picked_names = [ref_info['ch_names'][p] for p in picks]

# build index mapping from ref_info to desired order in your arrays (if necessary)
# If your contrast arrays are already in the same channel order as ref_info, you can index directly:
contrast_values_standard = standard_mean[picks]
contrast_values_deconv = deconv_mean[picks]

# Format contrast for deconvolved data
deconv_mean = np.mean(deconv_contrasts, axis=0)
deconv_mean *= 100 # Convert to percent
deconv_abs = max(abs(np.min(deconv_contrasts)), abs(np.max(deconv_contrasts)))
deconv_vlim = (-deconv_abs, deconv_abs)

# Format contrast from standard preprocessed data
standard_mean = np.mean(standard_contrasts, axis=0)
standard_mean *= 100 # Convert to percent
standard_abs = max(abs(np.min(standard_contrasts)), abs(np.max(standard_contrasts)))
standard_vlim = (-standard_abs, standard_abs)
    
# Calculate max value in contrast for plotting
combined = np.concatenate([standard_mean, deconv_mean])
shared_min = np.min(combined)
shared_max = np.max(combined)
shared_abs = max(abs(shared_min), abs(shared_max))
_vlim = (-shared_abs, shared_abs)

# Directory transformation.fif files was created
subjects_dir = '/freesurfer/subjects'

# Build montage, you may need to switch the montage depending on
# subject population
montage = mne.channels.make_standard_montage('artinis-brite23')

# Set standard montage (e.g., 10-20) on one subject’s Raw object
standard_raw = preproc_scan.copy()
standard_raw.set_montage(montage, on_missing='warn', match_case=False)

# Confirm picks are aligned
picks = mne.pick_types(preproc_scan.info, fnirs=True)
info_plot = mne.pick_info(preproc_scan.info, picks)

# Extract channel positions from info_plot (should match picks length)
ch_positions = np.array([ch['loc'][:3] for ch in info_plot['chs']])

# Make sure contrast_values has correct shape and order
contrast_values = standard_mean[picks]  # standard_mean is 1D with length == n_channels

# positions in meters (MNE stores loc in meters)
ch_positions = np.array([ch['loc'][:3] for ch in info_plot['chs']])  # already in meters

# Output what we've seen so far
print(f"Positions shape: {ch_positions.shape}, contrast shape: {contrast_values.shape}")
print("ch_positions:", ch_positions)
print("contrast_values shape:", contrast_values.shape)
print("contrast min/max:", np.min(contrast_values), np.max(contrast_values))

# Check no NaNs
assert not np.isnan(contrast_values).any()

# Plot alignment with montage and transform
plotter = mne.viz.plot_alignment(
    info=preproc_scan.info,
    trans='sub-001-trans.fif',
    subject='sub-001',
    subjects_dir=subjects_dir,
    coord_frame='head',
    surfaces=['head'],
    show_axes=True,
)

# Add points with contrasts as scalars
plotter.plotter.add_points(
    points=ch_positions,
    scalars=contrast_values,
    cmap='RdBu_r',
    clim=(contrast_values.min(), contrast_values.max()),  # or _vlim if valid
    point_size=20,
    render_points_as_spheres=True
)

# Add title and save plot
plotter.plotter.add_scalar_bar(title='GLM Contrast')
plotter.plotter.screenshot(f"{plot_dir}deconv_contrast_3D_standard_scalp.jpg")
plotter.plotter.close()


# Create info for plot
#picks = mne.pick_types(preproc_scan.info, fnirs='hbo')
picks = [0, 2, 4,  6,  8, 10, 12, 14, 16, 18]
info_for_plot = mne.pick_info(preproc_scan.info, sel=picks)
#for ch in info_for_plot['chs']:
#    ch['loc'][1] -= 0.02  # shift back by 1 unit on the y-axis
print(picks)

#Generate plot
fig, _axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot a topo map of deconvolved GLM contrasts using mask
deconv_image, deconv_cn = mne.viz.plot_topomap(
    deconv_mean[picks],
    info_for_plot,
    axes = _axes[0, 0],
    show = False,
    cmap = 'RdBu_r',
    vlim = deconv_vlim,
    contours = 0,
    extrapolate = 'local'
)
_axes[0, 0].set_title(f"Deconvolved GLM Contrast")

# Plot a topo map of traditional GLM contrast using mask 
standard_image, standard_cn = mne.viz.plot_topomap(
    standard_mean[picks],
    info_for_plot,
    axes = _axes[0, 1],
    show = False,
    cmap = 'RdBu_r',
    vlim = standard_vlim,
    contours = 0,
    extrapolate = 'local'
)
_axes[0, 1].set_title(f"Standard GLM Contrast with Glover HRF")

# Normalize output
norm = plt.Normalize(vmin = -shared_abs, vmax = shared_abs)

# Plot channel positions for visualization troubleshooting
for i, ch in enumerate(info_for_plot['chs']):
    print(f"{ch['ch_name']} position: {ch['loc'][:3]}")

# Get positions of channels and colors
positions = np.array([ch['loc'][:2] for ch in info_for_plot['chs']])  # x, y
colors = cm.RdBu_r(norm(deconv_contrasts[0][picks]))[:, :3]  # RGB, no alpha

# Create a scatter plot for the deconvolved GLM results
deconv_sc = _axes[1, 0].scatter(positions[:, 0], 
                positions[:, 1],
                c = deconv_mean[picks], 
                cmap = 'RdBu_r',
                vmin = -deconv_abs, 
                vmax = deconv_abs,
                s = 100, 
                edgecolors = 'k')
_axes[1, 0].axis('equal')
_axes[1, 0].grid(True)

# Create a scatter plot for the standard GLM results
standard_sc = _axes[1, 1].scatter(positions[:, 0], 
                positions[:, 1],
                c = standard_mean[picks], 
                cmap = 'RdBu_r',
                vmin = -standard_abs, 
                vmax = standard_abs,
                s = 100, 
                edgecolors = 'k')
_axes[1, 1].axis('equal')
_axes[1, 1].grid(True)

#Normalize plots
norm = plt.Normalize(vmin=-shared_abs, vmax=shared_abs)
sm = cm.ScalarMappable(cmap='RdBu_r', norm=norm)
sm.set_array([])  # Needed for matplotlib < 3.1

deconv_norm = plt.Normalize(vmin=-deconv_abs, vmax=deconv_abs)
deconv_sm = cm.ScalarMappable(cmap='RdBu_r', norm=norm)
deconv_sm.set_array([])  # Needed for matplotlib < 3.1

standard_norm = plt.Normalize(vmin=-standard_abs, vmax=standard_abs)
standard_sm = cm.ScalarMappable(cmap='RdBu_r', norm=norm)
standard_sm.set_array([])  # Needed for matplotlib < 3.1

# Add a colorbar to the figures
deconv_cbar = fig.colorbar(deconv_sm, 
                ax = _axes[1, 0], 
                orientation = 'vertical', 
                shrink = 0.6, 
                label = 'Contrast Value')

standard_cbar = fig.colorbar(standard_sm, 
                ax = _axes[1, 1], 
                orientation = 'vertical', 
                shrink = 0.6, 
                label = 'Contrast Value')

# Pick fNIRS channels
picks = mne.pick_types(preproc_scan.info, fnirs=True)  # returns indices

# Create new info object with only those channels
info_plot = mne.pick_info(preproc_scan.info, picks)

# Get their locations in head space
ch_locs = np.array([ch['loc'][:3] for ch in info_plot['chs']])

# Load the transform file (head -> MRI)
trans = mne.read_trans('lcbd-coreg-trans.fif')

#  Invert it to get MRI -> head 
mri_head_t = invert_transform(trans)

# Convert to MNI space
subjects_dir = '/freesurfer/subjects'
ch_mni = mne.head_to_mni(
    ch_locs, 
    subject='sub-001',
    mri_head_t=mri_head_t,
    subjects_dir=subjects_dir
)

# Define MNI 2mm volume dimensions and affine
shape = (91, 109, 91)
affine = np.array([
    [2, 0, 0, -90],
    [0, 2, 0, -126],
    [0, 0, 2, -72],
    [0, 0, 0, 1]
])

# Make sure contrast_values has correct shape and order
contrast_values = deconv_mean[picks]  # standard_mean is 1D with length == n_channels

# Now convert to voxel coordinates using the MNI affine
vox_coords = np.round(apply_affine(np.linalg.inv(affine), ch_mni)).astype(int)
assert np.all((vox_coords >= 0) & (vox_coords < np.array(shape)))
print(f"Voxel coordinates: {vox_coords}")
# Create empty 3D volume and insert contrast values
volume_data = np.zeros(shape)
counts = np.zeros_like(volume_data)
for (x, y, z), val in zip(vox_coords, contrast_values):
    volume_data[x, y, z] += val
    counts[x, y, z] += 1
    print(volume_data[x, y, z])

# Avoid divide-by-zero
nonzero = counts > 0
volume_data[nonzero] /= counts[nonzero]

# Create NIfTI image
deconv_image = nib.Nifti1Image(volume_data, affine)

fsaverage = datasets.fetch_surf_fsaverage()  # ~140MB

base_fsaverge = fsaverage.white_right
combo_fsaverage = fsaverage.sulc_right

# Sample onto right hemisphere surface
texture = surface.vol_to_surf(deconv_image, base_fsaverge)
_vmax = np.max(np.abs(texture))

# Calculate a volume to surface texture for plotting
texture = surface.vol_to_surf(
    deconv_image,
    base_fsaverge,
    radius = 5.0,  # mm
    interpolation = 'linear'
)

# (Optional) Smooth the volume with a gaussian filter
smoothed_volume = gaussian_filter(volume_data, sigma = 2)  # adjust sigma

# Create an imagve from the contrasts using computed affine
deconv_image = nib.Nifti1Image(smoothed_volume, affine)
texture = surface.vol_to_surf(deconv_image, base_fsaverge)

print(f"Conrasts sum: {smoothed_volume}")

# Plot surface stat map
plotting.plot_surf_stat_map(
    base_fsaverge,  # surface mesh
    texture, # surface-projected data
    hemi = 'right',
    view = [0, 135], # [0, 45] for right and [0, 135] for left
    colorbar = True,
    cmap = 'RdBu_r',
    threshold = 0.001,  # or set threshold to show only strong contrasts
    bg_map = combo_fsaverage,
    bg_on_data = True,
    vmin = -0.01,
    vmax = 0.01,
    title = "fNIRS GLM Contrast - Left Hemisphere",
    output_file = "plots/fnirs_glm_contrast_left.png"
)

# Save contrast plotting space for later use
nib.save(deconv_image, "fnirs_contrast_volume.nii.gz")

# Create all metadata for MNI projection
scan_filename = "/path/to/scan.fif"
montage = mne.channels.read_dig_fif(scan_filename)

# Compute transformation matrix from fNIRS montage coordinates to MNI space
trans = mne.channels.compute_native_head_t(montage)

# Grab subject mni coordinates using same fsaverage as transformation.fif
mni_coordinates = mne.head_to_mni(
    pos = np.array(channel['loc'][:3] for channel in info_for_plot['chs']),
    subject = 'fsaverage', # Name
    mri_head_t = trans, # Loaded transformation matrix generated from fNIRS
    subjects_dir = '/path/to/fsaverage')

# Assess the coordinates
shape = (91, 109, 91)
affine = np.array([
    [ 2,  0,  0, -90],
    [ 0,  2,  0, -126],
    [ 0,  0,  2, -72],
    [ 0,  0,  0,   1]
])

#Calculate voxel coordinates
vox_coords = np.round(apply_affine(np.linalg.inv(affine), mni_coordinates)).astype(int)

# Translate deconv contrasts to MNI space
deconv_data = np.zeros(shape)
for (x, y, z), val in zip(vox_coords, deconv_contrasts):
    if (0 <= x < shape[0]) and (0 <= y < shape[1]) and (0 <= z < shape[2]):
        deconv_data[x, y, z] = val

# Build image
deconv_image = nib.Nifti1Image(deconv_data, affine) 
deconv_header = deconv_image.header

# Save output
nib.save(deconv_image, f'{plot_dir}fnirs_deconv_contrast_map.nii.gz')

# Translate standard contrasts to MNI space
standard_data = np.zeros(shape)
for (x, y, z), val in zip(vox_coords, standard_contrasts):
    if (0 <= x < shape[0]) and (0 <= y < shape[1]) and (0 <= z < shape[2]):
        standard_data[x, y, z] = val

# Create a nifti image from the standard GLM outcomes volume
standard_image = nib.Nifti1Image(standard_data, affine) # Build image
standard_header = standard_image.header

# Save the contrasts cast to MNI space
nib.save(standard_image, f'{plot_dir}fnirs_standard_contrast_map.nii.gz')

# Define thresholds andc intensity of surface mesh map
_threshold = 0
intensity = 0.5

# Grab from datasets the surface fsaverage database for plotting
fsaverage = datasets.fetch_surf_fsaverage()

# Create a plot 
fig, _axes = plt.subplots(2, 1, figsize=(10, 8))

# Create a surface mesh for visualizing deconv results and plots
deconv_texture = surface.vol_to_surf(deconv_image, fsaverage.pial_left)
plotting.plot_surf_stat_map(surf_map = fsaverage.infl_left,
                            stat_map = deconv_texture, 
                            hemi = 'left', 
                            view = 'anterior', 
                            title = "Deconvolved GLM Contrasts", 
                            colorbar = False, 
                            threshold = _threshold, 
                            bg_on_data = True, 
                            cmap='Spectral', 
                            axes = _axes[0],
                            vmax = shared_abs, 
                            vmin = -shared_abs)
_axes[0].set_title(f"Deconvolved GLM Contrast")

# Create a surface to mesh texture for visualizing results and plot
standard_texture = surface.vol_to_surf(standard_image, fsaverage.pial_left)
plotting.plot_surf_stat_map(surf_map = fsaverage.infl_left, 
                            stat_map = standard_texture, 
                            hemi = 'left', 
                            view = 'anterior', 
                            title = "Standard GLM Contrasts", 
                            colorbar = True, 
                            threshold = _threshold, 
                            bg_on_data = True, 
                            cmap='Spectral',
                            axes = _axes[1],
                            vmax = shared_abs, 
                            vmin = -shared_abs)
_axes[1].set_title(f"Standard GLM Contrast with Glover HRF")

# Add overall title
fig.suptitle("P-CAT Flanker Congruent-Incongruent Contrast", fontsize = 14)

# Add shared colorbar for scatter plots
_axes[1, 1].set_xlabel('Contrast (% signal change)')

# Layout and save
plt.tight_layout()
plt.savefig(f"{plot_dir}P-CAT_combined_congruent-incongruent_contrasts.jpg")
plt.close(fig)

# Plot data
plotter = mne.viz.plot_alignment(
    info = info_for_plot,
    surfaces = [],  # or add 'head' if you want a background
    coord_frame = 'head',
    show_axes = True,
)

# Add colored dots at channel locations
plotter.plotter.add_points(
    points=positions,
    scalars=contrast_for_plot,
    cmap = 'RdBu_r',
    point_size = 15,
    render_points_as_spheres = True
)

# Add title with scalar bar
plotter.plotter.add_scalar_bar(title = "Contrast")

# Save and close 
plotter.plotter.screenshot(f"{plot_dir}P-CAT__{preprocessing.lower()}_contrast_colored_dots_3D.jpg")
plotter.plotter.close() 
