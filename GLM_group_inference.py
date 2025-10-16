import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

from scipy.stats import ttest_rel
from statsmodels.stats.multitest import fdrcorrection

data = np.load("group_level_contrasts.npz")
deconv = data["deconv"]     # subjects × channels
standard = data["standard"] # subjects × channels
subjects = data["subjects"]

n_subj, n_ch = deconv.shape

# Create a dataframe of the contrast results for running
# a mixed effect model on
df_list = []
for ch in range(deconv.shape[1]):       # channels: 0..19
    for s_idx in range(deconv.shape[0]):  # subjects: 0..28
        df_list.append({
            "subject": subjects[s_idx],
            "channel": ch,
            "pipeline": "deconv",
            "beta": deconv[s_idx, ch]
        })
        df_list.append({
            "subject": subjects[s_idx],
            "channel": ch,
            "pipeline": "standard",
            "beta": standard[s_idx, ch]
        })
df_long = pd.DataFrame(df_list)
print(df_long.head())

# Random intercept for subject, fixed effect of pipeline
model = smf.mixedlm("beta ~ pipeline", df_long, groups=df_long["subject"])
result = model.fit()
print(result.summary())

# Iterate through each channel
for ch in range(n_ch):
    df_ch = df_long[df_long["channel"] == ch]
    model = smf.mixedlm("beta ~ pipeline", df_ch, groups=df_ch["subject"])
    result = model.fit()
    print(f"Channel {ch}:", result.summary().tables[1])

# Grab number of subject, channels
n_subj, n_ch = deconv.shape

# difference per channel
diff = deconv - standard

# mean difference
mean_diff = np.nanmean(diff, axis=0)

# standard deviation of differences
sd_diff = np.nanstd(diff, axis=0, ddof=1)

# Cohen's d for paired data
cohens_d = mean_diff / sd_diff
print("Cohen's d per channel:", cohens_d)

# SNR = mean / standard deviation per channel
snr_deconv = np.nanmean(deconv, axis=0) / np.nanstd(deconv, axis=0, ddof=1)
snr_standard = np.nanmean(standard, axis=0) / np.nanstd(standard, axis=0, ddof=1)
print("SNR deconv:", snr_deconv)
print("SNR standard:", snr_standard)

# Run a t-test on the standard vs deconvolved results
tvals, pvals = ttest_rel(deconv, standard, axis=0, nan_policy='omit')

# FDR correction across channels
reject, pvals_fdr = fdrcorrection(pvals, alpha=0.05)

print("t-values:", tvals)
print("raw p-values:", pvals)
print("FDR-corrected p-values:", pvals_fdr)

#  Create a dataframe for  saving
results = pd.DataFrame({
    "channel": np.arange(n_ch),
    "mean_deconv": np.nanmean(deconv, axis=0),
    "mean_standard": np.nanmean(standard, axis=0),
    "cohens_d": cohens_d,
    "snr_deconv": snr_deconv,
    "snr_standard": snr_standard,
    "tval": tvals,
    "pval": pvals,
    "pval_fdr": pvals_fdr,
    "significant": reject
})

# Save to a csv
results.to_csv("t_test_results.csv")

# Print out results
print(results)

# Create a figure for displaying SNR
plt.figure(figsize=(12,5))
plt.bar(np.arange(n_ch)-0.2, snr_deconv, width=0.4, label='deconv')
plt.bar(np.arange(n_ch)+0.2, snr_standard, width=0.4, label='standard')
plt.xlabel('Channel')
plt.ylabel('SNR')
plt.legend()
plt.show()