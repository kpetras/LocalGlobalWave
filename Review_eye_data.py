import sys
import os

# Add the parent directory of the script (LocalGlobalWave) to the Python path
sys.path.append('/mnt/Data/LoGlo/LocalGlobalWave/LocalGlobalWave/')

from Modules.Utils import WaveData as wd
from Modules.Utils import ImportHelpers
import mne
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
from scipy.stats import f_oneway, ttest_rel

#%% Import from MNE-Data
root_dir = '//mnt/Data/DuguelabServer2/duguelab_general/DugueLab_Research/Current_Projects/KP_LGr_LoGlo/Data_and_Code/ReviewJoN/EyeData'
session2_dirs = []
savePath = root_dir
fileName = "*.asc"
fileList = glob.glob(os.path.join(root_dir, fileName), recursive=True)


results = []

for file in fileList:
    print("Loading file: " + file)
    raw = mne.io.read_raw_eyelink(file)
    annotations = raw.annotations

    # Extract participant ID (first 7 characters of the filename without the path)
    participant_id = os.path.basename(file)[:7]
    blockNr = os.path.basename(file)[-6:-4]

    # Define condition triggers (onsets)
    event_id = {'11': 11, '12': 12, '22': 22}
    events, event_id = mne.events_from_annotations(raw, event_id=event_id)

    # Create epochs based on condition onsets
    epochs = mne.Epochs(raw, events, event_id, tmin=-0.5, tmax=1.5, baseline=None, preload=True, reject=None, reject_by_annotation=False)

    # Count annotations (blinks, saccades, fixations) within each condition
    for condition, event_code in event_id.items():
        condition_epochs = epochs[event_code]
        condition_times = condition_epochs.events[:, 0] / raw.info['sfreq']  # Convert sample indices to times

        blink_count = 0
        saccade_count = 0
        fixation_count = 0

        for start_time in condition_times:
            end_time = start_time + 2  # Assuming epochs are 2 seconds long
            blinks = annotations[(annotations.description == 'BAD_blink') &
                                  (annotations.onset >= start_time) &
                                  (annotations.onset < end_time)]
            saccades = annotations[(annotations.description == 'saccade') &
                                    (annotations.onset >= start_time) &
                                    (annotations.onset < end_time)]
            fixations = annotations[(annotations.description == 'fixation') &
                                     (annotations.onset >= start_time) &
                                     (annotations.onset < end_time)]

            blink_count += len(blinks)
            saccade_count += len(saccades)
            fixation_count += len(fixations)

        # Append results for this participant and condition
        results.append({
            'Participant': participant_id,
            'Block': blockNr,
            'Condition': condition,
            'Blink Count': blink_count,
            'Saccade Count': saccade_count,
            'Fixation Count': fixation_count
        })

# Convert results to a DataFrame
df = pd.DataFrame(results)

from scipy.stats import f_oneway, ttest_rel
# Ensure numeric columns are aggregated
numeric_columns = ['Blink Count', 'Saccade Count']
averaged_df = df.groupby(['Participant', 'Condition'], as_index=False)[numeric_columns].mean()

# Perform statistical tests
for measure in numeric_columns:
    print(f"\nStatistical test for {measure}:")
    
    # Pivot the averaged data for statistical tests
    grouped = averaged_df.pivot(index='Participant', columns='Condition', values=measure)
    conditions = grouped.columns

    # Perform a repeated-measures ANOVA (e.g., one-way ANOVA)
    f_stat, p_value = f_oneway(*[grouped[cond].dropna() for cond in conditions])
    print(f"ANOVA F-statistic: {f_stat}, p-value: {p_value}")

    # Perform paired t-tests between conditions
    for i, cond1 in enumerate(conditions):
        for cond2 in conditions[i + 1:]:
            t_stat, p_val = ttest_rel(grouped[cond1].dropna(), grouped[cond2].dropna())
            print(f"Paired t-test between {cond1} and {cond2}: t-statistic = {t_stat}, p-value = {p_val}")

#plot
import seaborn as sns
import matplotlib.pyplot as plt

# Melt the averaged DataFrame for easier plotting
melted_df = averaged_df.melt(id_vars=['Participant', 'Condition'], 
                             value_vars=numeric_columns, 
                             var_name='Measure', 
                             value_name='Count')

# Create a bar plot with grouping by Measure (to compare conditions)
plt.figure(figsize=(10, 6))
sns.barplot(data=melted_df, x='Measure', y='Count', hue='Condition', ci='sd', palette='viridis')

# Add labels and title
plt.title('Comparison of Conditions for Blinks, Saccades', fontsize=14)
plt.xlabel('Measure', fontsize=12)
plt.ylabel('Average Count', fontsize=12)
plt.legend(title='Condition', fontsize=10)
plt.tight_layout()

# Show the plot
plt.show()
