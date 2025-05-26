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
from collections import defaultdict

#%% Import from MNE-Data
root_dir = '//mnt/Data/DuguelabServer2/duguelab_general/DugueLab_Research/Current_Projects/KP_LGr_LoGlo/Data_and_Code/ReviewJoN/EyeData'
session2_dirs = []
savePath = root_dir
fileName = "*.asc"
fileList = glob.glob(os.path.join(root_dir, fileName), recursive=True)

subject_files = defaultdict(list)
for file in fileList:
    # Extract subject name (first 8 chars of basename)
    subname = os.path.basename(file)[:8]
    subject_files[subname].append(file)

# For each subject, concatenate all epochs across blocks
subject_epochs = {}
event_id = {'11': 11, '12': 12, '22': 22}
results = []

for subname, files in subject_files.items():
    # Initialize counters for each condition
    condition_counts = {
        '11': {'Blink Count': 0, 'Saccade Count': 0, 'Fixation Count': 0},
        '12': {'Blink Count': 0, 'Saccade Count': 0, 'Fixation Count': 0},
        '22': {'Blink Count': 0, 'Saccade Count': 0, 'Fixation Count': 0}
    }
    
    epochs_list = []
    for file in sorted(files):  # sort to keep block order
        raw = mne.io.read_raw_eyelink(file)        
        annotations = raw.annotations        
        events, _ = mne.events_from_annotations(raw, event_id=event_id)
        epochs = mne.Epochs(raw, events, event_id, tmin=0, tmax=2, baseline=None, preload=True, reject=None, reject_by_annotation=False)
        epochs_list.append(epochs)

        # Count annotations (blinks, saccades, fixations) within each condition
        for condition, event_code in event_id.items():
            condition_epochs = epochs[event_code]
            condition_times = condition_epochs.events[:, 0] / raw.info['sfreq']  # Convert sample indices to times

            for start_time in condition_times:
                end_time = start_time + 2  # Assuming epochs are 2 seconds long
                blink_count = len(annotations[(annotations.description == 'BAD_blink') &
                                            (annotations.onset >= start_time) &
                                            (annotations.onset < end_time)])
                saccade_count = len(annotations[(annotations.description == 'saccade') &
                                              (annotations.onset >= start_time) &
                                              (annotations.onset < end_time)])
                fixation_count = len(annotations[(annotations.description == 'fixation') &
                                               (annotations.onset >= start_time) &
                                               (annotations.onset < end_time)])

                condition_counts[condition]['Blink Count'] += blink_count
                condition_counts[condition]['Saccade Count'] += saccade_count
                condition_counts[condition]['Fixation Count'] += fixation_count

    # After processing all files for this subject, append the aggregated results
    for condition, counts in condition_counts.items():
        results.append({
            'Participant': subname,
            'Condition': condition,
            'Blink Count': counts['Blink Count'],
            'Saccade Count': counts['Saccade Count'],
            'Fixation Count': counts['Fixation Count']
        })

# Convert results to a DataFrame
df = pd.DataFrame(results)
# Save the DataFrame to a CSV file
df.to_csv(os.path.join(savePath, 'eye_data_summary.csv'), index=False)

#%% Load and plot
from scipy.stats import f_oneway, ttest_rel
import seaborn as sns

df = pd.read_csv(os.path.join(savePath, 'eye_data_summary.csv'))
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
sns.barplot(data=melted_df, x='Measure', y='Count', hue='Condition', errorbar='sd', palette='viridis')

# Add labels and title
plt.title('Comparison of Conditions for Blinks, Saccades', fontsize=14)
plt.xlabel('Measure', fontsize=12)
plt.ylabel('Average Count', fontsize=12)
plt.legend(title='Condition', fontsize=10)
plt.tight_layout()

# Show the plot
plt.show()

##% Check against raw EEG
from Modules.Utils import WaveData as wd
from Modules.Utils import ImportHelpers
import mne
import numpy as np
import os
import pandas as pd

#%% Import from MNE-Data
root_dir = '/mnt/Data/DuguelabServer2/duguelab_general/DugueLab_Research/Current_Projects/LGr_GM_JW_DH_LD_WavesModel/Experiments/Data/data_MEEG/raw/'
session2_dirs = []
savePath = '/mnt/Data/DuguelabServer2/duguelab_general/DugueLab_Research/Current_Projects/KP_LGr_LoGlo/Data_and_Code/ReviewJoN/'
for dirpath, dirnames, filenames in os.walk(root_dir):
    if 'log' in dirnames:
        dirnames.remove('log') # skip 'log' folder
    if 'session2' in dirnames:
        session2_dirpath = os.path.join(dirpath, 'session2')
        session2_dirs.append(session2_dirpath)
file ="run01.fif"
#remove folder 90WCLR (that one doesn't have all the task data) from session2dirs
session2_dirs = [folder for folder in session2_dirs if "90WCLR" not in folder]

dimord = "trl_chan_time"
trialDict = {11 : "full trav out", 12 :"full stand", 21: "fov trav out", 22 : "full trav in" }

for folder in session2_dirs:

    data = ImportHelpers.load_MNE_fif_data(folder + '/' + file, allow_maxshield=True)
    events = mne.find_events(data, "STI101", min_duration = .004)
    events = events[np.where(np.logical_or(np.logical_or(events[:,2]==11, events[:,2]==12), np.logical_or( events[:,2] ==22, events[:,2] ==21)))]
    epochs = mne.Epochs(data, events,baseline = None, tmin=0, tmax=2)

    epochs.plot(picks = "BIO002")