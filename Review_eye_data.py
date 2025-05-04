import sys
import os

# Add the parent directory of the script (LocalGlobalWave) to the Python path
sys.path.append('/mnt/Data/LoGlo/LocalGlobalWave/LocalGlobalWave/')

from Modules.Utils import WaveData as wd
from Modules.Utils import ImportHelpers
import mne
import numpy as np
import os
import pandas as pd
import subprocess


#%% Import from MNE-Data
root_dir = '/mnt/Data/DuguelabServer2/duguelab_general/DugueLab_Research/Current_Projects/LGr_GM_JW_DH_LD_WavesModel/Experiments/Data/data_MEEG/ET/'
session2_dirs = []
savePath = '/mnt/Data/DuguelabServer2/duguelab_general/DugueLab_Research/Current_Projects/KP_LGr_LoGlo/Data_and_Code/ReviewJoN/'
for dirpath, dirnames, filenames in os.walk(root_dir):
    if 'log' in dirnames:
        dirnames.remove('log') # skip 'log' folder
    if 'session2' in dirnames:
        session2_dirpath = os.path.join(dirpath, 'session2')
        session2_dirs.append(session2_dirpath)
file ="*.edf"
#remove folder 90WCLR (that one doesn't have all the task data) from session2dirs
session2_dirs = [folder for folder in session2_dirs if "90WCLR" not in folder]


for folder in session2_dirs:
    print ("Loading folder: " + folder)
    parent_dir = os.path.dirname(folder)
    # make save folder:
    parent_dir = os.path.dirname(folder)
    parent_folder = os.path.basename(parent_dir)    
    # make save folder:
    savefolder = os.path.join(savePath, parent_folder)    
    print("Saving to: " + savefolder)
for file in os.listdir(folder):    
    print("Loading file: " + file)
    if file.endswith('.edf'):
        # Load the EDF file
        edf_path = os.path.join(folder, file)
        #convert to ascii format using e2a.sh
        base_name = os.path.splitext(file)[0]  # Get the file name without extension

        # Paths for saving converted files
        asc_dir = os.path.join(folder, "asc")
        edf_dir = os.path.join(folder, "edf")
        os.makedirs(asc_dir, exist_ok=True)
        os.makedirs(edf_dir, exist_ok=True)

        # Run the conversion commands
        try:
            # Convert to ASCII format
            subprocess.run(f"./edf2asc -s -miss -1.0 {edf_path}", shell=True, check=True)
            subprocess.run(f"cat {base_name}.asc | awk 'BEGIN{{FS=\" \"}}{{print $1\"\\t\"$2\"\\t\"$3\"\\t\"$4}}' > dat.tmp", shell=True, check=True)
            subprocess.run(f"mv dat.tmp {asc_dir}/{base_name}.dat", shell=True, check=True)
            subprocess.run(f"rm {base_name}.asc", shell=True, check=True)

            # Extract messages
            subprocess.run(f"./edf2asc -e {edf_path}", shell=True, check=True)
            subprocess.run(f"cat {base_name}.asc | grep -E 'MSG|START' > {base_name}.msg", shell=True, check=True)
            subprocess.run(f"mv {base_name}.msg {asc_dir}/", shell=True, check=True)
            subprocess.run(f"rm {base_name}.asc", shell=True, check=True)

            # Move the original EDF file
            subprocess.run(f"mv {edf_path} {edf_dir}/", shell=True, check=True)

            print(f"Successfully converted and saved files for {file}")
        except subprocess.CalledProcessError as e:
            print(f"Error processing {file}: {e}")

        raw =  mne.io.read_raw_eyelink(ascEDFPath)
        # Get the events from the raw data
        events = mne.find_events(raw, stim_channel='STI 014')
        # Create a DataFrame from the events
        events_df = pd.DataFrame(events, columns=['onset', 'event_id', 'duration'])
        # Save the DataFrame to a CSV file
        csv_filename = os.path.splitext(file)[0] + '_events.csv'
        csv_path = os.path.join(savefolder, csv_filename)
        events_df.to_csv(csv_path, index=False)