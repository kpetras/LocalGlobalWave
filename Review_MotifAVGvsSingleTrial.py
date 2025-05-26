import sys
import os

# Add the parent directory of the script (LocalGlobalWave) to the Python path
sys.path.append('/mnt/Data/LoGlo/LocalGlobalWave/LocalGlobalWave/')
from Modules.Utils import ImportHelpers, WaveData as wd, HelperFuns as hf
from Modules.PlottingHelpers import Plotting as plotting
from Modules.SpatialArrangement import SensorLayout
import numpy as numpy
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import pickle
import pandas as pd
from matplotlib.colors import ListedColormap
import itertools
#%%________Set files___________________________________________
folder = '/mnt/Data/LoGlo/'
avg_folder = '/mnt/Data/LoGlo/AVG/'
figsavefolder = '/mnt/Data/DuguelabServer2/duguelab_general/DugueLab_Research/Current_Projects/KP_LGr_LoGlo/Data_and_Code/ReviewJoN/AVG/' 

allMotifsFile = 'AllCondsMotifsEEG_NoThreshold'
MotifsFromGA_File = 'Motifs_EEG_avg_OpticalFlowAfterFilter_Hilbert'
figfolder = '/home/kirsten/Dropbox/Projects/LoGlo/loglofigures/NoThreshold/' 
avg_figfolder = '/home/kirsten/Dropbox/Projects/LoGlo/loglofigures_GA/EEG/' 
fileList = glob.glob(os.path.join(folder, "*",  "EEG_18_OpticalFlowAfterFilter_Hilbert_masked"), recursive=True)
oscillationThresholdFlag = False 
waveData = ImportHelpers.load_wavedata_object(avg_folder + 'EEG_Average_18_OpticalFlowAfterFilter_Hilbert_masked')
modality = 'EEG'


# allMotifsFile = 'AllCondsMotifsMEG_NoThreshold'
# MotifsFromGA_File = 'Motifs_Mag_avg_OpticalFlowAfterFilter_Hilbert'
# figfolder = '/home/kirsten/Dropbox/Projects/LoGlo/loglofigures_meg/NoThreshold/' 
# avg_figfolder = '/home/kirsten/Dropbox/Projects/LoGlo/loglofigures_GA/Mag/' 
# fileList = glob.glob(os.path.join(folder, "*",  "Mag_18_OpticalFlowAfterFilter_Hilbert_masked"), recursive=True)
# oscillationThresholdFlag = False 
# waveData = ImportHelpers.load_wavedata_object(avg_folder + 'Mag_Average_18_OpticalFlowAfterFilter_Hilbert_masked')
# modality = 'Mag'

# allMotifsFile = 'AllCondsMotifsGrad_NoThreshold'
# MotifsFromGA_File = 'Motifs_Grad_avg_OpticalFlowAfterFilter_Hilbert'
# figfolder = '/home/kirsten/Dropbox/Projects/LoGlo/loglofigures_grad/NoThreshold/' 
# avg_figfolder = '/home/kirsten/Dropbox/Projects/LoGlo/loglofigures_GA/Grad/' 
# fileList = glob.glob(os.path.join(folder, "*",  "Grad_18_OpticalFlowAfterFilter_Hilbert_masked"), recursive=True)
# oscillationThresholdFlag = False 
# waveData = ImportHelpers.load_wavedata_object(avg_folder + 'Grad_Average_18_OpticalFlowAfterFilter_Hilbert_masked')
# modality = 'Grad'


# 
#Load GA motifs

# load single trial motifs
filePath = fileList[0]

GA_motif_counts = []
allTrialInfo = []
#% single trial top motifs per subject
with open(folder + allMotifsFile +  '.pickle', 'rb') as handle:
    ST_motifs = pickle.load(handle)
#% motifs from averaged data
with open(avg_folder + MotifsFromGA_File + '.pickle', 'rb') as handle:
    GA_motifs = pickle.load(handle)
#load csv of GA motifs
GA_motif_df = pd.read_csv(f"{avg_figfolder}MotifCountsFull.csv")
with open(folder + 'GA_sorted' + allMotifsFile + '.pickle', 'rb') as handle:
    Motif_GA = pickle.load(handle)
with open(folder + allMotifsFile + 'AllTrialInfo.pickle', 'rb') as handle:
    allTrialInfo = pickle.load(handle)    
with open(figfolder + 'MatchSingleTrialsToTemplate_MotifsFromAVG_UVmaps.pickle', 'rb') as handle:
    templateMatch = pickle.load(handle)
nSubs=19
conds = ['full stand', 'full trav in', 'full trav out']#order is important here. Needs to match that of the GA motifs
avgCondInfo = np.array(list(itertools.chain.from_iterable([[cond]*nSubs for cond in conds])))
trial_to_cond_map = {i: cond for i, cond in enumerate(avgCondInfo)}      
freqs=[5,10]
ST_original_motif_df = pd.read_csv(f"{figfolder}MotifCountsFull.csv")

# Group by Condition, Timepoint, Frequency, and MotifInd and calculate the average count over subjects
motif_counts_ST_original = ST_original_motif_df.groupby(['Condition', 'Timepoint', 'Frequency', 'MotifInd', 'Subject']).size().reset_index(name='Count')

# make all possible combinations
conditions = ST_original_motif_df['Condition'].unique()
timepoints = ST_original_motif_df['Timepoint'].unique()
frequencies = ST_original_motif_df['Frequency'].unique()
motif_inds = ST_original_motif_df['MotifInd'].unique()
subjects = ST_original_motif_df['Subject'].unique()

complete_index = pd.MultiIndex.from_product([conditions, timepoints, frequencies, motif_inds, subjects], names=['Condition', 'Timepoint', 'Frequency', 'MotifInd', 'Subject'])
# Reindex to include all combinations, fill missing values with 0
motif_counts_ST_original = motif_counts_ST_original.set_index(['Condition', 'Timepoint', 'Frequency', 'MotifInd','Subject']).reindex(complete_index, fill_value=0).reset_index()
