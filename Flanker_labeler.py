import os, sys, datetime, csv, shutil, glob
from glob import glob
from datetime import datetime
import pandas as pd

class pcat:
    # This class is used to condense p-cat flanker behavioral data and
    # annotate fnris data with relavent data in subjects NIRS .evt files.
    # Within these .evt files the columns correspond to the following order...
    # fnris sample, block, directionality, congruency, direction, response
    #
    # Correct response can be infered
    def __init__(self):
        self.beh_folder = "/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R56/task_data/"
        self.nirs_folder = "/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R56/NIRS_data/"
        self.analysis_folder = '../../../../analysis/P-CAT/'

        self.ex_subs = []
        self.participant_num_len = 4
        self.sample_rate = 7.81250

        self.master = {}
        self.translator = { # Translator used to convert stimuli info to integers for NIRS .evts files
            'ND' : '0', # Directionality
            'D' : '1', 
            'I' : '0', # Congruency - .evt file column 
            'C' : '1',
            'L' : '0', # Direction
            'R' : '1',
            'None': '0', # User response
            'left': '0',
            'right': '1',
            'incorrect': '0', # Accuracy
            'correct': '1'
        }

        # Functional variables
        self.pad = lambda x, n: '0'*(n - len(str(x))) + str(x) # Function for padding zeros onto subject ID

    def iterate_behavioral(self):
        task_files = glob(f'{self.beh_folder}*/*lanker/*.csv')
        current_block = None
        for task_filepath in task_files: # For each task and it's given nickname find subjects data
            print(f"Loading {task_filepath}...")
            subject = task_filepath.split(self.beh_folder)[1][:4]
            subject_files =  glob(f'{self.beh_folder}{subject}/*lanker/*.csv')
            print(f"Found {subject_files}")

            if len(subject_files) > 1: # Make sure the file we have is the last file for the subject
                last_date = [0, 0]
                last_file = None
                for subject_file in subject_files:
                    date = [int(datum) for datum in subject_file[:-4].split('_')[-2:]]
                    if date[0] > last_date[0]:
                        last_file = subject_file
                        last_date = date
                    if date[1] > last_date[1]:
                        last_file = subject_file
                        last_date = date
                task_filepath = last_file

            print(f'Processing subject {subject}...')
            with open(task_filepath, 'r') as file: # Open file and grab data
                csvreader = csv.reader(file)
                data = [line for line in csvreader]
            
            if subject not in self.master.keys():
                self.master[subject] = {}
            
            for row_ind, row in enumerate(data): # Add in all data points to the list
                # Grab pertinent data
                if row[0] == None or row[0] == '': continue
                stimuli = row[0].split('/')
                if len(stimuli) == 1: 
                    print(f"Skipping {subject} row {row_ind}")
                    continue
                else:
                    stimuli = stimuli[1].split('.')[0] # Grab stimuli and remove excess
                stimuli = stimuli.split('_')

                if stimuli[0] == 'practice': continue
                
                block = stimuli[1].split('block')[1]
                if block != current_block:
                    current_block = block
                    block = 1
                else:
                    block = 0
                
                directionality = stimuli[2]
                if directionality == 'D':
                    direction = stimuli[3][1]
                    congruency = stimuli[3][0]
                else:
                    direction = stimuli[3][0]
                    congruency = 'ND'

                stimuli = False
                response = None # Find subject response and start for trial
                start = None 
                for header_ind, header in enumerate(data[0]): # Iterate through data sheet columns and find columns of interest
                    value = row[header_ind]
                    if value is None or value == 'None':
                        continue

                    if value:
                        if "lets_go.stopped" == header:
                            self.lets_go = float(value) # Grab nirs recording start
                            self.master[subject][str(self.lets_go)] = self.lets_go
                        if 'stimuli_' == header[:8]:
                            stimuli = True
                            if 'started' == header[-7:]:
                                start = float(value)
                        if 'keys' == header[-4:]: # If the column header is in our columns of interest  
                                response = value

                if response is None:
                    continue

                if stimuli is False:
                    continue

                if self.translator[response] == self.translator[direction]: accuracy = 'correct'
                else: accuracy = 'incorrect'
                
                # Translate info into .evt style numbers
                trial_data = [str(self.translator[datum]) if datum in self.translator.keys() else datum for datum in [block, directionality, congruency, direction, response, accuracy]]

                # Add info into mastersheet
                self.master[subject][str(start)] = trial_data

    def sync_nirs(self, replace = False):
        """ 
        Sync behavioral data with fNIRS data
         
        Arguments:
            replace (optional) - bool - Replace existing events  
        """
        for subject in self.master.keys(): # Iterate through subjects in extract file
            print(f'Syncing {subject} NIRS data')
            # Grab task specific task filenames
            nirs_session_dir = glob(self.nirs_folder+f"{subject}/")

            if len(nirs_session_dir) < 1:
                print(f'Flanker NIRS files not found for {subject}: {nirs_session_dir}')
                continue
            else:
                nirs_session_dir = nirs_session_dir[0]

            evts = glob(nirs_session_dir + "*/*.evt")
            print(f"Events found: {evts}")
            old_evt = None
            if len(evts) > 1:
                if replace == True:
                    for evt in evts:
                        if '_old.evt' == evt[-8:]:
                            old_evt = evt
                        else:
                            current_evt = evt
                            print(f"Current events: {current_evt}")
                else:            
                    print(f"{len(evts)} .evt files - Skipping subject {subject}: {nirs_session_dir}")
                    continue
            elif(len(evts) == 1):
                old_evt = evts[0]
            else:
                print(f"No event files found! Skipping...")
                continue

            # open original evt
            f = open(old_evt, 'r')
            line1 = f.readline()
            f.close()
            
            # first time marker is NIRS stim start 1
            NSstim1_t = line1.split('\t')[0]
            if NSstim1_t == None or NSstim1_t == '':
                print(f"Time marker not found! Skipping {subject}...")
                continue

            #output_lines = [line1]
            output_lines = []

            converted_stims = []
            timestamps = list(self.master[subject].keys())
            timestamps = [float(temp_timestamp) for temp_timestamp in timestamps]
            if len(timestamps) > 0:
                lets_go = float(timestamps.pop(0))
                initial_timestamp = float(timestamps[0])
                start_shift = float(initial_timestamp - lets_go)
            else:
                print(f"Empty timestamps, skipping...")
                continue

            for timestamp in timestamps:
                converted_stims.append(( # Add stim to current stims
                    self.timeconvert_psychopy_to_nirstar(
                        float(self.sample_rate),
                        float(NSstim1_t),
                        float(timestamp),
                        float(initial_timestamp),
                        float(start_shift)
                        ),
                    self.master[subject][str(timestamp)])
                )

            for (stim_time, stimuli) in converted_stims:
                line = str(round(stim_time))
                line += '\t1'
                for evt_col in stimuli:
                    line += "\t"
                    line += str(evt_col)
                #line += '\t0'
                line += "\n"
                    
                output_lines.append(line)

            # move OG evt to _old.evt
            if old_evt is None:
                shutil.move(
                    current_evt,
                    current_evt.replace(".evt", "_old.evt"))

            print(f"Outputting events too {current_evt}")
            f = open(current_evt, 'w')
            for line in output_lines:
                f.write(line)
            
            f.close()
        return

    def timeconvert_psychopy_to_nirstar(self, sample_rate, NSstim1_t,  PSstim_t, PSstim1_t, start_shift):
        """ Convert timestamps from psycopy to nirstar """
        NSevent_t = ((PSstim_t - PSstim1_t) * sample_rate) + NSstim1_t                    
        print(f"nt: {NSevent_t}\nn1: {NSstim1_t}\npt: {PSstim_t}\np1: {PSstim1_t}")
        return NSevent_t

    def save_master(self):
        """ Save events to a master sheet """
        final_master = {}
        headers = ['block', 'directionality', 'congruency', 'direction', 'response', 'accuracy']
        for subject in self.master.keys():
            for timepoint in self.master[subject].keys():
                row = f'{subject}-{timepoint}'
                if row not in final_master.keys():
                    final_master[row] = {potential_header:None for potential_header in headers}
                
                final_master[row] = self.master[subject][timepoint]
                            
        dataset = [['ID', 'Time'] + headers]
        for row in final_master.keys():
            data = final_master[row]
            dataset.append(row.split('-') + data)

        with open(f'{self.analysis_folder}P-CAT_Behavioral_Masterfile.csv', 'w') as file:
            csvwriter = csv.writer(file)
            for row in dataset:
                csvwriter.writerow(row)


class pcat_psu(pcat):

    def __init__(self):
        """
        This class is used to condense p-cat flanker behavioral data and
        annotate fnris data with relavent data in subjects NIRS .evt files.
        Within these .evt files the columns correspond to the following order...
        fnris sample, block, directionality, congruency, direction, response
        
        Correct response can be inferet
        """
        self.beh_folder = "/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R56/restructured_data/PSU_data/task_data/flanker/"#'../../../../study_data/P-CAT/R56/restructured_data/task_data/flanker/'
        self.nirs_folder = "/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R56/restructured_data/PSU_data/fnirs_data/flanker/"# '../../../../analysis/P-CAT/flanker_nirs/'
        self.analysis_folder = '../../../../analysis/P-CAT/'

        self.ex_subs = []
        self.participant_num_len = 4
        self.sample_rate = 7.81250

        self.master = {}

        self.translator = { # Translator used to convert stimuli info to integers for NIRS .evts files
            'ND' : '0', # Directionality
            'D' : '1', 
            'I' : '0', # Congruency - .evt file column 
            'C' : '1',
            'L' : '0', # Direction
            'R' : '1',
            'None': '0', # User response
            'left': '0',
            'right': '1',
            'incorrect': '0', # Accuracy
            'correct': '1'
        }

        # Functional variables
        self.pad = lambda x, n: '0'*(n - len(str(x))) + str(x) # Function for padding zeros onto subject ID

    def iterate_behavioral(self):
        """ Iterate through behavioral data and save to master df """

        task_files = glob(f'{self.beh_folder}*/*.csv')
        current_block = None
        for task_filepath in task_files: # For each task and it's given nickname find subjects data
            print(f"Loading {task_filepath}...")
            subject = task_filepath.split(self.beh_folder)[1][:4]
            subject_files =  glob(f'{self.beh_folder}{subject}/*.csv')
            print(f"Found {subject_files}")

            if len(subject_files) > 1: # Make sure the file we have is the last file for the subject
                last_date = [0, 0]
                last_file = None
                for subject_file in subject_files:
                    date = [int(datum) for datum in subject_file[:-4].split('_')[-2:]]
                    if date[0] > last_date[0]:
                        last_file = subject_file
                        last_date = date
                    if date[1] > last_date[1]:
                        last_file = subject_file
                        last_date = date
                task_filepath = last_file

            print(f'Processing subject {subject}...')
            with open(task_filepath, 'r') as file: # Open file and grab data
                csvreader = csv.reader(file)
                data = [line for line in csvreader]
            
            if subject not in self.master.keys():
                self.master[subject] = {}
            for row_ind, row in enumerate(data): # Add in all data points to the list
                # Grab pertinent data
                if row[0] == None or row[0] == '': continue
                stimuli = row[0].split('/')
                if len(stimuli) == 1: 
                    print(f"Skipping {subject} row {row_ind}")
                    continue
                else:
                    stimuli = stimuli[1].split('.')[0] # Grab stimuli and remove excess
                stimuli = stimuli.split('_')

                if stimuli[0] == 'practice': continue
                
                block = stimuli[1].split('block')[1]
                if block != current_block:
                    current_block = block
                    block = 1
                else:
                    block = 0
                
                directionality = stimuli[2]
                if directionality == 'D':
                    direction = stimuli[3][1]
                    congruency = stimuli[3][0]
                else:
                    direction = stimuli[3][0]
                    congruency = 'ND'

                stimuli = False
                response = None # Find subject response and start for trial
                start = None 
                for header_ind, header in enumerate(data[0]): # Iterate through data sheet columns and find columns of interest
                    value = row[header_ind]
                    if value is None or value == 'None':
                        continue

                    if value:
                        if "lets_go.stopped" == header:
                            self.lets_go = float(value) # Grab nirs recording start
                            self.master[subject][str(self.lets_go)] = 0
                        if 'stimuli_' == header[:8]:
                            stimuli = True
                            if 'started' == header[-7:]:
                                start = float(value)
                        if 'keys' == header[-4:]: # If the column header is in our columns of interest  
                                response = value

                if response is None:
                    continue

                if stimuli is False:
                    continue

                if self.translator[response] == self.translator[direction]: accuracy = 'correct'
                else: accuracy = 'incorrect'
                
                # Translate info into .evt style numbers
                trial_data = [str(self.translator[datum]) if datum in self.translator.keys() else datum for datum in [block, directionality, congruency, direction, response, accuracy]]

                # Add info into mastersheet
                self.master[subject][str(start)] = trial_data

    def sync_nirs(self, replace = False):
        """ Sync behavioral event withs with fNIRS scans and create new .evt files"""
        for subject in self.master.keys(): # Iterate through subjects in extract file
            print(f'Syncing {subject} NIRS data')
            # Grab task specific task filenames
            nirs_session_dir = glob(self.nirs_folder+f"{subject}/")

            if len(nirs_session_dir) < 1:
                print(f'Flanker NIRS files not found for {subject}: {nirs_session_dir}')
                continue
            else:
                nirs_session_dir = nirs_session_dir[0]

            # only do this if there's only 1 .evt file in study folder
            evts = glob(nirs_session_dir + "*.evt")
            print(f"Events found: {evts}")
            old_evt = None
            if len(evts) != 1:
                if replace == True:
                    for evt in evts:
                        if '_old' in evt:
                            old_evt = evt
                        else:
                            current_evt = evt
                else:            
                    print(f"{len(evts)} .evt files - Skipping subject {subject}: {nirs_session_dir}")
                    continue
            elif len(evts) == 1:
                old_evt = evts[0]
            else:
                print(f"Missing events files for {subject}, skipping...")
                continue

            # open original evt
            f = open(old_evt, 'r')
            line1 = f.readline()
            f.close()
            
            # first time marker is NIRS stim start 1
            NSstim1_t = line1.split('\t')[0]
            if NSstim1_t == None or NSstim1_t == '':
                print(f"Time marker not found! Skipping {subject}...")
                continue

            #output_lines = [line1]
            output_lines = []

            converted_stims = []
            timestamps = list(self.master[subject].keys())
            timestamps = sorted([float(temp_timestamp) for temp_timestamp in timestamps])
            if len(timestamps) > 0:
                lets_go = timestamps.pop(0)
                initial_timestamp = timestamps[0]
                start_shift = float(initial_timestamp - lets_go)
            else:
                continue

            for timestamp in timestamps:
                converted_stims.append(( # Add stim to current stims
                    self.timeconvert_psychopy_to_nirstar(
                        float(self.sample_rate),
                        float(NSstim1_t),
                        float(timestamp),
                        float(initial_timestamp),
                        float(start_shift)
                        ),
                    self.master[subject][str(timestamp)])
                )

            for (stim_time, stimuli) in converted_stims:
                line = str(round(stim_time))
                line += '\t1'
                for evt_col in stimuli:
                    line += "\t"
                    line += str(evt_col)
                #line += '\t0'
                line += "\n"
                    
                output_lines.append(line)

            # move OG evt to _old.evt
            if old_evt is None:
                shutil.move(
                    current_evt,
                    current_evt.replace(".evt", "_old.evt"))

            f = open(current_evt, 'w')
            for line in output_lines:
                f.write(line)
            
            f.close()
        return