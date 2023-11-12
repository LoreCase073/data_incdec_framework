import argparse
import glob
import json
import os
import shutil
import subprocess
import uuid
from collections import OrderedDict

from joblib import delayed
from joblib import Parallel
import pandas as pd
import cv2

def get_length(input_video):
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', input_video], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return float(result.stdout)


root_path = './Kinetics/'

download_info_path = os.path.join(root_path,'Download')

num_dir = len(os.listdir(download_info_path)) + 1
attempt_name = 'attempt_' + str(num_dir)
attempt_dir = os.path.join(download_info_path,attempt_name)
if not os.path.exists(attempt_dir):
    os.makedirs(attempt_dir)

    video_path = os.path.join(root_path,'Videos')

    info_path = os.path.join(root_path,'Info')

    train_csv_path = os.path.join(info_path,'tbdownloaded.csv')

    train_csv = pd.read_csv(train_csv_path)
    #list of elements in train, which we have to check if they are downloaded
    """ train_ids = train_csv['youtube_id'].tolist()
    time_start = train_csv['time_start'].tolist()
    time_end = train_csv['time_end'].tolist() """
    #iterate and change names to be matched in the directory convention
    train_ids = []
    for index, row in train_csv.iterrows():
        tmp_id = 'id_' + str(row['youtube_id']) + '_' + '{:06d}'.format(row['time_start']) + '_' + '{:06d}'.format(row['time_end'])
        train_ids.append(tmp_id)

    no_download_list = []

    download_list = []

    
    for n in os.listdir(video_path):
        name = n
        name_path = os.path.join(video_path,name)
        #remove directory if no mp4 file and change name if name of directory is different from mp4 file
        mp4_file = [file for file in os.listdir(name_path) if file.endswith('.mp4')]
        cat_csv = os.path.join(name_path,'category.csv')
        
        if mp4_file:
            mp4_name = mp4_file[0]
            mp4_name_nosfx = mp4_name.replace('.mp4','')
            if name != mp4_name_nosfx:
                new_dir_name = os.path.join(video_path,mp4_name_nosfx)
                os.rename(name_path,new_dir_name)
                cat_csv = os.path.join(new_dir_name,'category.csv')
                name = mp4_name_nosfx



        df = pd.read_csv(cat_csv)
        cat_row = next(df.iterrows())[1]
        id_video = cat_row['Filename']
        downloaded = cat_row['Downloaded']
        id_class = cat_row['Category']
        id_behavior = cat_row['Sub-behavior']
        log = cat_row['Log']

        if name != id_video:
            df.at[0,'Filename'] = name
            df.to_csv(cat_csv)
            id_video = name
        

        #to erase not correctly downloaded files, to be uncommented when used
        """ if not(mp4_file) and not(downloaded):
            shutil.rmtree(name_path) """

        #if elements is in train_ids
        if id_video in train_ids:
            #check if downloaded
            if downloaded:
                #rename video
                mp4_file = [file for file in os.listdir(name_path) if file.endswith('.mp4')]
                if mp4_file:
                    mp4_name = mp4_file[0]
                    

                    #check for length of the video
                    v = os.path.join(name_path,mp4_name)
                    duration = get_length(v)

                    #add to error if length is 0.0
                    if duration == 0.0:
                        new_tuple = (name, id_class, id_behavior, 'Error: length is 0.')
                        no_download_list.append(new_tuple)
                    else:
                        #add to list to be printed
                        new_tuple = (name, id_class, id_behavior,duration)
                        download_list.append(new_tuple)
            else:
                new_tuple = (name, id_class, id_behavior, log)
                no_download_list.append(new_tuple)



    df = pd.DataFrame(no_download_list, columns=['Filename', 'Category', 'Sub-behavior', 'Log'])
    output_filename = os.path.join(attempt_dir, 'error_log.csv')
    df.to_csv(output_filename, index=False)

    df2 = pd.DataFrame(download_list, columns=['Filename', 'Category', 'Sub-behavior', 'Seconds'])
    output_filename2 = os.path.join(attempt_dir, 'download_log.csv')
    df2.to_csv(output_filename2, index=False)
else:
    #TODO: fare che è dentro funzione...
    print('Errore...')