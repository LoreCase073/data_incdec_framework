import pandas as pd
import os
import argparse
import shutil

# existing_data_path = './data_incdec_framework/kinetics700_2020/train.csv'

# downloaded_csv_path = './Kinetics/Download/attempt_3/download_log.csv'

# log_csv = './Kinetics/Download/'

# outdir = './Kinetics/'

def make_csv(existing_data_path, downloaded_csv_path, log_csv_path, outdir, attempt_dir):
    # Define the provided activities
    activities = [
        'eating burger', 'eating cake', 'eating carrots', 'eating chips', 'eating doughnuts',
        'eating hotdog', 'eating ice cream', 'eating spaghetti', 'eating watermelon', 'sucking lolly',
        'tasting beer', 'tasting food', 'tasting wine', 'sipping cup',
        'texting', 'talking on cell phone', 'looking at phone',
        'smoking', 'smoking hookah', 'smoking pipe',
        'sleeping', 'yawning', 'headbanging', 'headbutting', 'shaking head',
        'scrubbing face', 'putting in contact lenses', 'putting on eyeliner', 'putting on foundation',
        'putting on lipstick', 'putting on mascara', 'brushing hair', 'brushing teeth', 'braiding hair',
        'combing hair', 'dyeing eyebrows', 'dyeing hair'
    ]

    # Number of elements to take from each category
    n = 100  # Change this value to take more or fewer elements

    # Read the official Kinetics training CSV file
    existing_data = pd.read_csv(existing_data_path)

    #path to latest attempt 
    old_csv = pd.read_csv(downloaded_csv_path)

    

    # Initialize an empty list to store matching activities
    matching_activities = []

    
    #remove elements that gave problems in previous tries
    filtered_existing_data = existing_data
    for dir in os.listdir(log_csv_path):
        tmp_path = os.path.join(log_csv_path, dir)
        tmp_csv_file = os.path.join(tmp_path,'error_log.csv')
        log_csv = pd.read_csv(tmp_csv_file)
        ids = log_csv['Filename'].tolist()
        ids_to_remove = [string[3:] for string in ids]
            
        filtered_existing_data = filtered_existing_data[~filtered_existing_data['youtube_id'].isin(ids_to_remove)]

    # Iterate through the provided activities and find matches in the existing data
    for activity in activities:
        matching_rows = filtered_existing_data[filtered_existing_data['label'] == activity].head(n)
        matching_activities.extend(matching_rows.to_dict('records'))

    # Create a new DataFrame from the matching activities
    new_data = pd.DataFrame(matching_activities)

    # files already downloaded, do not try again
    downloaded = old_csv['Filename'].tolist()
    downloaded_ids = [string[3:] for string in downloaded]

    #filtered_to_download = new_data[~new_data['youtube_id'].isin(downloaded_ids)]

    # Specify the output CSV file
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    info_dir = os.path.join(outdir,'Info')
    if not os.path.exists(info_dir):
        os.makedirs(info_dir)
    #if train.csv exists, move to old
    output_csv_file = os.path.join(info_dir,'train.csv')
    if os.path.exists(output_csv_file):
        old_dir_csv = os.path.join(attempt_dir,'train.csv')
        shutil.move(output_csv_file,old_dir_csv)
    #if tbdownloaded.csv exists, move to old
    to_download_csv = os.path.join(info_dir,'tbdownloaded.csv')
    if os.path.exists(to_download_csv):
        old_dir_csv = os.path.join(attempt_dir,'tbdownloaded.csv')
        shutil.move(to_download_csv,old_dir_csv)



    # Write the new data to the output CSV file
    new_data.to_csv(output_csv_file, index=True, columns=['label', 'youtube_id', 'time_start', 'time_end', 'split'])
    #filtered_to_download.to_csv(to_download_csv, index=True, columns=['label', 'youtube_id', 'time_start', 'time_end', 'split'])



    #Now we repeat with new data for evaluation and test set

    n=120
    eval_match = []
    test_match = []
    to_download = []

    # Iterate through the provided activities and find matches in the existing data
    for activity in activities:
        matching_rows = filtered_existing_data[filtered_existing_data['label'] == activity].head(n)
        eval_match.extend(matching_rows[100:110].to_dict('records'))
        test_match.extend(matching_rows[110:120].to_dict('records'))
        to_download.extend(matching_rows[:120].to_dict('records'))

    # Create a new DataFrame from the matching activities
    eval_data = pd.DataFrame(eval_match)
    test_data = pd.DataFrame(test_match)
    to_dl = pd.DataFrame(to_download)


    val_csv = os.path.join(info_dir,'validation.csv')
    if os.path.exists(val_csv):
        old_dir_csv = os.path.join(attempt_dir,'validation.csv')
        shutil.move(val_csv,old_dir_csv)
    test_csv = os.path.join(info_dir,'test.csv')
    if os.path.exists(test_csv):
        old_dir_csv = os.path.join(attempt_dir,'test.csv')
        shutil.move(test_csv,old_dir_csv)

    # Write the new data to the output CSV file
    eval_data.to_csv(val_csv, index=True, columns=['label', 'youtube_id', 'time_start', 'time_end', 'split'])
    test_data.to_csv(test_csv, index=True, columns=['label', 'youtube_id', 'time_start', 'time_end', 'split'])
    to_dl.to_csv(to_download_csv, index=True, columns=['label', 'youtube_id', 'time_start', 'time_end', 'split'])


if __name__ == '__main__':
    description = 'Helper script for downloading and trimming kinetics videos.'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('existing_data_path', type=str,
                   help=('Path to the existing data csv from Kinetics'))
    p.add_argument('downloaded_csv_path', type=str,
                   help=('CSV file containing the already downloaded files'))
    p.add_argument('log_csv_path', type=str,
                   help=('CSV file containing the log csv, with files to be ignored'))
    p.add_argument('outdir', type=str,
                   help=('output directory for all new csv'))
    
    
    args = p.parse_args()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    info_dir = os.path.join(args.outdir,'Info')
    if not os.path.exists(info_dir):
        os.makedirs(info_dir)
    old_dir = os.path.join(info_dir,'old')
    if not os.path.exists(old_dir):
        os.makedirs(old_dir)
    num_dir = len(os.listdir(old_dir)) + 1
    attempt_name = 'attempt_' + str(num_dir)
    attempt_dir = os.path.join(old_dir,attempt_name)
    if not os.path.exists(attempt_dir):
        os.makedirs(attempt_dir)
        make_csv(args.existing_data_path, args.downloaded_csv_path, args.log_csv_path, args.outdir, attempt_dir)
    else:
        print('Errore...')
    