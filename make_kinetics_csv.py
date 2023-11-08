import pandas as pd
import os
import argparse

# existing_data_path = './data_incdec_framework/kinetics700_2020/train.csv'

# downloaded_csv_path = './Kinetics/Download/attempt_1/download_log.csv'

# log_csv = './Kinetics/error_log.csv'

def make_csv(existing_data_path, downloaded_csv_path, log_csv_path):
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

    old_csv = pd.read_csv(downloaded_csv_path)

    log_csv = pd.read_csv(log_csv_path)

    # Initialize an empty list to store matching activities
    matching_activities = []

    #remove elements that gave problems in previous tries
    ids = log_csv['Filename'].tolist()
    ids_to_remove = [string[3:] for string in ids]
        
    filtered_existing_data = existing_data[~existing_data['youtube_id'].isin(ids_to_remove)]

    # Iterate through the provided activities and find matches in the existing data
    for activity in activities:
        matching_rows = filtered_existing_data[filtered_existing_data['label'] == activity].head(n)
        matching_activities.extend(matching_rows.to_dict('records'))

    # Create a new DataFrame from the matching activities
    new_data = pd.DataFrame(matching_activities)

    # files already downloaded, do not try again
    downloaded = old_csv['Filename'].tolist()
    downloaded_ids = [string[3:] for string in downloaded]

    filtered_to_download = new_data[~new_data['youtube_id'].isin(downloaded_ids)]

    # Specify the output CSV file
    kinetics_dir = './Kinetics/'
    if not os.path.exists(kinetics_dir):
        os.makedirs(kinetics_dir)
    info_dir = os.path.join(kinetics_dir,'Info')
    if not os.path.exists(info_dir):
        os.makedirs(info_dir)
    output_csv_file = './Kinetics/Info/train.csv'
    to_download_csv = './Kinetics/Info/tbdownloaded.csv'



    # Write the new data to the output CSV file
    new_data.to_csv(output_csv_file, index=True, columns=['label', 'youtube_id', 'time_start', 'time_end', 'split'])
    filtered_to_download.to_csv(to_download_csv, index=True, columns=['label', 'youtube_id', 'time_start', 'time_end', 'split'])



    #Now we repeat with new data for evaluation and test set

    n=140
    eval_match = []
    test_match = []

    # Iterate through the provided activities and find matches in the existing data
    for activity in activities:
        matching_rows = filtered_existing_data[filtered_existing_data['label'] == activity].head(n)
        eval_match.extend(matching_rows[100:120].to_dict('records'))
        test_match.extend(matching_rows[120:140].to_dict('records'))

    # Create a new DataFrame from the matching activities
    eval_data = pd.DataFrame(eval_match)
    test_data = pd.DataFrame(test_match)


    val_csv = './Kinetics/Info/validation.csv'
    test_csv = './Kinetics/Info/test.csv'

    # Write the new data to the output CSV file
    eval_data.to_csv(val_csv, index=True, columns=['label', 'youtube_id', 'time_start', 'time_end', 'split'])
    test_data.to_csv(test_csv, index=True, columns=['label', 'youtube_id', 'time_start', 'time_end', 'split'])


if __name__ == '__main__':
    description = 'Helper script for downloading and trimming kinetics videos.'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('existing_data_path', type=str,
                   help=('Path to the existing data csv from Kinetics'))
    p.add_argument('downloaded_csv_path', type=str,
                   help=('CSV file containing the already downloaded files'))
    p.add_argument('log_csv_path', type=str,
                   help=('CSV file containing the log csv, with files to be ignored'))
    
    
    args = p.parse_args()
    make_csv(args.existing_data_path, args.downloaded_csv_path, args.log_csv_path)

    