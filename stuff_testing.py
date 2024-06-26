#per fare download kinetics, prima installare youtube-dl
#https://pypi.org/project/youtube_dl/
#UPDATE: aggiornare o installare direttamente da:
#pip install --upgrade --force-reinstall git+https://github.com/ytdl-org/youtube-dl.git
#altrimenti da prpoblemi

#dopo installare il downloader del crawler...
#https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics
#seguire le istruzioni su come setuppare un environment per kinetics

#UPDATE: installando da problemi, qui dice che non serve installare, vedere se necessario installare a mano qualche pacchetto.
#https://stackoverflow.com/questions/66609054/ruamel-yaml-constructor-constructorerror-could-not-determine-a-constructor-for

#per fare download dei video: 
#mkdir <data_dir>; python download.py {dataset_split}.csv <data_dir>

#per farlo funzionare serve anche FFmpeg e FFprobe
#FFmpeg: https://anaconda.org/conda-forge/ffmpeg


#Fino a qui dovrebbe adesso funzionare tutto per fare download dei video

#Poi prendo il generate_video_jpgs.py da: https://github.com/kenshohara/3D-ResNets-PyTorch/blob/master/util_scripts/generate_video_jpgs.py
#e kinetics_json

""" 
Download videos using the official crawler.

Locate test set in video_directory/test.

Convert from avi to jpg files using util_scripts/generate_video_jpgs.py

python -m util_scripts.generate_video_jpgs mp4_video_dir_path jpg_video_dir_path kinetics

Generate annotation file in json format similar to ActivityNet using util_scripts/kinetics_json.py

The CSV files (kinetics_{train, val, test}.csv) are included in the crawler.

python -m util_scripts.kinetics_json csv_dir_path 700 jpg_video_dir_path jpg dst_json_path 
"""

#es: python generate_video_jpgs.py ./video/ ./jpgs/ kinetics



#to generate videos via command line

#mkdir <data_dir>; python download.py {dataset_split}.csv <data_dir>

#python ./ActivityNet/Crawler/Kinetics/download.py ./kinetics700_2020/trying.csv ./video/

#es: python generate_video_jpgs.py ./video/ ./jpgs/ kinetics
import pandas as pd

""" activities = [
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

validate_csv = './Kinetics/Validation_Info/validate_kinetics.csv'
output_csv = './Kinetics/Validation_Info/validation.csv'
df1 = pd.read_csv(validate_csv)

matching_activities = []

for activity in activities:
        matching_rows = df1[df1['label'] == activity]
        matching_activities.extend(matching_rows.to_dict('records'))

new_data = pd.DataFrame(matching_activities)

new_data.to_csv(output_csv, index=True, columns=['label', 'youtube_id', 'time_start', 'time_end', 'split']) """


""" error_log = './Kinetics/Download/attempt_13/error_log.csv'
df2 = pd.read_csv(error_log)
data_csv = './Kinetics/Info/tbdownloaded.csv'
data = pd.read_csv(data_csv)
print(data[data.duplicated(subset=['youtube_id'], keep=False)])
ids = df1['Filename'].tolist()
ids_to_remove = [string[3:] for string in ids]
filtered_existing_data = data[~data['youtube_id'].isin(ids_to_remove)]
print(filtered_existing_data)

ids = df2['Filename'].tolist()
ids_to_remove = [string[3:] for string in ids]
filtered_existing_data = filtered_existing_data[~filtered_existing_data['youtube_id'].isin(ids_to_remove)]


print(filtered_existing_data) """




second_csv = './Kinetics/second/Info/tbdownloaded.csv'
first_csv = './Kinetics/Info/tbdownloaded.csv'

first_data = pd.read_csv(first_csv)
second_data = pd.read_csv(second_csv)

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

nb_comon_elements = second_data['youtube_id'].isin(first_data['youtube_id']).sum()

print(len(first_data['youtube_id']) - nb_comon_elements)

data_existing = second_data[second_data['youtube_id'].isin(first_data['youtube_id'])]

data = second_data[~second_data['youtube_id'].isin(first_data['youtube_id'])]

print(data)
print(data_existing)
""" 
for activity in activities:
        matching_rows = second_data[second_data['label'] == activity]
        print(len(matching_rows)) """

""" first_csv = './data_incdec_framework/kinetics700_2020/train.csv'
first_data = pd.read_csv(first_csv)

first_data = first_data.sample(frac = 1)

first_data.to_csv('./data_incdec_framework/kinetics700_2020/train_shuffled.csv',index=False) """