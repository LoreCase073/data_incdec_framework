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

download_log = './Kinetics/Download/attempt_3/download_log.csv'
df1 = pd.read_csv(download_log)
error_log = './Kinetics/Download/attempt_3/error_log.csv'
df2 = pd.read_csv(error_log)
data_csv = './Kinetics/Info/train.csv'
data = pd.read_csv(data_csv)
print(data[data.duplicated(subset=['youtube_id'], keep=False)])
""" ids = df1['Filename'].tolist()
ids_to_remove = [string[3:] for string in ids]
filtered_existing_data = data[~data['youtube_id'].isin(ids_to_remove)]
print(filtered_existing_data)

ids = df2['Filename'].tolist()
ids_to_remove = [string[3:] for string in ids]
filtered_existing_data = filtered_existing_data[~filtered_existing_data['youtube_id'].isin(ids_to_remove)]


print(filtered_existing_data) """