import pandas as pd
import os

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

# Read existing CSV file
existing_data = pd.read_csv('./data_incdec_framework/kinetics700_2020/train.csv')

# Initialize an empty list to store matching activities
matching_activities = []

# Iterate through the provided activities and find matches in the existing data
for activity in activities:
    matching_rows = existing_data[existing_data['label'] == activity].head(n)
    matching_activities.extend(matching_rows.to_dict('records'))

# Create a new DataFrame from the matching activities
new_data = pd.DataFrame(matching_activities)

# Specify the output CSV file
kinetics_dir = './Kinetics/'
if not os.path.exists(kinetics_dir):
    os.makedirs(kinetics_dir)
info_dir = os.path.join(kinetics_dir,'Info')
if not os.path.exists(info_dir):
    os.makedirs(info_dir)
output_csv_file = './Kinetics/Info/train.csv'



# Write the new data to the output CSV file
new_data.to_csv(output_csv_file, index=True, columns=['label', 'youtube_id', 'time_start', 'time_end', 'split'])



#Now we repeat with new data for evaluation and test set

n=140
eval_match = []
test_match = []

# Iterate through the provided activities and find matches in the existing data
for activity in activities:
    matching_rows = existing_data[existing_data['label'] == activity].head(n)
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