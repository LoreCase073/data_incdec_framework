import pandas as pd

# Data dictionary
data_dict = {
    'food': [
        'eating burger', 'eating cake', 'eating carrots', 'eating chips', 'eating doughnuts',
        'eating hotdog', 'eating ice cream', 'eating spaghetti', 'eating watermelon',
        'sucking lolly', 'tasting beer', 'tasting food', 'tasting wine', 'sipping cup'
    ],
    'phone': [
        'texting', 'talking on cell phone', 'looking at phone'
    ],
    'smoking': [
        'smoking', 'smoking hookah', 'smoking pipe'
    ],
    'fatigue': [
        'sleeping', 'yawning', 'headbanging', 'headbutting', 'shaking head'
    ],
    'selfcare': [
        'scrubbing face', 'putting in contact lenses', 'putting on eyeliner', 'putting on foundation',
        'putting on lipstick', 'putting on mascara', 'brushing hair', 'brushing teeth', 'braiding hair',
        'combing hair', 'dyeing eyebrows', 'dyeing hair'
    ]
}

# Create a list to store class names and subcategories
data = []

# Iterate through the data_dict and populate the data list
for class_name, subcategories in data_dict.items():
    for subcategory in subcategories:
        data.append({'Class': class_name, 'Subcategory': subcategory})

# Create a pandas DataFrame from the data list
df = pd.DataFrame(data)

# Output CSV file
output_csv_file = 'classes.csv'

# Write the DataFrame to a CSV file
df.to_csv(output_csv_file, index=False)

print(f'CSV file "{output_csv_file}" has been created with class names and corresponding subcategories.')
