import os
import pandas as pd
import argparse
from pathlib import Path

def get_class_subcategory(id_video_path, category_file_name, subcat_name):
    if not id_video_path.is_dir():
        return
    
    csv_file_name = os.path.join(id_video_path,category_file_name)

    csv_file = pd.read_csv(csv_file_name)

    class_name = csv_file['Category'][0]
    subcat = csv_file[subcat_name][0]

    return class_name, subcat


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dir_path', default=None, type=Path, help='Directory path of videos')
    parser.add_argument(
        'dst_path',
        default='random_stuff',
        type=Path,
        help='Directory path to save of classes.csv')
    parser.add_argument(
        'dataset',
        default='kinetics',
        type=str,
        help='Dataset name (kinetics | vzc)')
    
    args = parser.parse_args()

    video_file_path = [x for x in sorted(args.dir_path.iterdir())]

    if args.dataset == 'kinetics':
        subcat_name = 'Sub-behavior'
    else:
        subcat_name = 'vehicle_id'

    category_file_name = 'category.csv'

    class_subcat_dict = {}

    for video_dir in video_file_path:
        class_name, subcat = get_class_subcategory(video_dir, category_file_name, subcat_name)
        if class_name not in class_subcat_dict:
            class_subcat_dict[class_name] = []
        if subcat not in class_subcat_dict[class_name]:
            class_subcat_dict[class_name].append(subcat)

    # Create a list to store class names and subcategories
    data = []

    # Iterate through the data_dict and populate the data list
    for class_name, subcategories in class_subcat_dict.items():
        for subcategory in subcategories:
            data.append({'Class': class_name, 'Subcategory': subcategory})

    # Create a pandas DataFrame from the data list
    df = pd.DataFrame(data)

    # Output CSV file
    output_csv_file = os.path.join(args.dst_path, 'classes.csv')

    # Write the DataFrame to a CSV file
    df.to_csv(output_csv_file, index=False)

    print(f'CSV file "{output_csv_file}" has been created with class names and corresponding subcategories.')