import subprocess
import argparse
import os
#python ./data_incdec_framework/create_kinetics_dataset.py ./data_incdec_framework/ ./Kinetics/Info/train.csv ./Kinetics/Videos/ jpgs

def download_kinetics(script_path, input_csv, output_dir,trim_format='%06d', num_jobs=5, tmp_dir='/tmp/kinetics',
                      behaviors=True):
    script = os.path.join(script_path,'download_kinetics.py')
    command = ['python', '%s' % script, '%s' % input_csv, '%s' % output_dir, 
               '-f', '%s' % trim_format,
               '-n', '%s' % num_jobs, 
               '-t', '%s' % tmp_dir, 
               '--behaviors','%s' % behaviors]
    command = ' '.join(command)

    
    output = subprocess.check_output(command, shell=True,
                                                stderr=subprocess.STDOUT)


def make_images(script_path, dir_path, dst_path, dataset='kinetics', n_jobs=5, fps=5, size=240):

    script = os.path.join(script_path,'generate_video_jpgs.py')
    command = ['python', '%s' % script,'"%s"' % dir_path, '"%s"' % dst_path, '"%s"' % dataset,
               '--n_jobs', '"%s"' % n_jobs,
               '--fps', '"%s"' % fps, 
               '--size', '"%s"' % size]
    command = ' '.join(command)

    
    output = subprocess.check_output(command, shell=True,
                                                stderr=subprocess.STDOUT)

if __name__ == '__main__':
    description = 'Helper script for downloading and trimming kinetics videos.'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('path_to_scripts', type=str,
                   help=('Path to the scripts'))
    p.add_argument('input_csv', type=str,
                   help=('CSV file containing the following format: '
                         'YouTube Identifier,Start time,End time,Class label'))
    p.add_argument('output_dir', type=str, default='./Kinetics/Videos',
                   help='Output directory where videos will be saved.')
    p.add_argument('dst_images', type=str, default='jpgs',
                   help='Output directory where videos will be saved.')
    
    
    args = p.parse_args()

    """ download_kinetics(args.path_to_scripts, args.input_csv, args.output_dir, num_jobs=2) """

    fps = 10
    make_images(args.path_to_scripts, args.output_dir, args.dst_images, 'kinetics', 
                n_jobs=1, fps=fps, size=240)