import os
import argparse
from utils import quat2h5, QuatSegment, SegmentType, TargetType
from collections import OrderedDict
import traceback

SEQ_PREFIX='processed_recorded_seq1_'

quat_cfg = OrderedDict([
    ('traj', QuatSegment(length=7, type=SegmentType.BOTH, target=TargetType.POS)),
    ('handjs', QuatSegment(length=6, type=SegmentType.BOTH, target=TargetType.POS)),
    ('handvel', QuatSegment(length=6, type=SegmentType.OBSERVATION, target=TargetType.VELOCITY)),
    ('handforce', QuatSegment(length=6, type=SegmentType.ACTION, target=TargetType.POS)),
])

def process_folders(root_path, out_path):
    """
    Process each input folder by calling quat2h5 function
    """
    input_folders = []
    for item in os.listdir(root_path):
        full_path = os.path.join(root_path, item)
        if os.path.isdir(full_path):
            input_folders.append(full_path)

    print(f"found {len(input_folders)} folders to process")

    for episodes_cnt, folder in enumerate(input_folders):
        if not os.path.exists(folder):
            print(f"Warning: Folder {folder} does not exist. Skipping...")
            continue
        
        basename = os.path.basename(folder)
        try:
            print(f"Processing folder: {folder}")
            seq_file_name = f"{SEQ_PREFIX}{basename}.quat"
            out_names = f'episode_{episodes_cnt}.hdf5'
            print(f"folder {folder}, seq_file_name {seq_file_name}, out_names {out_names}")
            quat2h5(folder, seq_file_name, 
                    os.path.join(out_path, out_names), quat_cfg=quat_cfg)
            print(f"Successfully processed {folder}")
        except Exception as e:
            print(f"Error processing folder {folder}: {str(e)}")
            traceback.print_exc()
        

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process folders to convert quaternion data to H5 format')
    parser.add_argument('root_folder', help='Input folders to process')
    parser.add_argument('out_folder', help='Output folder to write the episodes')

    # Parse arguments
    args = parser.parse_args()
    
    # Process the folders
    process_folders(args.root_folder, args.out_folder)

if __name__ == '__main__':
    main()