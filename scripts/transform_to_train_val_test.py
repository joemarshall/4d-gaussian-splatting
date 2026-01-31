import os
from pathlib import Path
import json
import random
import shutil

datasets_names = ['cloth', 'pendulums', 'robot', 'robot-task', 'spring', 'wheel']
fps = 30.0
for datasetname in datasets_names:
    source = Path("./data/particle_nerf/" + datasetname)
    
    file_list = os.listdir(source)
    train_split = 20 # Set the dataset split ratios
    test_split = 10
    val_split = 10
    random.seed(42)

    linked_fp_flag = ['_depth', '_normal'] # if images have linked depth and normal renders they'll usually be denotes e.g. as 'r_0_depth.png' or similar
                                        # We want to move these files as well...


    # assert (train_split + test_split + val_split == 1.), 'Make sure data split adds up to 1.'
    assert file_list != [], 'File list empty - make sure correct folder is selected.'
    assert os.path.exists(source/'transforms.json'), 'No transforms.json file found in directory'

    # Create image folders
    train_path = source / 'train'
    test_path = source / 'test'
    val_path = source / 'val'

    # Remove folders if they already exist (WARNING!!)
    if os.path.exists(train_path):
        os.rmdir(train_path)
    if os.path.exists(test_path):
        os.rmdir(test_path)
    if os.path.exists(val_path):
        os.rmdir(val_path)
    # Make folder in root
    os.mkdir(test_path)
    os.mkdir(train_path)
    os.mkdir(val_path)

    # Process data from 'transforms.json'
    with open(source/'transforms.json') as fp:
        contents = fp.read()
    meta = json.loads(contents) # load data as dict
    frames = meta['frames'] # get frame data
    dataset_size = len(frames) # get size of dataset
    random.shuffle(frames) # shuffle data


    # # Split shuffled data
    # train_idx = dataset_size
    # test_idx = int(test_split * dataset_size)
    # val_idx = test_idx + int(val_split * dataset_size)

    # # Ensure we have atleast one image the val folder 
    # assert len(frames[test_idx:]) > 0, 'Ensure `val_split` is large enough to have atleast 1 image in dataset'

    # Define our seperate dataset
    train_frames = [frame for frame in frames if int(frame['file_path'].split('_')[1])<20]
    test_frames = [frame for frame in frames if int(frame['file_path'].split('_')[1]) >= 20 and int(frame['file_path'].split('_')[1]) < 30]
    val_frames = [frame for frame in frames if int(frame['file_path'].split('_')[1]) >= 30]

    def copy_files(targ_path, frame, file_list):
        image_id = frame['file_path'].split('/')[-1]
        fp_ = targ_path / (image_id+'.png')  # append end of file path to new image path

        shutil.copy(source/('images')/(image_id+'.png'), source/fp_) # move source image to destination
        
        depth_fp_ = targ_path / (image_id+'.depth.png')  # append end of file path to new image path

        shutil.copy(source/('images')/(image_id+'.depth.png'), source/depth_fp_) # move source image to destination

        # Also append linked images if they exist
        # for flag in linked_fp_flag:
        #     linked_fp = image_id + flag # substring of image name (e.g. 'r_0_depth' in 'r_0_depth_0250.png')
        #     linked_fp_ = [file_path for file_path in file_list if linked_fp in file_path][0] # get the whole string
        
        #     shutil.move(source/linked_fp_, source/targ_path/linked_fp_)
            
        return str(fp_)[:-4], str(depth_fp_)[:-4]

    # Process frames for training, testing and validation
    for frame in train_frames:
        frame['file_path'], frame['depth_path'] = copy_files(Path('train'), frame, file_list)
        frame['time'] = float(frame['file_path'].split('/')[-1].split('_')[0]) / fps

    for frame in test_frames:
        frame['file_path'], frame['depth_path'] = copy_files(Path('test'), frame, file_list)
        frame['time'] = float(frame['file_path'].split('/')[-1].split('_')[0]) / fps
        
    for frame in val_frames:
        frame['file_path'], frame['depth_path'] = copy_files(Path('val'), frame, file_list)
        frame['time'] = float(frame['file_path'].split('/')[-1].split('_')[0]) / fps

    # Set the transforms file format you desire
    train_tranform = {
                    #   'camera_angle_x': meta['camera_angle_x'],
                    #   'camera_angle_y': meta['camera_angle_y'],
                    'fl_x': meta['fl_x'], 
                    'fl_y': meta['fl_y'], 
                    'cx': meta['cx'], 
                    'cy': meta['cy'], 
                    'h': meta['h'], 
                    'w': meta['w'], 
                        'frames': train_frames
                    }
        
    test_tranform = {
                    #  'camera_angle_x': meta['camera_angle_x'],
                    #  'camera_angle_y': meta['camera_angle_y'],
                    'fl_x': meta['fl_x'], 
                    'fl_y': meta['fl_y'], 
                    'cx': meta['cx'], 
                    'cy': meta['cy'], 
                    'h': meta['h'], 
                    'w': meta['w'], 
                        'frames': test_frames
                    }

    val_tranform = {
                    # 'camera_angle_x': meta['camera_angle_x'],
                    # 'camera_angle_y': meta['camera_angle_y'],
                    'fl_x': meta['fl_x'], 
                    'fl_y': meta['fl_y'], 
                    'cx': meta['cx'], 
                    'cy': meta['cy'], 
                    'h': meta['h'], 
                    'w': meta['w'], 
                        'frames': val_frames
                    }

    # Remove transforms files if they exist
    if os.path.exists(source / 'transforms_train.json'):
        os.remove(source / 'transforms_train.json')
    if os.path.exists(source / 'transforms_test.json'):
        os.remove(source / 'transforms_test.json')
    if os.path.exists(source / 'transforms_val.json'):
        os.remove(source / 'transforms_val.json')

    # Write the seperate transforms files
    with open(source / 'transforms_train.json', 'w') as out_file:
        json.dump(train_tranform, out_file, indent=4)
        
    with open(source / 'transforms_test.json', 'w') as out_file:
        json.dump(test_tranform, out_file, indent=4)

    with open(source / 'transforms_val.json', 'w') as out_file:
        json.dump(val_tranform, out_file, indent=4)