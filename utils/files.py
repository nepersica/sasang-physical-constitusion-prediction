import os
import shutil
from pathlib import Path
import glob
from tqdm import tqdm

def save_filename():
    data_path = './dataset/v2/image'
    data_list = os.listdir(data_path)
    
    num_train = int(len(data_list)*0.8)
    num_rest = len(data_list)-num_train

    if num_rest % 2 == 1:
        num_val = int(num_rest/2)+1
        num_test = int(num_rest/2)
    else:
        num_val = int(num_rest/2)
        num_test = int(num_rest/2)

    data_train = data_list[:num_train]
    data_val = data_list[num_train:num_train+num_val]
    data_test = data_list[num_train+num_val:]
         
    train_file = open("./dataset/v2/train.txt", "w+")
    for data in data_train:
        train_file.write(f'{data}\n')
    
    val_file = open("./dataset/v2/val.txt", "w+")
    for data in data_val:
        val_file.write(f'{data}\n')
        
    test_file = open("./dataset/v2/test.txt", "w+")
    for data in data_test:
        test_file.write(f'{data}\n')
    

def rename_subdir(pose_dir_path):
    direction_dirs = os.listdir(pose_dir_path)
    for direction_dir in direction_dirs:
        if direction_dir == '1.앞':
            os.rename(os.path.join(pose_dir_path, direction_dir), os.path.join(pose_dir_path, '1.front'))
        elif direction_dir == '2.앞좌':
            os.rename(os.path.join(pose_dir_path, direction_dir), os.path.join(pose_dir_path, '2.front-left'))
        elif direction_dir == '3.좌':
            os.rename(os.path.join(pose_dir_path, direction_dir), os.path.join(pose_dir_path, '3.left'))
        elif direction_dir == '4.뒤좌':
            os.rename(os.path.join(pose_dir_path, direction_dir), os.path.join(pose_dir_path, '4.back-left'))
        elif direction_dir == '5.뒤':
            os.rename(os.path.join(pose_dir_path, direction_dir), os.path.join(pose_dir_path, '5.back'))
        elif direction_dir == '6.뒤우':
            os.rename(os.path.join(pose_dir_path, direction_dir), os.path.join(pose_dir_path, '6.back-right'))
        elif direction_dir == '7.우':
            os.rename(os.path.join(pose_dir_path, direction_dir), os.path.join(pose_dir_path, '7.right'))
        elif direction_dir == '8.앞우':
            os.rename(os.path.join(pose_dir_path, direction_dir), os.path.join(pose_dir_path, '8.fron-right'))

def rename_dirs():
    dataset_dirs = os.listdir('./dataset/v2/image')
    
    for data_dir in dataset_dirs:
        data_path = os.path.join('./dataset/v2/image', data_dir)
        pose_dirs = os.listdir(data_path)
        
        for pose_dir in pose_dirs:
            pose_dir_path = os.path.join('./dataset', data_dir, 'Image', pose_dir)
            if os.path.isdir(pose_dir_path):
                rename_subdir(pose_dir_path)
                if pose_dir == '1.차렷':
                    os.rename(pose_dir_path, os.path.join(data_path, '1.attention'))
                elif pose_dir == '2.A자':
                    os.rename(pose_dir_path, os.path.join(data_path, '2.A'))
                elif pose_dir == '3.T자':
                    os.rename(pose_dir_path, os.path.join(data_path, '3.T'))
                elif pose_dir == '4.Y자':
                    os.rename(pose_dir_path, os.path.join(data_path, '4.Y'))
                elif pose_dir == '5.앞으로':
                    os.rename(pose_dir_path, os.path.join(data_path, '5.forward'))
                elif pose_dir == '6.걷기':
                    os.rename(pose_dir_path, os.path.join(data_path, '6.walk'))

def move_label_files():
    image_path = './dataset/v2/origin_image'
    image_dest_path = './dataset/v2/image'
    dir_list = os.listdir(image_path)
    
    json_dest_path = './dataset/v2/label'
    json_path = './dataset/v1/label'
    
    outlier_file = open("./dataset/json outlier.txt", "w+")
    for dir in tqdm(dir_list):
        dir_list = os.listdir(os.path.join(image_path, dir))
        if not os.path.isdir(os.path.join(json_dest_path, dir)):
            os.mkdir(os.path.join(json_dest_path, dir))
        if not os.path.isdir(os.path.join(image_dest_path, dir)):
            os.mkdir(os.path.join(image_dest_path, dir))    
        
        for image in dir_list:
            json = image[:-4]
            depart_path = os.path.join(json_path, dir, 'json')
            
            try:
                shutil.copy(os.path.join(depart_path, json+'.json'), os.path.join(json_dest_path, dir, json+'.json'))
                shutil.copy(os.path.join(image_path, dir, image), os.path.join(image_dest_path, dir, image))
            except:
                outlier_file.write(f"{os.path.join(depart_path, json+'.json')}\n")
                # print(f"{os.path.join(depart_path, json+'.json')} is not exist")
        
        if len(os.listdir(os.path.join(json_dest_path, dir))) == 0:
            shutil.rmtree(os.path.join(json_dest_path, dir))
            shutil.rmtree(os.path.join(image_dest_path, dir))

if __name__ == '__main__':
    # rename_dirs()
    # move_label_files()
    save_filename()

