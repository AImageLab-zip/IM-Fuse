import os
import shutil
from argparse import ArgumentParser
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def copy_one_file(args):
    input_path, output_path = args
    shutil.copy(input_path,output_path)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-path',type=str,required=True)
    parser.add_argument('--output-path',type=str,required=True)
    args=parser.parse_args()
    
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    print('File is the --input-path will NOT be deleted.')
    if os.path.exists(output_path):
        print(f'--output-path --> {output_path} already exists.')
        confirm = input(f"Are you sure you want to delete it? (y/n): ").lower()
        if confirm == 'y':
            shutil.rmtree(output_path)
            print(f'Proceeding with dataset restructuring...')
        else:
            print('Deletion cancelled. Quitting the program.')
            exit()

    datalist = Path(os.path.abspath(__file__)).parent / 'datalist'
    with open(datalist/'train.txt') as f:
        train_subs = [line.strip() for line in f.readlines()]
    with open(datalist/'val.txt') as f:
        val_subs = [line.strip() for line in f.readlines()]
    with open(datalist/'test.txt') as f:
        test_subs = [line.strip() for line in f.readlines()]

    train_val_subs = train_subs + val_subs

    dataset138 = output_path/'Dataset138_MaM'
    train_images = dataset138/'imagesTr'
    train_labels = dataset138/'labelsTr'
    test_images = dataset138/'imagesTs'
    test_labels = dataset138/'labelsTs'

    os.makedirs(train_images)
    os.makedirs(train_labels)
    os.makedirs(test_images)
    os.makedirs(test_labels)

    dataset_json = Path(os.path.abspath(__file__)).parent / 'dataset.json'
    shutil.copy(dataset_json,dataset138/'dataset.json')
    input_modals = [prefix + '.nii.gz' for prefix in ['t2f','t1n','t1c','t2w']]
    output_modals = [prefix + '.nii.gz' for prefix in ['0000','0001','0002','0003']]

    input_train_val_images = [input_path / sub / (sub + '-' + modal) for sub in train_val_subs for modal in input_modals]
    output_train_val_images = [train_images / (sub + '_' + modal) for sub in train_val_subs for modal in output_modals]

    input_test_images = [input_path / sub / (sub + '-' + modal) for sub in test_subs for modal in input_modals]
    output_test_images = [test_images / (sub + '_' + modal) for sub in test_subs for modal in output_modals]

    input_train_val_labels = [input_path/sub/(sub+'-seg.nii.gz')for sub in train_val_subs]
    output_train_val_labels = [train_labels/(sub + '.nii.gz') for sub in train_val_subs]

    input_test_labels = [input_path/sub/(sub+'-seg.nii.gz')for sub in test_subs]
    output_test_labels = [test_labels/(sub + '.nii.gz') for sub in test_subs]

    all_files = list(zip(input_train_val_images,output_train_val_images)) + \
                list(zip(input_test_images,output_test_images)) + \
                list(zip(input_train_val_labels,output_train_val_labels)) + \
                list(zip(input_test_labels,output_test_labels))
    
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(copy_one_file,all_files),total=len(all_files),desc='Copying dataset'))

