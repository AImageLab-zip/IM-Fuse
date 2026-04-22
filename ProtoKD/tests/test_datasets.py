from ProtoKD.code.datasets import BraTS_Train

dataset = BraTS_Train(data_file_path='/work/grana_neuro/missing_modalities/BRATS2023_Training_protokd_npy')
print(f'Length of dataset: {len(dataset)}')
for element in dataset:
    print(element['name'])