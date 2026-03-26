
from torch.utils import data
from torchvision import transforms as T
import os
import numpy as np
import SimpleITK as sitk
from utils import tsfm_tfusion
from pathlib import Path

class Brain(data.Dataset):
    def __init__(self, data_file, selected_modal, base_dir, inputs_transform=None,
                 labels_transform=None, t_join_transform=None, join_transform=None, phase='train',test_modals=None):
        self.selected_modal = selected_modal
        self.c_dim = len(self.selected_modal)
        self.inputs_transform = inputs_transform
        self.labels_transform = labels_transform
        self.join_transform = join_transform
        self.t_join_transform = t_join_transform
        self.data_file = data_file
        self.dataset = {}
        self.phase = phase
        self.base_dir = base_dir
        self.test_modals = test_modals
        self.init()

    def init(self):

        self.dataset['data'] = []
        lines = [line.rstrip() for line in open(self.data_file, 'r')]
        flag_m = 0
        converted_md = 0
        if self.phase == 'test':
            for modal in self.test_modals:
                if modal=='t1c':
                    converted_md+=1
                if modal == 't1n':
                    converted_md+=2
                if modal == 't2w':
                    converted_md +=4
                if modal == 't2f':
                    converted_md+=8
        for i, sub in enumerate(lines):

            sub_path = Path(self.base_dir) / sub

            flag_m += 1
            flag_m %= 15

            if self.phase == 'test':
                self.dataset['data'].append([sub_path, sub,  converted_md])
            else:
                self.dataset['data'].append([sub_path, sub, flag_m % 15 + 1])

        print('[*] Load {}, which contains {} paired volumes with random missing modilities, {}'.format(self.data_file,
                                                                                       len(self.dataset['data']),
                                                                                       self.selected_modal))
    def __getitem__(self, idex):

        sub_path, pid, m_d = self.dataset['data'][idex]

        label_path = sub_path / (sub_path.name + '-seg.nii.gz') 
        volume_label = sitk.GetArrayFromImage(sitk.ReadImage(label_path)).astype(np.float32)
        unprocessed_volume_label = volume_label.copy()

        volumes = []
        crop_size = None
        for modal in self.selected_modal:
            m_path = sub_path / (sub_path.name + f'-{modal}.nii.gz')
            volumes.append(sitk.GetArrayFromImage(sitk.ReadImage(m_path)).astype(np.float32))

        if self.join_transform:
            volumes, volume_label, crop_size = self.join_transform(volumes, volume_label, self.phase)
        if self.t_join_transform:
            volumes, volume_label, _ = self.t_join_transform(volumes, volume_label, self.phase)

        if self.inputs_transform:
            volumes[0] = self.inputs_transform(volumes[0]) #t1c
            volumes[1] = self.inputs_transform(volumes[1]) #t1n
            volumes[2] = self.inputs_transform(volumes[2]) #t2w
            volumes[3] = self.inputs_transform(volumes[3]) #t2f

        if self.labels_transform:
            volume_label = self.labels_transform(volume_label)

        return volumes[0], volumes[1], volumes[2], volumes[3], \
               volume_label, pid, m_d, crop_size, unprocessed_volume_label

    def __len__(self):
        return len(self.dataset['data'])

def get_loaders(base_dir,data_files, selected_modals, batch_size=1, num_workers=0,test_modals=['t1c']):
    rs = np.random.RandomState(1234)
    join_tsfm = tsfm_tfusion.Compose([
        tsfm_tfusion.ThrowFirstZ(),
        tsfm_tfusion.RandomCrop(128)
    ])
    train_join_tsfm = tsfm_tfusion.Compose([
        tsfm_tfusion.RandomFlip(rs),
        tsfm_tfusion.RandomRotate(rs, angle_spectrum=10),
    ])
    input_tsfm = T.Compose([
        tsfm_tfusion.Normalize(),
        tsfm_tfusion.NpToTensor()
    ])
    label_tsfm = T.Compose([
        tsfm_tfusion.ToLongTensor()
    ])


    datasets = dict(train=Brain(data_files['train'], selected_modals, base_dir, inputs_transform=input_tsfm,
                        labels_transform=label_tsfm, t_join_transform=train_join_tsfm, join_transform=join_tsfm, phase='train'),
                    val=Brain(data_files['val'], selected_modals, base_dir, inputs_transform=input_tsfm,
                        labels_transform=label_tsfm, t_join_transform=None, join_transform=join_tsfm, phase='val'),
                    test=Brain(data_files['test'], selected_modals, base_dir, inputs_transform=input_tsfm,
                        labels_transform=label_tsfm, t_join_transform=None, join_transform=join_tsfm, phase='test',test_modals=test_modals)
                    )
    loaders = {x: data.DataLoader(dataset=datasets[x], batch_size=batch_size,
                                  shuffle=(x == 'train'),
                                  num_workers=num_workers)
               for x in ('train', 'val', 'test')}
    return loaders

