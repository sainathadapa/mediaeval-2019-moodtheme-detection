import os, pickle, csv
import numpy as np
import torch
from torch.utils import data
from transforms import get_transforms

def get_spectogram(full_spect, size=1400):
    full_len = full_spect.shape[1]
    if full_len > size:
        audio = full_spect[:, :size]
    else:
        diff = size-full_len
        audio = full_spect
        while(diff > 0):
            if diff>full_len:
                audio = np.concatenate((audio,full_spect), axis=1)
                diff = diff-full_len
            else:
                audio = np.concatenate((audio, full_spect[:,:diff]), axis=1)
                diff = 0                
    return audio

class AudioFolder(data.Dataset):
    def __init__(self, root, tsv_path, labels_to_idx, num_classes=56, spect_len=4096, train=True):
        self.train = train
        self.root = root
        self.num_classes = num_classes
        self.spect_len = spect_len
        self.labels_to_idx = labels_to_idx
        self.prepare_data(tsv_path)
        if train:
            self.transform = get_transforms(
                                train=True,
                                size=spect_len,
                                wrap_pad_prob=0.5,
                                resize_scale=(0.8, 1.0),
                                resize_ratio=(1.7, 2.3),
                                resize_prob=0.33,
                                spec_num_mask=2,
                                spec_freq_masking=0.15,
                                spec_time_masking=0.20,
                                spec_prob=0.5
                            )
        else:
            self.transform = get_transforms(False, spect_len)

    def __getitem__(self, index):
        fn = os.path.join(self.root, self.paths[index][:-3]+'npy')
        full_spect = np.array(np.load(fn))
        audio = get_spectogram(full_spect, self.spect_len)
        audio = self.transform(audio)
        tags = self.tags[index]
        labels = self.one_hot(tags)
        return audio, labels

    def __len__(self):
        return len(self.paths)
    
    def one_hot(self, tags):
        labels = torch.LongTensor(tags)
        target = torch.zeros(self.num_classes).scatter_(0, labels, 1)
        return target
    
    def prepare_data(self, path_to_tsv):
    
        all_dict = {
        'PATH': [],
        'TAGS': []
        }
        with open(path_to_tsv) as tsvfile:
            tsvreader = csv.reader(tsvfile, delimiter="\t")
            next(tsvreader) #Reading the first line
            for line in tsvreader:
                all_dict['PATH'].append(line[3])
                all_dict['TAGS'].append(line[5:])

        self.paths = all_dict['PATH']
        self.tags = [[self.labels_to_idx[j] for j in i] for i in all_dict['TAGS']]



def get_audio_loader(root, tsv_path, labels_to_idx, batch_size=16, num_workers=4, shuffle=True, drop_last=True):
    data_loader = data.DataLoader(dataset=AudioFolder(root, tsv_path, labels_to_idx, num_classes=56, train=True),
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  drop_last=drop_last)
    return data_loader