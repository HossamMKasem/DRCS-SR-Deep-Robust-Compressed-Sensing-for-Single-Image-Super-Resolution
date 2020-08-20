from __future__ import print_function
import zipfile
import os
import torch
import torchvision.transforms as transforms
from transforms import preprocess_img, RandomAffineTransform
from torch.utils.data import Dataset


class TrafficSignsDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.from_numpy(images)
        self.images = self.images.permute(0, 3, 1, 2)
        self.labels = torch.LongTensor(labels.argmax(1))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


def initialize_data(folder):
    train_zip = folder + '/train_images.zip'
    test_zip = folder + '/test_images.zip'
    if not os.path.exists(train_zip) or not os.path.exists(test_zip):
        raise(RuntimeError("Could not find " + train_zip + " and " + test_zip
                           + ', please download them from https://www.kaggle.com/c/nyu-cv-fall-2017/data '))
    # extract train_data.zip to train_data
    train_folder = folder + '/train_images'
    if not os.path.isdir(train_folder):
        print(train_folder + ' not found, extracting ' + train_zip)
        zip_ref = zipfile.ZipFile(train_zip, 'r')
        zip_ref.extractall(folder)
        zip_ref.close()
    # extract test_data.zip to test_data
    test_folder = folder + '/test_images'
    if not os.path.isdir(test_folder):
        print(test_folder + ' not found, extracting ' + test_zip)
        zip_ref = zipfile.ZipFile(test_zip, 'r')
        zip_ref.extractall(folder)
        zip_ref.close()

    # make validation_data by using images 00000*, 00001* and 00002* in each class
    val_folder = folder + '/val_images'
    if not os.path.isdir(val_folder):
        print(val_folder + ' not found, making a validation set')
        os.mkdir(val_folder)
        for dirs in os.listdir(train_folder):
            if dirs.startswith('000'):
                os.mkdir(val_folder + '/' + dirs)
                for f in os.listdir(train_folder + '/' + dirs):
                    if f.startswith('00000') or f.startswith('00001') or f.startswith('00002'):
                        # move file to validation folder
                        os.rename(train_folder + '/' + dirs + '/' + f,
                                  val_folder + '/' + dirs + '/' + f)


def decrease_data(folder, num_max):
    for dirs in os.listdir(folder):
        if os.path.isdir(folder + '/' + dirs):
            curr_max = len(os.listdir(folder + '/' + dirs)) * num_max // 100
            for i, f in enumerate(os.listdir(folder + '/' + dirs)):
                if i > curr_max:
                    print(folder + '/' + dirs + '/' + f)
                    os.remove(folder + '/' + dirs + '/' + f)
