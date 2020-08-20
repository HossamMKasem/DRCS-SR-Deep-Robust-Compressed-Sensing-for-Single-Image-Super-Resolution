from __future__ import print_function
import argparse
import torch
from modules.utils import utils
from trainer import Trainer

# Training settings
parser = argparse.ArgumentParser(description='PyTorch GTSRB example')
parser.add_argument('--data', type=str, default='/data/hossamkasem/13-CS/2-Trained_Network/3-Super_resolution/1-R/7-EDSR/1-50', metavar='D',
                    help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
# SGD should use lr = 0.01
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--dropout', type=float, default=0.5, metavar='D',
                    help='Dropout rate (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--patience', type=int, default=10, metavar='P',
                    help='patience for early stopping (default: 10)')
parser.add_argument('--weight', type=float, default=0, metavar='W',
                    help='Weight decay for adam optimizer (default: 0)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--train_pickle', type=str, default="./train.p", metavar='TP',
                    help="Pickle containing training features and labels")
parser.add_argument('--cnn', type=str, default=None, metavar='C',
                    help="Number of filters per CNN layer")
parser.add_argument('--resume', type=str, default=None, metavar='MO',
                    help="Location of model file if present")
parser.add_argument('--locnet', type=str, default="200,300,200", metavar='LN',
                    help="Number of filters per CNN layer")
parser.add_argument('--locnet2', type=str, default=None, metavar='LN2',
                    help="Number of filters per CNN layer")
parser.add_argument('--locnet3', type=str, default=None, metavar='LN3',
                    help="Number of filters per CNN layer")
parser.add_argument('--st', action='store_true',
                    help="Specifies if we want to use spatial transformer networks")
parser.add_argument('--extra_debug', action='store_true',
                    help="Use for printing more debugging information (default: false)")
parser.add_argument('--no_use_pickle', action='store_true',
                    help="Specifies if a pickle file is not to be used. If passed pickle argument is ignored and data is used instead. (Default: False)")
parser.add_argument('--save_loc', type=str, default=".", help="Location to save model")


def main():
    params = parser.parse_args()
    cuda = torch.cuda.is_available()
    torch.manual_seed(params.seed)
    print(params)
    params.use_pickle = not params.no_use_pickle
    torch.cuda.manual_seed(params.seed)
    train_dataset=torch.load('/data/hossamkasem/13-CS/1-Dataset/1-Transformed/4-Pytorch_dataset/1-comverting_bin_images/1-R/1-50/Original_Train_image.pt')
    train_dataset_label=torch.load('/data/hossamkasem/13-CS/1-Dataset/1-Transformed/4-Pytorch_dataset/1-comverting_bin_images/1-R/1-50/Original_Train_label_image.pt')
    val_dataset = torch.load('/data/hossamkasem/13-CS/1-Dataset/1-Transformed/4-Pytorch_dataset/1-comverting_bin_images/1-R/1-50/Original_Val_image.pt')
    val_dataset_label=torch.load('/data/hossamkasem/13-CS/1-Dataset/1-Transformed/4-Pytorch_dataset/1-comverting_bin_images/1-R/1-50/Original_val_label_image.pt')
    #intializing
    trainer = Trainer(params, train_dataset,train_dataset_label, val_dataset,val_dataset_label)
    trainer.load()
    trainer.train()


if __name__ == '__main__':
    main()
