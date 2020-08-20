import argparse, os
import torch
#import pickle
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision




print("===> Loading Saved_Model")
model=torch.load('Saved_Model.pt')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
model = model.to(device)
print (model)
print("===> Loading datasets")
Total_test_data=torch.load('/data/hossamkasem/12-Tests_code/Pytorch/Test_Dataset/R/X2/Set5/Dataset_Mat_file_To_PT/Test_Data_dataset.pt')
print(len(Total_test_data))
Label_dataset=torch.load('/data/hossamkasem/12-Tests_code/Pytorch/Test_Dataset/R/X2/Set5/Dataset_Mat_file_To_PT/Test_Label_dataset.pt')
train_ds=TensorDataset(Total_test_data,Label_dataset)
training_data_loader = DataLoader(dataset=train_ds, num_workers=1, batch_size=20, shuffle=False)
print(len(training_data_loader))

model.eval()

for iteration, batch in enumerate(training_data_loader, 1):
        input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input = input.to(device)
        target = target.to(device)
        Output=model(input)
        torch.save(Output,'/data/hossamkasem/12-Tests_code/Pytorch/DFTN_Pytorch/Set5_Output_Testing_Results/PT_Files/Output/Output_batch_{}.pt'.format(iteration))
        torch.save(input,'/data/hossamkasem/12-Tests_code/Pytorch/DFTN_Pytorch/Set5_Output_Testing_Results/PT_Files/Input/input_batch_{}.pt'.format(iteration))
        torch.save(target,'/data/hossamkasem/12-Tests_code/Pytorch/DFTN_Pytorch/Set5_Output_Testing_Results/PT_Files/Label/target_batch_{}.pt'.format(iteration))


Length=len(training_data_loader)
counter=0
for ii in range(Length):
    temp_Output=torch.load('/data/hossamkasem/12-Tests_code/Pytorch/DFTN_Pytorch/Set5_Output_Testing_Results/PT_Files/Output/Output_batch_{}.pt'.format(ii+1))
    temp_Input=torch.load('/data/hossamkasem/12-Tests_code/Pytorch/DFTN_Pytorch/Set5_Output_Testing_Results/PT_Files/Input/input_batch_{}.pt'.format(ii+1))
    temp_target=torch.load('/data/hossamkasem/12-Tests_code/Pytorch/DFTN_Pytorch/Set5_Output_Testing_Results/PT_Files/Label/target_batch_{}.pt'.format(ii+1))
    Length_2=len(temp_Output)
    for iii in range(Length_2):
        counter=counter+1
        torchvision.utils.save_image (temp_Input[iii],'/data/hossamkasem/12-Tests_code/Pytorch/DFTN_Pytorch/Set5_Output_Testing_Results/Images/Input/input_{}.png'.format(counter))
        torchvision.utils.save_image (temp_Output[iii],'/data/hossamkasem/12-Tests_code/Pytorch/DFTN_Pytorch/Set5_Output_Testing_Results/Images/Outtput/output_{}.png'.format(counter))
        torchvision.utils.save_image (temp_target[iii],'/data/hossamkasem/12-Tests_code/Pytorch/DFTN_Pytorch/Set5_Output_Testing_Results/Images/Label/label_{}.png'.format(counter))
        print(counter)		



		
















