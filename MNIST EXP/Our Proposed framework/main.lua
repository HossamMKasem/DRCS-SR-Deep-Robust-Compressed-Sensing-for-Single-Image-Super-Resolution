require 'torch'
require 'nn'
local gtsrb = require 'gtsrb'
local lapp = require 'pl.lapp'

torch.setdefaulttensortype('torch.FloatTensor')

local opt = lapp [[
GTSRB Training script
Main options
  -o,--output   (default "Trained_network.bin")              Output model
  --eval                                  Only run eval
  --val                                   Use a validation set instead of the test set
  --script                                Write accuracy on last line of output for benchmarks
  --no_cuda                               Do not use CUDA
  -n            (default 60000)           Use only N samples for training
  -e            (default 100)              Number of epochs
  --Number_RB    (default 3)               Number_RB

Network generation
  --cnn         (default "1,1,100")   Basic network factory parameters (see doc for more details)
  --net         (default "")              Path to user defined module for networks
  --ms                                    Use multi-scale network

Normalization
  --no_norm                               Do not globally normalize the training and test samples
  --no_lnorm                              Do not locally normalize the training and test samples
  --no_cnorm                              Do not use contrastive normalization in conv

Learning hyperparameters
  -s,--seed     (default 1)               Random seed value (-1 to disable)
  -b,--bs       (default 50)              Mini batch size
  --lr          (default 0.01)            Initial learning rate
  --lrd         (default 0)               Learning rate decay
  --wd          (default 0)               Weight decay
  --mom         (default 0)               Momentum

Spatial transformer options
  --st                                    Add a spatial transformer module
  --locnet      (default "30,60,30")      Localization network parameters
  --locnet2     (default "")              Localization network parameters for second st
  --locnet3     (default "")              Localization network parameters for third st (idsia net only)
  --rot                                   Force the st to use rotation
  --sca                                   Force the st to use scale
  --tra                                   Force the st to use translation
]]

if opt.seed > 0 then
  torch.manualSeed(opt.seed)
  math.randomseed(opt.seed)
end
if not opt.no_cuda then
  require 'cunn'
end

print("Loading training data...")
local train_dataset, test_dataset
--[[
if opt.val then
  -- In this case our variable 'test_dataset' contains the validation set
  train_dataset, test_dataset = gtsrb.dataset.get_train_dataset(opt.n, true)
else
  train_dataset = gtsrb.dataset.get_train_dataset(opt.n)
  test_dataset = gtsrb.dataset.get_test_dataset()
end
]]

train_dataset=torch.load('/data/hossamkasem/13-CS/1-Dataset/1-Transformed/1-MNIST/1-R/1-0.5/combined_train_half_initial.bin')
test_dataset=torch.load('/data/hossamkasem/13-CS/1-Dataset/1-Transformed/1-MNIST/1-R/1-0.5/combined_test_half_initial.bin')
print(opt)
print(train_dataset)
print(test_dataset)

--[[
local mean, std
if not opt.no_norm then
  print('Performing global normalization...')
  mean, std = gtsrb.dataset.normalize_global(train_dataset)
  gtsrb.dataset.normalize_global(test_dataset, mean, std)
end

if not opt.no_lnorm then
  print('Performing local normalization...')
  gtsrb.dataset.normalize_local(train_dataset)
  gtsrb.dataset.normalize_local(test_dataset)
end
]]


local network
require 'stn'
network=torch.load('/data/hossamkasem/13-CS/2-Trained_Network/1-Mnist/1-Transformation /1-R/1-50/1-Our_CS/Saved_network/Trained_Network_20.bin')
--[[
if opt.eval then
  if opt.output then
    print('Loading network from '..opt.output)
    if opt.st then
      require 'stn'
    end
    network = torch.load(opt.output)
  else
    error('Must supply the network to use in eval mode')
  end
else
  print('Building the network...')
  network = gtsrb.networks.new(opt)
end
]]
local criterion = nn.CrossEntropyCriterion()
if not opt.no_cuda then
  network = network:cuda()
  criterion = criterion:cuda()
end
print(network)

print("Initializing the trainer...")
gtsrb.trainer.initialize(network, criterion, opt)

local _, accuracy

if opt.eval then
  _, accuracy = gtsrb.trainer.test(test_dataset)
else
  local epoch = 1
  while opt.e == -1 or opt.e >= epoch do
    print('Starting epoch '..epoch)

    gtsrb.trainer.train(train_dataset,epoch)

    _, accuracy = gtsrb.trainer.test(test_dataset,epoch)
	print(accuracy)

    if opt.output ~= '' then
     torch.save('/data/hossamkasem/13-CS/2-Trained_Network/1-Mnist/1-Transformation /1-R/1-50/1-Our_CS/Saved_network/Trained_Network_'..epoch..'.bin', network)
      torch.save(opt.output.."norm", {mean, std})
	  torch.save('/data/hossamkasem/13-CS/2-Trained_Network/1-Mnist/1-Transformation /1-R/1-50/1-Our_CS/Saved_network/accuracy_'..epoch..'.bin', accuracy)
    end
    epoch = epoch + 1
  end
  
end

if opt.script then
  print(accuracy)
end
