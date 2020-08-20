require 'nn'
require 'optim'
require 'sys'

-- Load Facebook optim package
paths.dofile('Optim.lua')


local trainer = {}

-- This function should be called before any other on the trainer package.
-- Takes as input a torch network, a criterion and the options of the training
function trainer.initialize(network, criterion, options)
  local optim_state = {
      learningRate = options.lr,
      momentum = options.mom,
      learningRateDecay = options.lrd,
      weightDecay = options.wd,
    }

  trainer.tensor_type = torch.getdefaulttensortype()
  if not options.no_cuda then
    trainer.tensor_type = 'torch.CudaTensor'
  end
  trainer.batch_size = options.bs
  trainer.network = network
  if criterion then
    trainer.criterion = criterion
    trainer.optimizer = nn.Optim(network, optim_state)
  end
end

-- Main training function.
-- This performs one epoch of training on the network given during
-- initialization using the given dataset.
-- Returns the mean error on the dataset.
function trainer.train(dataset,epoch)
  if not trainer.optimizer then
    error('Trainer not initialized properly. Use trainer.initialize first.')
  end
  -- do one epoch
  print('<trainer> on training set:')
  local epoch_error = 0
  local nbr_samples = dataset.data:size(1)
  local size_samples = dataset.data:size()[dataset.data:dim()]
  local time = sys.clock()

  -- generate random training batches
  local indices = torch.randperm(nbr_samples):long():split(trainer.batch_size)
  indices[#indices] = nil -- remove last partial batch

  -- preallocate input and target tensors
  local inputs = torch.zeros(trainer.batch_size, 1,
                                    size_samples, size_samples,
                                    trainer.tensor_type)
  local targets = torch.zeros(trainer.batch_size, 1,
                                      trainer.tensor_type)
   							  

  for t,ind in ipairs(indices) do
    -- get the minibatch
    inputs:copy(dataset.data:index(1,ind))
    targets:copy(dataset.label:index(1,ind))
	torch.save('ind.bin',ind)
	

    epoch_error = epoch_error + trainer.optimizer:optimize(optim.sgd,
                                  inputs,
                                  targets,
                                  trainer.criterion)

    -- disp progress
    xlua.progress(t*trainer.batch_size, nbr_samples)
  end
  -- finish progress
  xlua.progress(nbr_samples, nbr_samples)

  -- time taken
  time = sys.clock() - time
  time = time / nbr_samples
  print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')
  print("<trainer> mean MSE  error (train set) = " .. epoch_error/nbr_samples)
  mean_Time=torch.zeros(epoch)
  mean_MSE_error_Train=torch.zeros(epoch)
  mean_Time[epoch]= time*1000
  mean_MSE_error_Train[epoch]=epoch_error/nbr_samples
  torch.save('/data/hossamkasem/13-CS/2-Trained_Network/1-Mnist/1-Transformation /1-R/1-50/1-Our_CS/MSE/Time_Train/mean_MSE_error_Train'..epoch..'.bin',mean_Time)
  torch.save('/data/hossamkasem/13-CS/2-Trained_Network/1-Mnist/1-Transformation /1-R/1-50/1-Our_CS/MSE/MSE_Train/mean_MSE_error_Train'..epoch..'.bin',epoch_error/nbr_samples)
  torch.save('/data/hossamkasem/13-CS/2-Trained_Network/1-Mnist/1-Transformation /1-R/1-50/1-Our_CS/MSE/MSE_Train/mean_MSE_error_Train_total.bin',mean_MSE_error_Train)
  return epoch_error
end

-- Main testing function.
-- This performs a full test on the given dataset using the network
-- given during the initialization.
-- Returns the mean error on the dataset and the accuracy.
function trainer.test(dataset,epoch)
  if not trainer.network then
    error('Trainer not initialized properly. Use trainer.initialize first.')
  end
  -- test over given dataset
  print('')
  print('<trainer> on testing Set:')
  local time = sys.clock()
  local nbr_samples = dataset.data:size(1)
  local size_samples = dataset.data:size()[dataset.data:dim()]
  local epoch_error = 0
  local correct = 0
  local all = 0

  -- generate indices and split them into batches
  local indices = torch.range(1,nbr_samples):long()
  indices = indices:split(trainer.batch_size)

  -- preallocate input and target tensors
  local inputs = torch.zeros(trainer.batch_size, 1,
                                    size_samples, size_samples,
                                    trainer.tensor_type)
  local targets = torch.zeros(trainer.batch_size, 1,
                                      trainer.tensor_type)

 
  for t,ind in ipairs(indices) do
    -- last batch may not be full
    local local_batch_size = ind:size(1)
    -- resize prealocated tensors (should only happen on last batch)
    inputs:resize(local_batch_size,1,size_samples,size_samples)
    targets:resize(local_batch_size, 1)

    inputs:copy(dataset.data:index(1,ind))
    targets:copy(dataset.label:index(1,ind))
    
    -- test samples
    local scores = trainer.network:forward(inputs)
    epoch_error = epoch_error + trainer.criterion:forward(scores,
                                  targets)

    local _, preds = scores:max(2)
    correct = correct + preds:float():eq(targets:float()):sum()
    all = all + preds:size(1)
    
    -- disp progress
    xlua.progress(t*trainer.batch_size, nbr_samples)
  end
  -- finish progress
  xlua.progress(nbr_samples, nbr_samples)
   local accuracy = correct / all
  -- timing
  time = sys.clock() - time
  time = time / nbr_samples
  print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')
  print("<trainer> mean MSE error (te st set) = " .. epoch_error/nbr_samples)
  mean_MSE_error_Test=torch.zeros(epoch)
  mean_Time_Test=torch.zeros(epoch)
  mean_Time_Test[epoch]=time*1000
  mean_MSE_error_Test[epoch]=epoch_error/nbr_samples
  torch.save('/data/hossamkasem/13-CS/2-Trained_Network/1-Mnist/1-Transformation /1-R/1-50/1-Our_CS/MSE/Time_Test/mean_Time_Test'..epoch..'.bin',mean_Time_Test)
  torch.save('/data/hossamkasem/13-CS/2-Trained_Network/1-Mnist/1-Transformation /1-R/1-50/1-Our_CS/MSE/MSE_Test/mean_MSE_error_Test'..epoch..'.bin',epoch_error/nbr_samples)
  torch.save('/data/hossamkasem/13-CS/2-Trained_Network/1-Mnist/1-Transformation /1-R/1-50/1-Our_CS/MSE/MSE_Test/mean_MSE_error_Test_total.bin',mean_MSE_error_Test)
   return epoch_error, accuracy
end

return trainer
