--[[ That would be the license for Optim.lua
     BSD License

     For fbcunn software

     Copyright (c) 2014, Facebook, Inc. All rights reserved.

     Redistribution and use in source and binary forms, with or without modification,
     are permitted provided that the following conditions are met:

     * Redistributions of source code must retain the above copyright notice, this
     list of conditions and the following disclaimer.

     * Redistributions in binary form must reproduce the above copyright notice,
     this list of conditions and the following disclaimer in the documentation
     and/or other materials provided with the distribution.

     * Neither the name Facebook nor the names of its contributors may be used to
     endorse or promote products derived from this software without specific
     prior written permission.

     THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
     ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
     WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
     DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
     ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
     (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
     LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
     ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
     (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
     SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
]]

-- Copyright 2004-present Facebook. All Rights Reserved.

local pl = require('pl.import_into')()

-- from fblualib/fb/util/data.lua , copied here because fblualib is not rockspec ready yet.
-- deepcopy routine that assumes the presence of a 'clone' method in user
-- data should be used to deeply copy. This matches the behavior of Torch
-- tensors.
local function deepcopy(x)
    local typename = type(x)
    if typename == "userdata" then
        return x:clone()
    end
    if typename == "table" then
        local retval = { }
        for k,v in pairs(x) do
            retval[deepcopy(k)] = deepcopy(v)
        end
        return retval
    end
    return x
end

local Optim, parent = torch.class('nn.Optim')


-- Returns weight parameters and bias parameters and associated grad parameters
-- for this module. Annotates the return values with flag marking parameter set
-- as bias parameters set
function Optim.weight_bias_parameters(module)
    local weight_params, bias_params
    if module.weight then
        weight_params = {module.weight, module.gradWeight}
        weight_params.is_bias = false
    end
    if module.bias then
        bias_params = {module.bias, module.gradBias}
        bias_params.is_bias = true
    end
    return {weight_params, bias_params}
end

-- The regular `optim` package relies on `getParameters`, which is a
-- beastly abomination before all. This `optim` package uses separate
-- optim state for each submodule of a `nn.Module`.
function Optim:__init(model, optState, checkpoint_data)
    assert(model)
    assert(checkpoint_data or optState)
    assert(not (checkpoint_data and optState))

    self.model = model
    self.modulesToOptState = {}
    -- Keep this around so we update it in setParameters
    self.originalOptState = optState

    -- Each module has some set of parameters and grad parameters. Since
    -- they may be allocated discontinuously, we need separate optState for
    -- each parameter tensor. self.modulesToOptState maps each module to
    -- a lua table of optState clones.
    if not checkpoint_data then
        self.model:apply(function(module)
            self.modulesToOptState[module] = { }
            local params = self.weight_bias_parameters(module)
            -- expects either an empty table or 2 element table, one for weights
            -- and one for biases
            assert(pl.tablex.size(params) == 0 or pl.tablex.size(params) == 2)
            for i, _ in ipairs(params) do
                self.modulesToOptState[module][i] = deepcopy(optState)
                if params[i] and params[i].is_bias then
                    -- never regularize biases
                    self.modulesToOptState[module][i].weightDecay = 0.0
                end
            end
            assert(module)
            assert(self.modulesToOptState[module])
        end)
    else
        local state = checkpoint_data.optim_state
        local modules = {}
        self.model:apply(function(m) table.insert(modules, m) end)
        assert(pl.tablex.compare_no_order(modules, pl.tablex.keys(state)))
        self.modulesToOptState = state
    end
end

function Optim:save()
    return {
        optim_state = self.modulesToOptState
    }
end

local function _type_all(obj, t)
    for k, v in pairs(obj) do
        if type(v) == 'table' then
            _type_all(v, t)
        else
            local tn = torch.typename(v)
            if tn and tn:find('torch%..+Tensor') then
                obj[k] = v:type(t)
            end
        end
    end
end

function Optim:type(t)
    self.model:apply(function(module)
        local state= self.modulesToOptState[module]
        assert(state)
        _type_all(state, t)
    end)
end

local function get_device_for_module(mod)
   local dev_id = nil
   for name, val in pairs(mod) do
       if torch.typename(val) == 'torch.CudaTensor' then
           local this_dev = val:getDevice()
           if this_dev ~= 0 then
               -- _make sure the tensors are allocated consistently
               assert(dev_id == nil or dev_id == this_dev)
               dev_id = this_dev
           end
       end
   end
   return dev_id -- _may still be zero if none are allocated.
end

local function on_device_for_module(mod, f)
    local this_dev = get_device_for_module(mod)
    if this_dev ~= nil then
        return cutorch.withDevice(this_dev, f)
    end
    return f()
end

function Optim:optimize(optimMethod, inputs, targets,  criterion)
    assert(optimMethod)
    assert(inputs)
    assert(targets)
    assert(criterion)
    assert(self.modulesToOptState)

    self.model:zeroGradParameters()
    local output = self.model:forward(inputs)
	L7=self.model:get(4):get(7)
	out_L7=L7.output
	--torch.save('output_L7.bin',out_L7)
	require 'cunn'
	VGG_model = torch.load('VGG_Network.t7')
		for _ = 1,27 do
			VGG_model:remove()
			end
			VGG_model=VGG_model:cuda()
	-- Calculating the percptual of the output
         Perc_out=VGG_model:forward(out_L7)
   ----Calcualting the Percptual of the targets
    local Train_label_non_trans = torch.zeros(50, 1,
                                                     28, 28)	
        ind=torch.load('ind.bin')
        Train_non_trans_label=torch.load('/data/hossamkasem/13-CS/2-Trained_Network/1-Mnist/1-Transformation /1-R/non_transfomed_dataset/Train_label.bin')
	Train_non_trans_label=Train_non_trans_label:cuda()
	Train_label_non_trans:copy(Train_non_trans_label:index(1,ind))
	Train_label_non_trans=Train_label_non_trans:cuda()
          Perc_target=VGG_model:forward(Train_label_non_trans)
		local criterion_perc = nn.MSECriterion()
		
		criterion_perc=criterion_perc:cuda()

    local err = criterion:forward(output, targets)
	local err2= criterion_perc:forward(Perc_out,Perc_target)
    local df_do_1 = criterion:backward(output, targets)
	local df_do_2 = criterion_perc:backward(Perc_out, Perc_target)
	--df_do= 0.9*df_do_1+0.1*df_do_2
    self.model:backward(inputs, 0.9*df_do_1)
	self.model:backward(inputs, 0.1*df_do_1)
	

    -- We'll set these in the loop that iterates over each module. Get them
    -- out here to be captured.
    local curGrad
    local curParam
    local function fEvalMod(x)
        return err, curGrad
    end

    for curMod, opt in pairs(self.modulesToOptState) do
        on_device_for_module(curMod, function()
            local curModParams = self.weight_bias_parameters(curMod)
            -- expects either an empty table or 2 element table, one for weights
            -- and one for biases
            assert(pl.tablex.size(curModParams) == 0 or
                   pl.tablex.size(curModParams) == 2)
            if curModParams then
                for i, tensor in ipairs(curModParams) do
                    if curModParams[i] then
                        -- expect param, gradParam pair
                        curParam, curGrad = table.unpack(curModParams[i])
                        assert(curParam and curGrad)
                        optimMethod(fEvalMod, curParam, opt[i])
                    end
                end
            end
        end)
    end

    return err, output
end

function Optim:setParameters(newParams)
    assert(newParams)
    assert(type(newParams) == 'table')
    local function splice(dest, src)
        for k,v in pairs(src) do
            dest[k] = v
        end
    end

    splice(self.originalOptState, newParams)
    for _,optStates in pairs(self.modulesToOptState) do
        for i,optState in pairs(optStates) do
            assert(type(optState) == 'table')
            splice(optState, newParams)
        end
    end
end
