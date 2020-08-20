local optim = require 'optim'
local cuda = pcall(require, 'cutorch') -- Use CUDA if available
local hasCudnn, cudnn = pcall(require, 'cudnn') -- Use cuDNN if available
require 'dpnn'
require 'nn'
require 'cunn' -- https://github.com/soumith/cudnn.torch/issues/129
------

cmd = torch.CmdLine()
cmd:option('-autoencoder', 'autoencoder t7 file to load')
cmd:option('-gpu',  1, 'Which GPU device to use')
opts = cmd:parse(arg)
print(opts)

cutorch.setDevice(opts.gpu)

print('=========[LOAD AUTOENCODER]=========')
print(opts.autoencoder)
autoencoder = nil
autoencoder = torch.load(opts.autoencoder)
print(autoencoder)
autoencoder:float()
autoencoder:clearState()
collectgarbage()
print('Converting to cuda')
autoencoder:cuda()
print('==========[SUCCESS!]==============\n')


