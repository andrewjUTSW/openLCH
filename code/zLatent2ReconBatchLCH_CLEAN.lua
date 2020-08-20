-- Script to explore latent encoding space via image recons from the autoencoder
-- Andrew R. Jamieson Nov. 2017, revamped Nov. 2018, May 2019

-- zLatent2ReconBatchLCH.lua
-- Synopsis: Reconstructs a series of images using a trained autoencoder and input latent codes. 

-- [IMPORTANT] INPUTS:
--						autoencoder: trained autoencoder model to load (.t7)
--						zLatentFile: file containing z codes
--	      	   OUTPUTS:
--						reconPathFile: path for files corresponding  the zLatents Recons.


-- BioHPC Environment configurations
-- module add cuda80; module add torch; module add cudnn/5.1.5; 

-- or Singularity containter:
-- singularity pull shub://andrewjUTSW/openLCH:latest
-- singularity exec --nv openLCH_latest.sif /bin/bash -c '<code>'

-- # sample images/run
-- th -i ./zLatent2ReconBatchLCH_CLEAN.lua \
-- -autoencoder output/autoencoder_eval.t7 \
-- -zLatentFile output/embeddings_sampleTest.csv \
-- -reconPath output/zRecon/ \
-- -nLatentDims 24

-----------
require 'debugRepl'
local optim = require 'optim'
local gnuplot = require 'gnuplot'
local image = require 'image'
local cuda = pcall(require, 'cutorch') 
local hasCudnn, cudnn = pcall(require, 'cudnn')

require 'dpnn'
require 'paths'
require 'imtools_LCH'
require 'utils'
require 'nn'
require 'torchx'
require 'cunn' -- https://github.com/soumith/cudnn.torch/issues/129
require 'utilsImg'
require 'utils'
log = require 'log'

mse = nn.MSECriterion()
count = cutorch.getDeviceCount()

log.trace('GPU is ON')
for i = 1, count do
    log.trace('Device ' .. i .. ':')
    freeMemory, totalMemory = cutorch.getMemoryUsage(i)
    log.trace('\t Free memory ' .. freeMemory)
    log.trace('\t Total memory ' .. totalMemory)
end

-- Function settings
cmd = torch.CmdLine()
-- AAE Model Parameters
cmd:option('-autoencoder',  './autoencoder_eval.t7',  'autoencoder t7 file to load (should be _eval)')
cmd:option('-imsize',       256, 'desired size of images in dataset')
cmd:option('-gaussSigmaIn', 0,   'Gaussian sigma (in percentge of image size) for masking/multiplying by training images')
cmd:option('-lcn',          0,   'Local Contast Normalization performed on all images')
-- Input parameters
cmd:option('-zLatentFile',  './latentCodes.csv',  'csv file containing the latent codes for generating synthetic images')
cmd:option('-nLatentDims',  56,  'AAE model latent space dimension (zdim)')
-- Output parameters
cmd:option('-reconPath',   './zReconOut/', 'path for output reconstructed images to go')
-- Process Execution Configurations
cmd:option('-gpu',          1,   'Which GPU device to use')
cmd:option('-batchSize',    50, 'max num. codes computed on GPU at once')
cmd:option('-debug',        0,  'turn on debug mode')
opts = cmd:parse(arg)

opts.timeStamp = os.date("%d%b%y%H%M")
exprName = opts.timeStamp
print('==============================')
print(opts)
print('==============================')

log.trace('Setting default GPU to ' .. opts.gpu)
cutorch.setDevice(opts.gpu)
torch.setdefaulttensortype('torch.FloatTensor')

log.trace('----------------------------------')
log.trace('---------[DATA INITIALIZATION]-------------')
log.trace('--{Reading zLatent Embeddings files')-------')
local nZ = opts.nLatentDims
zEmbeddings = utils.csv2tensor(opts.zLatentFile, opts.nLatentDims)
local numCodes = zEmbeddings:size(1)
log.trace('----------------------------[done]------')


if opts.debug == 1 then
    debugRepl()
end

log.trace('----------------------------------')
log.trace('-----[DATA INITIALIZATION COMPLETE]--------')
log.trace('----------------------------------')

log.trace('----------------------------------')
log.trace('=========[LOADING AUTOENCODER]=======')
autoencoder = nil
print(opts.autoencoder)
autoencoder = torch.load(opts.autoencoder)
autoencoder:clearState()
autoencoder:evaluate()
collectgarbage()
log.trace('Converting to cuda')
autoencoder:cuda()
autoencoder:evaluate()
local imSize = opts.imsize
log.trace('=[DONE LOADING AUTOENCODER]=======')
log.trace('----------------------------------')

-- Batch Process zCodes 
indices = torch.linspace(1, numCodes, numCodes):long():split(opts.batchSize)

-- Need to batch this later?
local reconX = torch.zeros(numCodes, 1, imSize, imSize)

start = 1
ii = 1
ticAll = torch.tic()
for k,v in ipairs(indices) do
    log.trace('----------------------------------')
    collectgarbage()
    stop = start + v:size(1) - 1
    local vb = v:totable()

    local z_in = zEmbeddings:index(1, v)
	
    tic = torch.tic()
    autoencoder.modules[2]:forward(z_in:cuda());	
    toc = torch.toc(tic)
    print('done reconstructing images in ' .. toc .. ' seconds')	
    log.trace('----------------------------------')
    local xOut_batch = autoencoder.modules[2].output:float()
	
    if opts.debug == 1 then
        print('[DEBUG MODE]: type break to Display sample images of recon')
        debugRepl()
        sampleView_recon = image.toDisplayTensor(xOut_batch, 0, torch.floor(xOut_batch:size(1)^.5))
        gnuplot.figure(1);
        gnuplot.imagesc(sampleView_recon[1])
        print('[DEBUG MODE]: please check Displayed sample images of recon')
        debugRepl()
    end

    -- write out images 
    tic = torch.tic()
    paths.mkdir(opts.reconPath)
    for i = 1, xOut_batch:size(1) do
        image.save(opts.reconPath .. '/reconImg' .. '_'.. exprName ..'_i' .. ii .. '.png', xOut_batch[i][1])
        ii = ii + 1
    end
    toc = torch.toc(tic)
    print('done saving ' ..#vb .. ' images in ' .. toc .. ' seconds')
    log.trace('----------------------------------')
    
    start = stop + 1
end
log.trace('-------------**------------------------')
log.trace('------------*****----------------------')
log.trace('-------------**------------------------')
tocAll = torch.toc(ticAll)
print('done saving images in ' .. toc .. ' seconds')
log.trace('-----------******----------------------')