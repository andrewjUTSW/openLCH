-- Training AAE for LCH image data

-- Adapted from the Allen Institute for Cell Science
-- https://github.com/AllenCellModeling/torch_integrated_cell/

-- Andrew R. Jamieson Nov. 2017 
-- UT Southwestern Medical Center

-- BioHPC Environment configurations
-- module add cuda80; module add torch; module add cudnn/5.1.5; 

-- or Singularity containter:
-- singularity pull shub://andrewjUTSW/openLCH:latest
-- singularity exec --nv openLCH_latest.sif /bin/bash -c '<code>'

-- th ./run_mainLCH_AAE_Train_2.lua \
-- -nLatentDims 56 \
-- -imsize 256 \
-- -savedir LCH/sampleCode/outputNew/ \
-- -imPathFile /LCH/sampleCode/imagePathList.txt \
-- -modelname AAEconv_CLEAN \
-- -epochs 100 \
-- -saveProgressIter 1 \
-- -saveStateIter 1 \
-- -batchSize 50 \
-- -batchSizeLoad 20000 \
-- -miniBatchSizeLoad 1000 \
-- -gpu 1 \
-- -epochs 25'


require 'nn'
require 'cunn'
require 'paths'
require 'dpnn'
require 'utilsImg'

optim = require 'optim'
gnuplot = require 'gnuplot'
image = require 'image'
nn = require 'nn'
optnet = require 'optnet'
gnuplot = require 'gnuplot'
log = require 'log'

cuda = pcall(require, 'cutorch')
local hasCudnn, cudnn = pcall(require, 'cudnn')
log.trace('Has cudnn: ')
print(hasCudnn)

local path = require 'pl.path'
require 'debugRepl'
require 'imtools_LCH'

unpack = table.unpack or unpack

count = cutorch.getDeviceCount()

log.trace('GPU is ON')
for i = 1, count do
    log.trace('Device ' .. i .. ':')
    freeMemory, totalMemory = cutorch.getMemoryUsage(i)
    log.trace('\t Free memory ' .. freeMemory)
    log.trace('\t Total memory ' .. totalMemory)
end

cmd = torch.CmdLine()
--model settings
cmd:option('-modelname',        'AAEconv_CLEAN', 'model name')
cmd:option('-nLatentDims',       56, 'dimensionality of the latent space')
cmd:option('-seed',              1, 'random seed')
cmd:option('-load_autoencoder',  '', 'load a specific autoencoder')
-- saving settings
cmd:option('-savedir',           './trainingOutput', 'save dir')
cmd:option('-saveStateIter',     1, 'Iterations between saving model states')
cmd:option('-saveProgressIter',  1, 'Iterations between save progress states')
cmd:option('-snapshotFrequency', 200, 'Iterations between snapshoot progress states')
cmd:option('-numZsamples',       8, 'number of mcmc samples to preview training progress')
cmd:option('-saveEmbedding',     false, '[0/1] save embedding codes for data set')
-- gpu settings
cmd:option('-cpu',               false, 'CPU only (useful if GPU memory is too low)')
cmd:option('-gpu',               1, 'GPU to use')
cmd:option('-cudnnBenchmark',    false, 'benchmark mode for cudnn')
cmd:option('-cudnnFastest',      false, 'fastest mode for cudnn')

-- data settings
cmd:option('-imPathFile',        '', 'text file with image paths per line')
cmd:option('-imPathFileTest',     '', 'text file with image paths per line for Testing or preselected')
cmd:option('-imsize',             256, 'desired size of images in dataset')
cmd:option('-dataProvider',      'DynDataProviderRobust_2', 'data provider object')
cmd:option('-batchSizeLoad',     10000, 'batch size for pre-loading images from disk')
cmd:option('-miniBatchSizeLoad',  500, 'mini batch size for DynDataProvider Parallel loading images into RAM')
cmd:option('-useParallel',        0, 'attempt to load images with parallel threads')
cmd:option('-numThreadsLoad',     10, 'attempt to load images with parallel threads')

-- training settings
cmd:option('-batchSize',         50, 'Minibatch size for updating AAE network')
cmd:option('-optimizer',         'adam', 'optimization method')
cmd:option('-learningRate',      0.0002, 'learning rate')
cmd:option('-mcmc',              1, 'MCMC samples')
cmd:option('-sampleStd',         1, 'Standard deviation of Gaussian distribution to sample from')
cmd:option('-advGenRatio',       1E-4, 'ratio for advGen update')
cmd:option('-advLatentRatio',    1E-4, 'ratio for advLatent update')
cmd:option('-ganNoise',          0, 'injection noise for the GAN')
cmd:option('-ganNoiseAllLayers', false, 'add noise on all GAN layers')
cmd:option('-epochs',            150, 'number of epochs')
cmd:option('-detailError',       0, 'log error for every mini-batch')

cmd:option('-skipGanD',          false, 'use a GAN on the decoder')
cmd:option('-beta1',             0.5, 'beta1 parameter for ADAM descent')
cmd:option('-ndat',              -1, 'number of training data to use')
cmd:option('-learningRateDecay', 0.999, 'learning rate decay')
-- Image Training Transformation Settings
cmd:option('-rotImage',          0, 'rotate training images randomly')
cmd:option('-gaussSigmaIn',      0, 'Gaussian sigma (in percentge of image size) for masking/multiplying by training images')
cmd:option('-lcn',               0, 'Local Contast Normalization performed on all images')

-- display settings
cmd:option('-verbose', false, 'verbosity setting')
opts = cmd:parse(arg)
opts.timeStamp = os.date("%d%b%y_%H%M")

print(opts)
-- debugRepl()
if opts.epochs < opts.saveStateIter then
    opts.saveStateIter = opts.epochs
end
if opts.epochs < opts.saveProgressIter then
    opts.saveProgressIter = opts.epochs
end

-- Set up Torch
log.trace('----------------------------------')
log.trace('Setting up Torch')
log.trace('----------------------------------')
log.trace('Setting default GPU to ' .. opts.gpu)

cutorch.setDevice(opts.gpu)
torch.setnumthreads(12)
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opts.seed) 
cutorch.manualSeed(opts.seed)

log.trace('----------------------------------')
log.trace('-----[DATA SETUP]-----------------')
log.trace('-----[DATA SETUP]-----------------')
log.trace('----------------------------------')
DataProvider = require(opts.dataProvider)

print(opts)
log.trace('Set up images')
local optsImg =  {}
optsImg.save_dir = opts.savedir
optsImg.imPathFile = opts.imPathFile
optsImg.imsize = opts.imsize
optsImg.gaussSigmaIn = opts.gaussSigmaIn
optsImg.lcn = opts.lcn
optsImg.rotImage = opts.rotImage

log.trace('=========[LOAD DATASET PATHS]=========')
local tpaths = {}
tpaths['train'] = utils.readlines_from(opts.imPathFile) -- 
if #opts.imPathFileTest > 0 then
  log.trace('---[Loading specific test data set]------')
  tpaths['test'] = utils.readlines_from(opts.imPathFileTest) 
else
  tpaths['test'] = utils.readlines_from(opts.imPathFile)
end
log.trace('Number of image paths for training: ' .. #tpaths['train'])
log.trace('Number of image paths for testing: ' .. #tpaths['test'])

optsImg.paths = tpaths
optsImg.numThreads = opts.numThreadsLoad
dataProvider = DataProvider.create(optsImg)

local tokens = utils.split(opts.savedir, '/')
local exprName = tokens[#tokens]

if opts.cpu then
  cuda = false
end
log.trace('----------------------------------')
log.trace('-----[DATA SETUP COMPLETE]--------')
log.trace('----------------------------------')
print(opts)

log.trace('----------------------------------')
log.trace('---------[MODEL SETUP]------------')
log.trace('----------------------------------')
setup = require 'setup_LCH_Auto'

-- Create model
local model_opts = {}; 
model_opts.nLatentDims = opts.nLatentDims
model_opts.nChIn = 1 -- model_opts.channel_inds_in:size(1)
model_opts.nChOut = 1 --model_opts.channel_inds_out:size(1)
model_opts.dropoutRate = 0.2
model_opts.imsize = opts.imsize
model_opts.save_dir = opts.savedir
model_opts.model_name = opts.modelname
model_opts.image_dir = optsImg.image_dir
model_opts.test_model = true
model_opts.load_autoencoder = opts.load_autoencoder -- if we have
print(model_opts)
local snapshot_dir = model_opts.save_dir .. '/snapshots'
epoch_start = 1
setup.getModel(model_opts)
log.outfile = model_opts.save_dir .. '/loggerTrace_' .. opts.timeStamp .. '_' .. exprName .. '.log'
log.trace('done model set up')
local allOpts = {}
allOpts['model_opts'] = model_opts
allOpts['opts'] = opts
allOpts['optsImg'] = optsImg
torch.save(model_opts.save_dir .. '/allOpts_'.. opts.timeStamp .. '_' .. exprName.. '.t7', allOpts, 'ascii')
log.trace('----------------------------------')
log.trace('---------[MODEL SETUP COMPLETE]---')
log.trace('----------------------------------')


log.trace('----------------------------------')
log.trace('-- Optimization Setup')
log.trace('----------------------------------')

-- Get parameters
local theta, gradTheta = autoencoder:getParameters()
local thetaAdv, gradThetaAdv = adversary:getParameters()

-- Create optimizer function evaluation
local x -- Minibatch
local tMSE = {}
local xHat_mse_mean
criterion_mse = nn.MSECriterion()

local feval = function(params)
    if theta ~= params then
      theta:copy(params)
    end

    gradTheta:zero()
    gradThetaAdv:zero()
    
    local xHat = autoencoder:forward(x) 
    local loss = criterion:forward(xHat, x)
    
    xHat_mse_mean = criterion_mse:forward(xHat, x)

    tMSE = {}
    for i_z = 1, xHat:size(1) do
      tMSE[i_z] = criterion_mse:forward(xHat[i_z], x[i_z])
    end

    local gradLoss = criterion:backward(xHat, x)
    autoencoder:backward(x, gradLoss)

    local real = torch.Tensor(x:size(1), opts.nLatentDims):normal(0, 1):typeAs(x)
    local YReal = torch.ones(x:size(1)):typeAs(x) 
    local YFake = torch.zeros(x:size(1)):typeAs(x)

    local pred = adversary:forward(real)
    local realLoss = criterion:forward(pred, YReal)
    local gradRealLoss = criterion:backward(pred, YReal)
    adversary:backward(real, gradRealLoss)

    pred = adversary:forward(autoencoder.modules[1].output)
    local fakeLoss = criterion:forward(pred, YFake)
    advLoss = realLoss + fakeLoss
    local gradFakeLoss = criterion:backward(pred, YFake)
    local gradFake = adversary:backward(autoencoder.modules[1].output, gradFakeLoss)

    local minimaxLoss = criterion:forward(pred, YReal)
    loss = loss + minimaxLoss
    local gradMinimaxLoss = criterion:backward(pred, YReal)
    local gradMinimax = adversary:updateGradInput(autoencoder.modules[1].output, gradMinimaxLoss)
    autoencoder.modules[1]:backward(x,gradMinimax);

    return loss, gradTheta
end

local advFeval = function(params)
  if thetaAdv ~= params then
    thetaAdv:copy(params)
  end
  return advLoss, gradThetaAdv
end

log.trace('----------------------------------')
log.trace('-- Optimization Setup COMPLETE')
log.trace('----------------------------------')


log.trace('----------------------------------')
log.trace('-- [TRAINING INITIALIZING]-------')
log.trace('----------------------------------')
-- Train
log.trace('Training')
autoencoder:training()
adversary:training()

cudnn.benchmark = false
cudnn.fastest = false

if hasCudnn and cuda then
  cudnn.convert(autoencoder, cudnn)
  cudnn.convert(adversary, cudnn)
end

local optimParams = {learningRate = opts.learningRate}
local advOptimParams = {learningRate = opts.learningRate}
local __, loss, lossA
local losses, advLosses = {}, {}

local ndat = #dataProvider.train.paths
local ntest = #dataProvider.test.paths

logger = optim.Logger(path.join(model_opts.save_dir, 'AAE_loss_' .. opts.timeStamp .. '_' .. exprName.. '.log'))
logger:setNames{'datetime', 'epoch', 'minibatch', 'autoencoder loss', 'adversarial Loss', 'imRecon_MSE_mean', 'imRecon_MSE_std'}

if opts.batchSizeLoad > ndat then
  opts.batchSizeLoad = ndat
end

paths.mkdir(opts.savedir) 

-- Start of main EPOCH loop --
local totBatchCount = 1;
for epoch = epoch_start, opts.epochs do
  
  local ticEpoch = torch.tic()
  local indices1 = torch.randperm(ndat):long():split(opts.batchSizeLoad)
 
   -- indices1[#indices1] = nil
  local N = #indices1 * opts.batchSizeLoad

  log.trace('Epoch ' .. epoch .. '/' .. opts.epochs)
  local ind_c1 = 1

  for t1,v1 in ipairs(indices1) do

    log.debug('=========[  batch # '.. i .. '     ]=========')
    log.debug('=========[loading big batch # ' .. opts.batchSizeLoad .. ' into memory]=========')
    local x1, x_out1 = nil
    
    if opts.useParallel == 1 then
      log.warn('Attempting parallel data load')
      x1, x_out1 = dataProvider:getImagesParallel(v1, nil, nil, opts.miniBatchSizeLoad, opts.rotImage, opts.gaussSigmaIn, opts.lcn)
    else
      x1, x_out1 = dataProvider:getImages(v1, nil, nil, opts.miniBatchSizeLoad, opts.rotImage, opts.gaussSigmaIn, opts.lcn)
    end
    
    local indices = torch.linspace(1, x1:size(1), x1:size(1)):long():split(opts.batchSize)
    local ind_c = 1
    for t,v in ipairs(indices) do

      collectgarbage()    
      collectgarbage()    
      x = x1:index(1, v)
      if cuda then
        x = x:cuda()
      end
      local tic = torch.tic()
      __, loss = optim[opts.optimizer](feval, theta, optimParams)
      losses[#losses + 1] = loss[1]
      
      -- Train adversary
      __, lossA = optim[opts.optimizer](advFeval, thetaAdv, advOptimParams)     
      
      local mse_std = torch.std((torch.Tensor(tMSE)))
      local mse_mean = torch.mean(torch.Tensor(tMSE))

      advLosses[#advLosses + 1] = lossA[1]
      log.trace('epoch ' .. epoch .. ' | miniBatch ['.. ind_c ..'/' ..#indices ..'[subbatch] -- ' .. ind_c1 .. '/' .. #indices1 ..'[DataLoads] ] | autoencoder loss :' .. loss[1] .. ' | advLosses :' .. lossA[1].. ' | meanMSE :' .. mse_mean .. ' | stdMSE :' .. mse_std)
      
      if opts.detailError == 1 then
          logger:add{os.date(), epoch, ind_c, loss[1], lossA[1], xHat_mse_mean, mse_std}
      else
        if totBatchCount % 50 == 0 then
          logger:add{os.date(), epoch, ind_c, loss[1], lossA[1], xHat_mse_mean, mse_std}
        end      
      end

      if totBatchCount % 200 == 0 then
        log.warn('intermediate save state: ')
        if paths.filep(model_opts.save_dir .. '/adversary.t7 ') then
          log.warn('copying previous adversary state')
          os.execute('cp ' .. model_opts.save_dir .. '/adversary.t7 ' .. model_opts.save_dir .. '/adversary_backup.t7')
        end
        torch.save(model_opts.save_dir .. '/adversary.t7', adversary)-- 'binary', false)
        if paths.filep(model_opts.save_dir .. '/autoencoder.t7') then
          log.warn('copying previous autoencoder state')
          os.execute('cp ' .. model_opts.save_dir .. '/autoencoder.t7 ' .. model_opts.save_dir .. '/autoencoder_backup.t7')
        end
        log.info('Saving ... autoencoder...')
        torch.save(model_opts.save_dir .. '/autoencoder.t7', autoencoder)--, 'binary', false)
        log.info('DONE Saving ... autoencoder-training...')


        log.debug('~~~~~~~~~~garbage collecting~~~~~~~~~~~~~~~')  
        collectgarbage()    
        collectgarbage()
        autoencoder:evaluate()
        adversary:clearState()
        autoencoder:clearState()
        collectgarbage()    
        collectgarbage()
        autoencoder:training()
        log.debug('~~~~~~~~~~DONE garbage collecting~~~~~~~~~~~~~~~')
      end
      
      if totBatchCount % opts.snapshotFrequency == 0 then
        
        autoencoder:evaluate()

        log.warn('+++++++++++++++++++ generating SNAPHOT images +++++++++++++++++++++')
        -- Current Snapshop save
        paths.mkdir(snapshot_dir)
        log.trace('----First, RANDOM selection---')
        local rtIndxAll = torch.randperm(ndat);
        local rtInd = rtIndxAll:index(1,torch.linspace(1,16,16):long());

        local x_inA, x_outA = dataProvider:getImages(rtInd, 'train', nil, opts.rotImage, opts.gaussSigmaIn, opts.lcn)
        local recon_trainA = utilsImg.evalImAuto(x_inA,x_outA)
        image.save(snapshot_dir .. '/RECON_RAND_images_'.. opts.timeStamp ..'E'.. epoch .. '_['.. totBatchCount*opts.batchSize ..'-'.. ndat ..']' .. '.png', recon_trainA)
        
        log.trace('----Next, pre-selected, fixed set of images---')

        local x_in, x_out = dataProvider:getImages(torch.linspace(1,100,100):long(), 'test', nil, nil, opts.gaussSigmaIn, opts.lcn)
        local recon_testA = utilsImg.evalImAuto(x_in,x_out)  

        image.save(snapshot_dir .. '/RECON_TEST_images_'.. opts.timeStamp ..'E'.. epoch .. '_['.. totBatchCount*opts.batchSize ..'-'.. ndat ..']' .. '.png', recon_testA)

        log.trace('>>>---MCMC--- samples---')
        local outputA = autoencoder.modules[2]:forward(torch.Tensor(opts.numZsamples * opts.numZsamples, model_opts.nLatentDims):normal(0, opts.sampleStd):typeAs(x_inA:cuda())):clone();
        autoencoder:forward(outputA)
        image.save(snapshot_dir .. '/MCMCSamples_'.. opts.timeStamp ..'E'.. epoch .. ' ['.. totBatchCount*opts.batchSize ..'-'.. ndat ..']' .. '.png', image.toDisplayTensor(autoencoder.modules[2].output, 0, opts.numZsamples))
        log.trace('>>>---DONE with MCMC--- samples---')         
        
        log.debug('Flipping Autoencoder back to --training-- mode')        
        autoencoder:training()

        log.warn('+++++++++++++++++++ DONE +++++++++++++++++++++')

      end
      x, x_out = nil, nil
      ind_c = 1 + ind_c
      totBatchCount = totBatchCount + 1
    end
  ind_c1 = 1 + ind_c1
  
  end
  collectgarbage()
  collectgarbage()    
  
  -- Plot training curve(s)
  local plots = {{'Autoencoder', torch.linspace(1, #losses, #losses), torch.Tensor(losses), '-'}}
  plots[#plots + 1] = {'Adversary', torch.linspace(1, #advLosses, #advLosses), torch.Tensor(advLosses), '-'}  
  
  if epoch % opts.saveStateIter == 0 then
      log.trace('==================================')
      log.trace('==== >>>>> Saving model state<<<==')
    
      -- rotate_tmp = model_opts.rotate
      -- dataProvider.opts.rotate = false
     
      autoencoder:training()

      adversary:clearState()
      autoencoder:clearState()      
      
      torch.save(model_opts.save_dir .. '/_plots.t7', plots, 'binary', false)
      torch.save(model_opts.save_dir .. '/epoch.t7', epoch, 'binary', false)
    
      if paths.filep(model_opts.save_dir .. '/adversary.t7') then
        os.execute('cp ' .. model_opts.save_dir .. '/adversary.t7 ' .. model_opts.save_dir .. '/adversary_backup.t7')
      end
      if paths.filep(model_opts.save_dir .. '/autoencoder.t7') then
        os.execute('cp ' .. model_opts.save_dir .. '/autoencoder.t7 ' .. model_opts.save_dir .. '/autoencoder_backup.t7')
        os.execute('cp ' .. model_opts.save_dir .. '/autoencoder_eval.t7 ' .. model_opts.save_dir .. '/autoencoder_eval_backup.t7')
      end
      log.trace('...done')
      torch.save(model_opts.save_dir .. '/adversary.t7', adversary)-- 'binary', false)
      collectgarbage()    
      collectgarbage()    
      log.info('Saving ... autoencoder...')
      torch.save(model_opts.save_dir .. '/autoencoder.t7', autoencoder)--, 'binary', false)
      log.info('DONE Saving ... autoencoder-training...')
      autoencoder:evaluate()
      log.info('Saving ... autoencoder-eval...')
      torch.save(model_opts.save_dir .. '/autoencoder_eval.t7', autoencoder)--, 'binary', false)
      log.info('DONE -Saving ... autoencoder-eval...')
      autoencoder:training()

      torch.save(model_opts.save_dir .. '/rng.t7', torch.getRNGState(), 'binary', false)
      torch.save(model_opts.save_dir .. '/rng_cuda.t7', cutorch.getRNGState(), 'binary', false)            

      adversary:cuda()
      autoencoder:cuda()
      
      log.trace('=====[DONE with SAVE]=========')
      log.trace('==============================')
  end
  if epoch % opts.saveProgressIter == 0 then
      local tic = torch.tic()
      log.trace('==============================')
      log.trace('Saving Progress')
      log.trace('==============================')
      autoencoder:evaluate()
      
      local x_in, x_out = dataProvider:getImages(torch.linspace(1,16,16):long(), 'train')
      recon_train = utilsImg.evalImAuto(x_in,x_out)
      local x_in, x_out = dataProvider:getImages(torch.linspace(1,16,16):long(), 'test')
      recon_test = utilsImg.evalImAuto(x_in,x_out)  
      local reconstructions = torch.cat(recon_train, recon_test, 2)
      image.save(model_opts.save_dir .. '/'.. opts.timeStamp .. '_' .. exprName.. '_progressAUTO'.. epoch ..'.png', reconstructions)
      
      log.trace('>>>---MCMC--- samples---')
      local output = autoencoder.modules[2]:forward(torch.Tensor(opts.numZsamples * opts.numZsamples, model_opts.nLatentDims):normal(0, opts.sampleStd):typeAs(x_in:cuda())):clone();
      autoencoder:forward(output)
      image.save(model_opts.save_dir .. '/'.. opts.timeStamp .. '_' .. exprName.. 'Samples'.. epoch ..'.png', image.toDisplayTensor(autoencoder.modules[2].output, 0, opts.numZsamples))
      log.trace('>>>---DONE with MCMC--- samples---')

      if opts.saveEmbedding then
        log.trace('>>>---Calculating Embeddings===++++++++')
        embeddings = {}
        embeddings.train = torch.zeros(ndat, model_opts.nLatentDims)
        indices = torch.linspace(1,ndat,ndat):long():split(opts.batchSize)

        start = 1
        for t,v in ipairs(indices) do
            collectgarbage()
            stop = start + v:size(1) - 1

            x_in = dataProvider:getImages(v, 'train')
            x_in = x_in:cuda()

            codes = autoencoder.modules[1]:forward(x_in)
            embeddings.train:sub(start, stop, 1,model_opts.nLatentDims):copy(codes)
            
            start = stop + 1
        end
        log.trace('>>>---Calculating Embeddings [TEST set] ===++++++++')
        embeddings.test = torch.zeros(ntest, model_opts.nLatentDims)
        indices = torch.linspace(1,ntest,ntest):long():split(opts.batchSize)
        
        start = 1
        for t,v in ipairs(indices) do
            collectgarbage()
            stop = start + v:size(1) - 1

            x_in = dataProvider:getImages(v, 'test')
            x_in = x_in:cuda()
                 
            codes = autoencoder.modules[1]:forward(x_in)
            embeddings.test:sub(start, stop, 1, model_opts.nLatentDims):copy(codes)
            
            start = stop + 1
        end
      end
      local toc = torch.toc(tic)
      log.trace('Done in :' .. toc)
      log.trace('==============================')
      x_in = nil
      
      autoencoder:training()          

      torch.save(model_opts.save_dir .. '/progress_embeddings.t7', embeddings, 'ascii')
      embeddings = nil
      
      torch.save(model_opts.save_dir .. '/plots_tmp.t7', plots, 'binary', false)
      torch.save(model_opts.save_dir .. '/epoch_tmp.t7', epoch, 'binary', false)
  end
  plots = nil
  log.trace('Epoch done in :' .. torch.toc(ticEpoch))
  totBatchCount = 1;
end

log.trace('===============================')
log.trace('== TRAINING Script FINISHED!--------->>> ')
log.trace('===============================')
