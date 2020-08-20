-- call_DynComputeEmbeddingsRobust_2.lua
--		Compute latent space embeddings using trained image autoencoder 
--		and export to .csv file.

-- BioHPC Environment configurations
-- module add cuda80; module add torch; module add cudnn/5.1.5; 

-- or Singularity containter:
-- singularity pull shub://andrewjUTSW/openLCH:latest
-- singularity exec --nv openLCH_latest.sif /bin/bash -c '<code>'

-- th ./call_DynComputeEmbeddingsRobust_2.lua \
-- -autoencoder output/autoencoder_eval.t7 \
-- -imsize 256 \
-- -dataProvider DynDataProviderRobust_2 \
-- -imPathFile imageList3.txt \
-- -batchSize 100 \
-- -batchSizeLoad 20000 \
-- -miniBatchSizeLoad 2500 \
-- -gpu 2 \
-- -useParallel 1 \
-- -numThreads 3 \
-- -embeddingFile output/embeddings_sampleTest.csv

local optim = require 'optim'
local gnuplot = require 'gnuplot'
local image = require 'image'
local cuda = pcall(require, 'cutorch') -- Use CUDA if available
local hasCudnn, cudnn = pcall(require, 'cudnn') -- Use cuDNN if available
require 'dpnn'
require 'paths'
require 'imtools_LCH'
require 'utils'
require 'nn'
require 'torchx'
require 'cunn' -- https://github.com/soumith/cudnn.torch/issues/129
require 'debugRepl'
log = require 'log'
------

cmd = torch.CmdLine()
-- Input Image config
cmd:option('-t7ImageDataOptsfile', '', 't7 opts file for loading image data')
cmd:option('-imPathFile', '', 'text file with image paths per line')
cmd:option('-dataProvider',   'DynDataProviderRobust_2', 'data provider object')
cmd:option('-imsize',         256, 'data set ImageSize, if different will be rescaled.')
cmd:option('-useParallel',    0, 'attempt to load images with parallel threads')
cmd:option('-numThreads',     10, 'attempt to load images with parallel threads')

-- autoencoder and output 
cmd:option('-autoencoder', 'autoencoder t7 file to load')
cmd:option('-embeddingFile', 'embeddings.csv', 'path to emebdding csv text readable data file')

-- Compute config
cmd:option('-gpu',                1, 'Which GPU device to use')
cmd:option('-batchSize',          100, 'batch size for procressing AAE emebddings in CUDA on GPU')
cmd:option('-batchSizeLoad',      10000, 'batch size for pre-loading images from disk into memory')
cmd:option('-miniBatchSizeLoad',  2500, 'mini batch size for DynDataProvider Parallel loading images into RAM')

-- Image Pre-processing config (feeds into the dataloader)
cmd:option('-lpf',           0, 'Low-pass Filter, Gaussian Blur Sigman')
cmd:option('-gaussSigmaIn',  0, 'Gaussian sigma (in percentge of image size) for masking/multiplying by training images')
cmd:option('-lcn',           0, 'Local Contast Normalization performed on all images')
cmd:option('-lpfKernelSize', 55, 'Low-pass Filter, Gaussian Blur kernel Size')
cmd:option('-illumShift',    0, 'illumination shift in images')

opts = cmd:parse(arg)
print(opts)

cutorch.setDevice(opts.gpu)
ef = opts.embeddingFile;
log.outfile = string.sub(ef, 1, #ef-4) .. '.log'
log.level = 'trace'
log.trace('Initializing DataProvider')

if opts.t7ImageDataOptsfile ~= '' then
	optsData = torch.load(opts.t7ImageDataOptsfile)
	log.trace('==========[LOAD DATASET from OPTS t7 file]=========')
	DataProvider = require(optsData.dataProvider)
	data = DataProvider.create(optsData)
else
	print('==========[READING IMAGES from image path list file]====\n')
	local tpaths = {}
	tpaths['train'] = utils.readlines_from(opts.imPathFile) -- 
	print('Number of image paths for training: ' .. #tpaths['train'])
	print('==========[LOAD DATASET]=============================\n')
	opts.paths = tpaths
	DataProvider = require(opts.dataProvider)
	data = DataProvider.create(opts)
end
log.trace('==========[SUCCESS]===================================')
log.trace('set up logger')

collectgarbage()
collectgarbage()
log.trace('==========[LOAD AUTOENCODER]==========================')
autoencoder = nil
print(opts.autoencoder)
autoencoder = torch.load(opts.autoencoder)
log.trace('=========[DONE loading AUTOENCODER]===================')
autoencoder:clearState()
autoencoder:evaluate()
collectgarbage()
log.trace('Converting to cuda')
autoencoder:cuda()
autoencoder:evaluate()
log.trace('==========[SUCCESS!]================================\n')

log.trace('==========[verifying AUTOENCODER ...]==================')
local xTest = data:getImages(torch.LongTensor{1,1}, nil, nil, nil, opts.gaussSigma, opts.lcn, opts.lpf, opts.lpfKernelSize, opts.illumShift)
local xHat_auto = autoencoder:forward(xTest:cuda())
local codes = autoencoder.modules[1].output
local xHat_auto2 = autoencoder.modules[2]:forward(codes)
local nZ = codes:size(2)
log.trace('==========[DONE verifying AUTOENCODER...]=============')

log.trace('==========[Begin Computing EMBEDDINGS]================')

ndat = #data.train.paths

local indices1 = torch.linspace(1,ndat,ndat):long():split(opts.batchSizeLoad)
tcodes = {}
xTest = nil
codes = nil

-- Pre-allocated memory for embeddings
local embeddings = torch.zeros(ndat, nZ);
local i = 1;
for t1,v1 in ipairs(indices1) do
	collectgarbage()
	collectgarbage()
	-- First load big batch batchSizeLoad into memory
	log.trace('=========[  batch # '.. i .. '     ]=========')
	log.trace('=========[loading big batch # ' .. opts.batchSizeLoad .. ' into memory]=========')
	local x1, x_out1 = nil
	sys.tic()
	
	if opts.useParallel == 1 then
		log.warn('Attempting parallel data load')
		x1, x_out1 = data:getImagesParallel(v1, nil, nil, opts.miniBatchSizeLoad, nil, opts.gaussSigma, opts.lcn, opts.lpf, opts.lpfKernelSize, opts.illumShift) 
	else
		x1, x_out1 = data:getImages(v1, nil, nil, nil, opts.gaussSigma, opts.lcn, opts.lpf, opts.lpfKernelSize, opts.illumShift)
	end
	
	local dataloadTime = sys.toc()
	log.trace('***************** DATA LOAD TOTAL time: ' .. dataloadTime..' *********************')
	log.trace('=========[DONE]=========')
	
	local indicesG = torch.linspace(torch.min(v1), torch.max(v1), v1:size(1)):long():split(opts.batchSize)
	local indices = torch.linspace(1, x1:size(1), x1:size(1)):long():split(opts.batchSize)
	
	local j = 1;
	log.trace('=========[Start] << ' .. i ..' >>mini-batch embeddings]=========')
	for t,v in ipairs(indices) do
		autoencoder:evaluate()
		local start = torch.min(v);
		local stop = torch.max(v);
		
		local startG = torch.min(indicesG[j]);
		local stopG = torch.max(indicesG[j]);

		local x = x1:index(1, v)
		local s = sys.tic()
		
		if x:size(1) == 1 then
			x = torch.cat(x,x,1) -- CUDA needs at least 2 tensors for some reason?
			local xHat = autoencoder:forward(x:cuda())
			local codes = autoencoder.modules[1].output
			embeddings:sub(startG, stopG, 1, nZ):copy(codes[1])		
		else
			sys.tic()
			local xHat = autoencoder:forward(x:cuda())
			local codes = autoencoder.modules[1].output
			embeddings:sub(startG, stopG, 1, nZ):copy(codes)
		end
		local GPUtime = sys.toc();
		
		if j % 20 == 0 then
			-- debugRepl()
			log.trace('    	Start (absolute) : '..startG .. '    ')
			log.trace('		Stop  (absolute) : '..stopG.. '    ')
			log.trace('		Start (batch)    : '..start.. '    ')
			log.trace('		Stop  (batch)    : '..stop.. '    ')
			log.trace('GPU time:' .. GPUtime)	
			xlua.progress(j, #indices); 
		end
		j = j + 1;
	end
	
	log.trace('=========[DONE with mini-batch embeddings]=========')
	log.trace('=========^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^=========')
	log.trace('&&*****************DATA LOAD TOTAL time:' .. dataloadTime..'*********************')
	log.trace('                                                                                ')
	log.trace(xlua.progress(i, #indices1)); i = i + 1;
	log.trace('                                                                                ')
	log.trace('&&*****************DATA LOAD TOTAL time:' .. dataloadTime..'*********************')
	collectgarbage()
	collectgarbage()

end

log.trace('=========[converting embeddings to table]=========')
log.trace('=========[writing out embeddgins to .csv file...]=========')
utils.csv_write_tensor(opts.embeddingFile, embeddings, sep)
log.trace('=========[COMPLETE]=========')
collectgarbage()