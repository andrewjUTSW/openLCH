-- Script to explore latent encoding space, given a seed image, dimension by dimension.
-- Andrew Jamieson Nov. 2017, revamped Nov. 2018, May 2019

-- See Input Parameter options below.

-- BioHPC Environment configurations
-- module add cuda80; module add torch; module add cudnn/5.1.5; 

-- or Singularity containter:
-- singularity pull shub://andrewjUTSW/openLCH:latest
-- singularity exec --nv openLCH_latest.sif /bin/bash -c '<code>'

-- Running the script
-- th -i ./exploreZ_LatentSpace_LCH_single_2.lua \
-- -imPathFile LCH/sampleCode/imagePathList.txt \
-- -autoencoder LCH/sampleCode/outputNew/autoencoder_eval.t7 \
-- -outDir LCH/sampleCode/outputNew/zExploreOut \
-- -img1 10 \
-- -uR .5 \
-- -numSteps 20 \
-- -nLatentDims 56'

-----------
require 'debugRepl'
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
cmd:option('-autoencoder',  'autoencoder t7 file to load (should be _eval)')
cmd:option('-imPathFile',  'text file with one image path per line')
cmd:option('-dataProvider', 'DynDataProviderRobust_2', 'data provider object for rendering images from storage')
cmd:option('-outDir',       './zExplore', 'path where to dump saved image galleries')
cmd:option('-img1',         0, 'Start image index')
cmd:option('-gpu',          1, 'Which GPU device to use')
cmd:option('-imsize',       256, 'desired size of images in dataset')
cmd:option('-nLatentDims',  56, 'model zdim')
cmd:option('-uR',  			.5, 'size of each latent space step for sampling in each direction (e.g., uR = 1 [-5,-4,-3,-2,-1,0,1,2,3,4,5] ')
cmd:option('-numSteps',  	8, '# of sampling steps in each direction (e.g., numSteps = 5, with uR = 1 [-5,-4,-3,-2,-1,0,1,2,3,4,5] ')
cmd:option('-batchSize',    100, 'Minibatch size for updating AAE network')
cmd:option('-zeroOrigin',   0, 'use origin image as 0')
cmd:option('-gaussSigmaIn', 0, 'Gaussian sigma (in percentge of image size) for masking/multiplying by training images')
cmd:option('-lcn',          0, 'Local Contast Normalization performed on all images')

opts = cmd:parse(arg)
opts.timeStamp = os.date("%d%b%y%H%M")

print('==============================')
print(opts)
print('==============================')

-- debugRepl()
log.trace('Setting default GPU to ' .. opts.gpu)
cutorch.setDevice(opts.gpu)
torch.setdefaulttensortype('torch.FloatTensor')

log.trace('----------------------------------')
log.trace('-----[DATA SETUP]-----------------')
log.trace('-----{Reading image file list}-----------------')
log.trace('----------------------------------')
DataProvider = require(opts.dataProvider)
local optsImg =  {}
local tpaths = {}
tpaths['train'] = utils.readlines_from(opts.imPathFile) -- 
tpaths['test'] = utils.readlines_from(opts.imPathFile) -- 
log.trace('Number of image paths for training: ' .. #tpaths['train'])
log.trace('Number of image paths for testing: ' .. #tpaths['test'])
verify_images = false
optsImg.paths = tpaths
optsImg.imsize = opts.imsize
optsImg.verify_images = verify_images
dataProvider = DataProvider.create(optsImg)

log.trace('----------------------------------')
log.trace('-----[DATA SETUP COMPLETE]--------')
log.trace('----------------------------------')
print(opts)

autoencoder = nil
log.trace('=========[LOAD AUTOENCODER]=========')
autoencoder = nil
print(opts.autoencoder)
autoencoder = torch.load(opts.autoencoder)
log.trace('=========[done loading AUTOENCODER]=========')
autoencoder:clearState()
autoencoder:evaluate()
collectgarbage()
log.trace('Converting to cuda')
autoencoder:cuda()
autoencoder:evaluate()
log.trace('==========[SUCCESS!]==============\n')
print('=========[DONE LOADING MODEL]=========')

exprName = opts.timeStamp

print('Converting to cuda')
autoencoder:cuda()
print('done loading model state')
print('grab some test data!')

local allOpts = {}
allOpts['opts'] = opts
allOpts['optsImg'] = optsImg

function saveImage(imgIn)
	io.write('Save Image? [yes]')		
	io.flush()
	local user_input = io.read()
	if user_input == 'yes' then
		io.write('----> Name of file: ')		
		io.flush()
		user_input = io.read()
		image.save(user_input .. '_'.. exprName ..'.png', imgIn)
	end
end

local ndat = #dataProvider.train.paths
local numSteps = opts.numSteps
local nZ = opts.nLatentDims
local stepSize = opts.uR
local imSize = 256
local outDir = opts.outDir

local v = torch.FloatTensor(2)


if opts.img1 == 0 then 

	print('-----------[No image input index -- grabbing random subset to select from .....]')

	local numRows = 20
	io.write('How many data samples to look at at a time? ['.. numRows .. ']?: ')
	numRows_in = io.read()
	io.flush()
	if numRows_in ~= '' then
		numRows = tonumber(numRows_in)
	end

	local indices = torch.randperm(ndat):long():split(numRows)
	local img1  = 1
	local selected_set 
	for t,v in ipairs(indices) do
		x, x_out = dataProvider:getImages(v, 'train')
		local recon_train = utilsImg.evalImAuto(x, x_out)
		sampleView = image.toDisplayTensor(autoencoder.modules[2].output, 0, torch.floor(numRows^.5))
		gnuplot.figure(4);
		gnuplot.title('Training Samples and Autoencoder Reconstructions'); 
		gnuplot.imagesc(recon_train[1])	
		io.write('Examine next batch? [type stop to quit/enter to continue] : ')	
		io.flush()
		user_input = io.read()
		if user_input == 'stop' then
			print('moving on...')
			selected_set = v
			break
		end 
	end

	print(selected_set)

	io.write('Select Starting image [enter for '.. img1 .. ']?: ')
	imgSel_in = io.read()
	io.flush()
	if imgSel_in ~= '' then
		img1 = tonumber(imgSel_in)
		img1 = selected_set[img1]
	end
	print('image1 = ' .. img1)	
else
	img1 = opts.img1
end

v[1] = img1
v[2] = img1

while true do

	if opts.zeroOrigin == 0 then
		x, x_out, imgPathName = dataProvider:getImages(v, 'train', nil, opts.gaussSigmaIn, opts.lcn)
		local imSize = x_out:size(3)
		local recon_train = utilsImg.evalImAuto(x, x_out)
		sampleView_img = image.toDisplayTensor(autoencoder.modules[2].output, 0, torch.floor(2^.5))
		gnuplot.figure(1);
		gnuplot.title('Origin Image and Autoencoder Reconstruction'); 
		gnuplot.imagesc(recon_train[1])	


		print('-----[ImageName]-------------------------------')
		print(imgPathName)
		print('-----------------------------------------------')

		xHat = autoencoder:forward(x:cuda());
		codes = autoencoder.modules[1].output
		origin = codes[1]:clone():float()
	else
		origin = torch.zeros(nZ, 1):float()
		v[1] = 0
		
	end
	
	local shCode = {}
	shCode['+'] = {}
	shCode['-'] = {}

  	local tic = torch.tic()
  	
  	-- ex. (5(step) x 56(ndims) x 56 (size of vectors))
	shCode['+'] = torch.FloatTensor(numSteps, nZ, nZ)
	shCode['-'] = torch.FloatTensor(numSteps, nZ, nZ)

	for i_step = 1, numSteps do
		local cTemp = torch.eye(nZ)*(stepSize*i_step)
		local tempZ_pos = {}
		local tempZ_neg = {}
		local tempStepPos = torch.FloatTensor(nZ, nZ)
		local tempStepNeg = torch.FloatTensor(nZ, nZ)
		for iZ = 1, cTemp:size(2) do
			-- ex. (56(ndims) x 56 (size of vectors))
			tempZ_pos[iZ] = cTemp:index(2, torch.LongTensor{iZ}) + origin
			tempZ_neg[iZ] = -1*cTemp:index(2, torch.LongTensor{iZ}) + origin
			
			tempStepPos:select(1, iZ):copy(tempZ_pos[iZ])
			tempStepNeg:select(1, iZ):copy(tempZ_neg[iZ])
		end
		shCode['-']:select(1, i_step):copy(tempStepNeg)
		shCode['+']:select(1, i_step):copy(tempStepPos)
	end

	-- reverse the negative direction for viewing convenience
	seq_len = shCode['-']:size(1)
	local allShift = torch.cat(shCode['-']:index(1, torch.linspace(seq_len,1,seq_len):long()) , shCode['+'], 1)
	local zShiftlist = allShift:reshape(nZ*numSteps*2, nZ)
	local zlistSize = zShiftlist:size(1)
	local toc = torch.toc(tic)
	print('done constructing matricies in ' .. toc .. ' seconds')

	----------------------------------------------------------------
    reconX = torch.zeros(zlistSize, 1, imSize,imSize)
    indices = torch.linspace(1,zlistSize,zlistSize):long():split(opts.batchSize)
    start = 1
    tic = torch.tic() 
    for t,vp in ipairs(indices) do
        collectgarbage()
        stop = start + vp:size(1) - 1
        local z_in = zShiftlist:index(1, vp)
		autoencoder.modules[2]:forward(z_in:cuda());		
		local xOut_batch = autoencoder.modules[2].output:float()
        reconX:sub(start, stop):copy(xOut_batch)
        
        start = stop + 1
    end
	toc = torch.toc(tic)
	print('done reconstructing images in ' .. toc .. ' seconds')
	
	----------------------------------------------------------------
	sampleView_recon = image.toDisplayTensor(reconX, 0, nZ)
	print('...saving image ...')
	imgFileName = 'zShift_exploreGallery_lcn'..opts.lcn..'_idx'.. v[1] .. '-' .. stepSize .. '-'.. numSteps ..'.png'
	imgFileName = outDir .. '/' .. imgFileName
	paths.mkdir(outDir)
	image.save(imgFileName, sampleView_recon)
	print('Exported exploration gallery image to disk, please check ' .. imgFileName)

	io.write('How many steps in each direction to explore? ['.. numSteps .. ']?: ')
	numSteps_in = io.read()
	io.flush()
	if numSteps_in ~= '' then
		numSteps = tonumber(numSteps_in)
	end

	io.write('Size of each step? ['.. stepSize .. ']?: ')
	stepSize_in = io.read()
	io.flush()
	if stepSize_in ~= '' then
		stepSize = tonumber(stepSize_in)
	end

	io.write('Enter ORIGIN image index [1-'..ndat..'](hit enter to keep:' .. v[1] ..'): ')
	imgSel_in = io.read()
	io.flush()
	if imgSel_in ~= '' then
		img1 = tonumber(imgSel_in)
		v[1] = img1
		v[2] = img1
	end


end