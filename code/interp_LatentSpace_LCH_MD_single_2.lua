-- interp_LatentSpace_LCH_MD_single_2.lua
-- Synopsis: Script to generate synthetic images interpolated between two actual reconstructed images.
-- See Input/output Parameter options below.

-- Andrew R. Jamieson 
-- UT Southwestern Medical Center

-- BioHPC Environment configurations
-- module add cuda80; module add torch; module add cudnn/5.1.5; 

-- or Singularity containter:
-- singularity pull shub://andrewjUTSW/openLCH:latest
-- singularity exec --nv openLCH_latest.sif /bin/bash -c '<code>'

-- Running the script
-- module add cuda80; module add torch;module add cudnn/5.1.5; 
-- th -i ./interp_LatentSpace_LCH_MD_single_2.lua \
-- -imPathFile imageList3.txt \
-- -autoencoder output/autoencoder_eval.t7 \
-- -outDir output/interpOut/ \
-- -img1 1 \
-- -img2 2 \
-- -gpu 2

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
cmd:option('-autoencoder', 'autoencoder t7 file to load (should be _eval)')
cmd:option('-imPathFile',  'text file with one image path per line')
cmd:option('-dataProvider', 'DynDataProviderRobust_2', 'data provider object for rendering images from storage')
cmd:option('-outDir',      './interpOut/', 'path where to dump saved image galleries')
cmd:option('-img1',         1, 'Start image index')
cmd:option('-img2',         2, 'end image index')
cmd:option('-gpu',          1, 'Which GPU device to use')
cmd:option('-imsize',       256, 'desired size of images in dataset')

opts = cmd:parse(arg)
opts.timeStamp = os.date("%d%b%y%H%M")

print('==============================')
print(opts)
print('==============================')

local outDir = opts.outDir .. opts.timeStamp

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
optsImg.paths = tpaths
optsImg.imsize = opts.imsize
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
	io.write('Save Image? [yes/no]')		
	io.flush()
	local user_input = io.read()
	if user_input == 'yes' then
		io.write('----> Name of file: ')		
		io.flush()
		user_input = io.read()
		image.save(user_input .. '_'.. exprName ..'.png', imgIn)
	end
end


function saveIndividualImages(xOutIn, start_imageIn, end_imageIn)
	io.write('Save individual Images? [no/yes]')		
	io.flush()
	local user_input = io.read()
	if user_input == 'yes' then
		io.write('----> Name of file: ')		
		io.flush()
		user_input = io.read()
		paths.mkdir(outDir)
		for i = 1, xOut:size(1) do
			image.save(outDir .. '/' .. user_input .. '_'.. exprName ..'_i' .. i .. '.png', xOut[i][1])
		end
		image.save(outDir .. '/' .. user_input .. '_start_image_' .. exprName ..'.png', start_imageIn)
		image.save(outDir .. '/' .. user_input .. '_end_image_' .. exprName ..'.png', end_imageIn)
	end
end

local ndat = #dataProvider.train.paths
local ntest = #dataProvider.test.paths
local batchSize = 25

print('=========[Select two images for linear interpolation]=========')

local v = torch.FloatTensor(2)
v[1] = opts.img1
v[2] = opts.img2

while true do

	x, x_out, imgPathName = dataProvider:getImages(v, 'train')
	local recon_train = utilsImg.evalImAuto(x, x_out)
	sampleView_img = image.toDisplayTensor(autoencoder.modules[2].output, 0, torch.floor(2^.5))
	gnuplot.figure(4);
	gnuplot.title('Training Samples and Autoencoder Reconstructions'); 
	gnuplot.imagesc(recon_train[1])	

	xHat = autoencoder:forward(x:cuda());
	codes = autoencoder.modules[1].output

	code1 = codes[1]
	code2 = codes[2]

	io.write('How many steps for interpolation? ['.. batchSize .. ']?: ')
	batchSize_in = io.read()
	io.flush()
	if batchSize_in ~= '' then
		batchSize = tonumber(batchSize_in)
	end
	line  = torch.linspace(0, 1, batchSize)
	zLin = torch.FloatTensor(batchSize, code1:size(1)):cuda()
	for i = 1, batchSize do
		zLin:select(1, i):copy(code1 * line[i] + code2 * (1 - line[i]))
	end

	autoencoder.modules[2]:forward(zLin);		
	xOut = autoencoder.modules[2].output:clone()
	
	start_image = x[1]:clone()
	end_image = x[2]:clone()

	-- Create MSE measurement holders for both
	local tMSE_start = {}
	local tMSE_end = {}
	local dist_code1 = {}
	local dist_code2 = {}
	for i_z = 1,xOut:size(1) do
		tMSE_start[i_z] = mse:forward(start_image:cuda(), xOut[i_z])
		tMSE_end[i_z] = mse:forward(end_image:cuda(), xOut[i_z])
		dist_code1[i_z] = torch.dist(code1, zLin[i_z])
		dist_code2[i_z] = torch.dist(code2, zLin[i_z])
	end

	sampleView_recon = image.toDisplayTensor(xOut, 0, torch.floor(batchSize^.5))
	gnuplot.figure(888);
	gnuplot.title('Linear interpolation between two (REAL) images');
	gnuplot.imagesc(sampleView_recon[1])
	print('-----------------')
	print(imgPathName)
	print('-----------------')
		
	print('-----------------')
	print(tMSE_end)
	print(tMSE_start)
	print('-----------------')

	saveIndividualImages(xOut,start_image,end_image)

	io.write('Save MSE into text file? [type yes to continue, otherwise skipped] : ')	
	io.flush()
	user_input = io.read()
	if user_input == 'yes' then
		
		table.insert(tMSE_end, v[1])
		table.insert(tMSE_end, imgPathName[2])
		table.insert(tMSE_start, v[2])
		table.insert(tMSE_start, imgPathName[1])
		
		io.write('----> Name of file: ')		
		io.flush()
		user_input = io.read()
	    utils.csv_write(user_input .. '_MSE_' .. v[1] .. '-' .. v[2].. exprName .. '.csv', {tMSE_start,tMSE_end, dist_code1, dist_code2})

	    print('... saving original images ...')
	    image.save(user_input .. '_2images_' .. v[1] .. '-' .. v[2] .. exprName .. '.png', recon_train[1])

		print('DONE saving MSE data! ')
	end	

	io.write('Examine another pair? [type stop to quit/enter to continue] : ')	
	io.flush()
	user_input = io.read()
	if user_input == 'stop' then
		print('DONE! ')
		break
	end	

	io.write('Select Starting image: ')
	imgSel_in = io.read()
	io.flush()
	if imgSel_in ~= '' then
		img1 = tonumber(imgSel_in)
		v[1] = img1
	end
	print('image1 = ' .. img1)	

	io.write('Select ending image: ')
	imgSel_in = io.read()
	io.flush()
	if imgSel_in ~= '' then
		img2 = tonumber(imgSel_in)
		v[2] = img2
	end
	print('image2 = ' .. img2)
end