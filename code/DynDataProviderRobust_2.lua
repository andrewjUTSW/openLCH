-- This DataProvider Dynamically loads images from the paths table provided.

require 'nn'
require 'torchx'
require 'image'

require 'debugRepl'
require 'imtools_LCH'
require 'utils'


local path = require 'pl.path'

local DataProvider = {
    data = nil,
    labels = nil,
    opts = nil,
    numThreads = 1
}

function DataProvider.create(opts)

    local verify_images = opts.verify_images
    local data = {}     
    local tpaths = opts.paths
    local imsize = opts.imsize
    local fail_load = {}

    if verify_images then
        local c = 0
        local length =  #tpaths['train']
        print('verifying ' .. #tpaths['train'] .. ' training images ... ')
        for i = 1,#tpaths['train'] do
            if not pcall(function() local im_tmp = image.load(tpaths['train'][i], 1, 'double') end) then
                print('Failed to verify image: ' .. tpaths['train'][i])
                table.insert(fail_load, tpaths['train'][i]) 
            end
            xlua.progress(i, length); 
            print(i .. ': ')
        end
        if #fail_load >= 1 then
            print('>>>>!!!!!== ====[FAILED VERIFY!!!!]<<<<=== ===')
            print(fail_load)
            print('###### ############## ##### ## ### ### #### ##')
        end
    end
    data.train = {}
    data.test = {}
    data.train.paths = tpaths['train'];
    data.test.paths = tpaths['test'];
    data.optsIn = opts
    if opts.numThreads then
        numThreads = opts.numThreads
    end
    data.fail_load = fail_load

    local self = data
    self.opts = {}
    setmetatable(self, { __index = DataProvider })    
    return self
end

function DataProvider:getImages(indices, train_or_test, im_scale, rotImageIn, gaussSigmaIn, lcnIn, lpfilterIn, lpfKernelSizeIn, illumShiftIn)

    local lpf = false
    local lpfilterIn = lpfilterIn -- 10
    if lpfilterIn == nil or lpfilterIn == 0 then
        lpf = false
    else
        lpf = true
    end

    
    local illumShift = 0
    local illumShiftIn = illumShiftIn
    if illumShiftIn == nil or illumShiftIn == 0 then
        illumShift = 0
    else
        illumShift = illumShiftIn
    end

    local lpfKernelSize = 25
    local lpfKernelSizeIn = lpfKernelSizeIn
    if lpfKernelSizeIn == nil then
        lpfKernelSize = 25
    else
        lpfKernelSize = lpfKernelSizeIn
    end

    local lcn = false
    local lcnIn = lcnIn
    if lcnIn == nil or lcnIn == 0 then
        lcn = false
    else
        lcn = true
    end
    
    local rotImage = rotImageIn
    if rotImage == nil or rotImage == 0 then
        rotImage = false
    end

    local gaussImg = false
    local gaussSigma = gaussSigmaIn
    
    if gaussSigmaIn == nil or gaussSigmaIn == 0 then
        gaussImg = false
    else
        gaussImg = true
    end

    if im_scale == nil then
        im_scale = self.optsIn.imsize;
    end

    if train_or_test == nil then
        train_or_test = 'train'
    end

    local p = {}
    for i = 1, indices:size(1) do
        p[i] = self[train_or_test].paths[indices[i]]
    end

    local im_tmp = image.load(p[1], 1, 'double')
    local im_size_tmp = im_tmp:size() -- place holder for the channel
    local im_size = torch.LongStorage(4) -- now for a set of vectors...
    
    im_size[1] = #p --  number of images (should be just one)
    im_size[2] = im_size_tmp[1] -- num channels
    im_size[3] = torch.round(im_scale) -- x 
    im_size[4] = torch.round(im_scale) -- y
    local im_size_x = torch.round(im_scale) -- x 
    local im_size_y = torch.round(im_scale) -- y

    local im_out = torch.Tensor(im_size)

    local gauss2D
    if gaussImg then
        gauss2D = image.gaussian{width=im_size_x, height=im_size_y, sigma_horz=gaussSigma,sigma_vert=gaussSigma,normalize=false}:double();
    end


    local gauss2D_lpf
    if lpf then
        gauss2D_lpf = image.gaussian(lpfKernelSize, lpfilterIn):double();
    end
    
    -- debugRepl()

    for i = 1, im_size[1] do
        local imIn = image.load(p[i], 1, 'double')
        

        if rotImage and torch.rand(1)[1] < 0.2 then
            rad = (torch.rand(1)*2*math.pi)[1]
            flip = torch.rand(1)[1]>0.5

            if flip then
                imIn = image.hflip(imIn)
            end
            imIn = image.rotate(imIn, rad);
        end

        if gaussImg then
            imIn = torch.cmul(imIn, gauss2D);        
        end

        if lpf then
            imIn = image.convolve(imIn,gauss2D_lpf,'same')
            image.minmax{tensor=imIn,inplace=true}
        end

        if lcn then
            imIn = image.convolve(imIn,image.laplacian(lcnIn),'same')
        end

        if illumShift ~= 0 then
            imIn = image.minmax{tensor=imIn+illumShift,min=0,max=1,saturate=true};        
        end

        im_out[i] = image.scale(imIn, im_scale, im_scale, 'bilinear')
        xlua.progress(i, im_size[1]);
    end
    
    collectgarbage()
    collectgarbage()

    return im_out, im_out:clone(), p
end


function DataProvider:getImagesParallel(indicesIn, train_or_test, im_scaleIn, batchSizeIn, rotImageIn, gaussSigmaIn, lcnIn, lpfilterIn, lpfKernelSizeIn, illumShiftIn)
    
    -- https://github.com/torch/threads/blob/master/test/test-threads-shared.lua
    print(data.optsIn)
    threads = require 'threads'
    threads.Threads.serialization('threads.sharedserialize')
    
    local nthread = numThreads
    print('num thread ' .. nthread)
    local batchSize = batchSizeIn
    local im_scale = im_scaleIn
    if batchSizeIn == nil then
        batchSize = 2500;
    end

    if im_scaleIn == nil then
        im_scale = self.optsIn.imsize;
    end

    local illumShift = 0
    local illumShiftIn = illumShiftIn
    if illumShiftIn == nil or illumShiftIn == 0 then
        illumShift = 0
    else
        illumShift = illumShiftIn
    end

    local train_or_test
    if train_or_test == nil then
        train_or_test = 'train'
    end

    local rotImage = rotImageIn
    if rotImage == nil or rotImage == 0 then
        rotImage = false
    end

    local gaussImg = false
    local gaussSigma = gaussSigmaIn -- .25
    if gaussSigmaIn == nil or gaussSigmaIn == 0 then
        gaussImg = false
    else
        gaussImg = true
    end


    local lpf = false
    local lpfilterIn = lpfilterIn -- 10
    if lpfilterIn == nil or lpfilterIn == 0 then
        lpf = false
    else
        lpf = true
    end

    local lpfKernelSize = 25
    local lpfKernelSizeIn = lpfKernelSizeIn -- 10
    if lpfKernelSizeIn == nil then
        lpfKernelSize = 25
    else
        lpfKernelSize = lpfKernelSizeIn
    end

    local lcn = false
    local lcnIn = lcnIn -- 10
    if lcnIn == nil or lcnIn == 0 then
        lcn = false
    else
        lcn = true
    end

    local lcn = false
    if lcnIn == nil or lcnIn == 0 then
        lcn = false
    else
        lcn = true
    end

    local p = {}
    for i = 1, indicesIn:size(1) do
        p[i] = self[train_or_test].paths[indicesIn[i]]
    end

    -- Load a single example to get dimensions...
    local im_tmp = image.load(p[1], 1, 'double')
    local im_size_tmp = im_tmp:size() -- place holder for the channel
    local im_chan_size = im_size_tmp[1] -- num channels
    local im_size_x = torch.round(im_scale) -- x 
    local im_size_y = torch.round(im_scale) -- y
    
    local gauss2D
    if gaussImg then
        gauss2D = image.gaussian{width=im_size_x, height=im_size_y, sigma_horz=gaussSigma,sigma_vert=gaussSigma,normalize=false}:double();
    end
    local lapKer
    if lcn then  
        lapKer = image.laplacian(lcnIn)
    end


    local gauss2D_lpf
    if lpf then
        gauss2D_lpf = image.gaussian(lpfKernelSize, lpfilterIn):double();
    end

    local tG = {}
    local indices = torch.linspace(1, #p, #p):long():split(batchSize)
    local njob = #indices;
    local i = 1
    for t,v in ipairs(indices) do
        local im_size = torch.LongStorage(4) -- now for a set of vectors...
        im_size[1] = v:size(1) --  number of images (should be just one)
        im_size[2] = im_chan_size
        im_size[3] = im_size_x
        im_size[4] = im_size_y        
        tG[i] = torch.Tensor(im_size);
        i = i + 1;
    end
    collectgarbage()    
    collectgarbage()
    local msg = 'jalo!'
    local pool = threads.Threads(
       nthread,
       function(threadid)
            require 'image'
        end,
       function(threadid)
          print('starting a new thread/state number ' .. threadid)
          gmsg = msg -- get it the msg upvalue and store it in thread state
       end
    )

    local jobdone = 0
    for i=1,njob do
       pool:addjob( 
          function() 
            local nP = indices[i]:size(1)
            print('Thread # ' .. __threadid .. ' -- size of indices ' .. nP)
            for k = 1, nP do
                im_idx = indices[i][k];

                local imIn = image.load(p[im_idx], 1, 'double');

                if rotImage and torch.rand(1)[1] < 0.1 then
                    rad = (torch.rand(1)*2*math.pi)[1]
                    flip = torch.rand(1)[1]>0.5

                    if flip then
                        imIn = image.hflip(imIn)
                    end
                    imIn = image.rotate(imIn, rad);
                end

                if gaussImg then
                    imIn = torch.cmul(imIn, gauss2D);
                end

                if lpf then
                    imIn = image.convolve(imIn,gauss2D_lpf,'same')
                    image.minmax{tensor=imIn,inplace=true}
                end

                if lcn then  
                    imIn = image.convolve(imIn,lapKer,'same')
                end

                if illumShift ~= 0 then
                    imIn = image.minmax{tensor=imIn+illumShift,min=0,max=1,saturate=true};        
                end

                tG[i][{k,{},{},{}}] = image.scale(imIn, im_scale, im_scale, 'bilinear');    


                if k % 250 == 0 then
                    xlua.progress(k, nP); 
                end
            end
             return __threadid
          end,

          function(id)
             jobdone = jobdone + 1
          end
       )
    end

    for i=1,njob do
       pool:addjob(
          function()
             collectgarbage()
             collectgarbage()
          end
       )
    end

    collectgarbage()
    collectgarbage()

    pool:synchronize()
    print(string.format('%d jobs done', jobdone))
    pool:terminate()
    collectgarbage()
    collectgarbage()
    
    local im_out = torch.cat(tG,1);
    return im_out, im_out:clone(), p
end

return DataProvider


