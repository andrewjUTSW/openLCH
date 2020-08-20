require 'torch'
require 'image'
require 'paths'
require 'colormap'
require 'utils'

imtools_LCH = {}

function imtools_LCH.loadSingleImage(im_path, im_scale)
    if im_scale == nil then
        im_scale = 64;
    end

    local p = im_path
    local im_tmp = image.load(p[1], 1, 'double')
    local im_size_tmp = im_tmp:size() -- place holder for the channel
    local im_size = torch.LongStorage(4) -- now for a set of vectors...
    
    im_size[1] = 1 --  number of images (should be just one)
    im_size[2] = im_size_tmp[1] -- num channels
    im_size[3] = torch.round(im_scale) -- x 
    im_size[4] = torch.round(im_scale) -- y
    
    local im_out = torch.Tensor(im_size)
    im_out = image.scale(image.load(p, 1, 'double'), im_scale, im_scale, 'bilinear')
    return im_out
end

function imtools_LCH.load_img(im_dir, im_pattern, im_scale)
    if im_scale == nil then
        im_scale = 64;
    end

    local p = {}
    local c = 1
    
    for f in paths.files(im_dir, im_pattern) do
        p[c] = im_dir .. f
        c = c+1
    end
    
    -- natural sort
    local p = utils.alphanumsort(p)
    
    local im_tmp = image.load(p[1], 1, 'double')
    local im_size_tmp = im_tmp:size() -- place holder for the channel
    local im_size = torch.LongStorage(4) -- now for a set of vectors...
    
    im_size[1] = #p --  number of images
    im_size[2] = im_size_tmp[1] -- num channels
    im_size[3] = torch.round(im_scale) -- x 
    im_size[4] = torch.round(im_scale) -- y
    
    local im_out = torch.Tensor(im_size)
    
    for i = 1,im_size[1] do
        im_out[i] = image.scale(image.load(p[i], 1, 'double'), im_scale, im_scale, 'bilinear')
    end
    
    return im_out, p
end


function imtools_LCH.mat2img(img)
    if img:size(2) < 3 then
        local padDims = 3-img:size(2)
        img = torch.cat(img, torch.zeros(img:size(1), padDims, img:size(4), img:size(4)):typeAs(img), 2)
    end
    return img
end