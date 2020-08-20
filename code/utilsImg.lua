require 'torch'
require 'debugRepl'
require 'gnuplot'

utilsImg = {}

-- Note: variables intertwined with the main script 
function utilsImg.evalImAuto(x_in, x_out)
    local xHat, latentHat, latentHat_var = autoencoder:forward(x_in:cuda())
    xHat = imtools_LCH.mat2img(xHat)
    x_out = imtools_LCH.mat2img(x_out)
    -- Plot reconstructions
    recon = torch.cat(image.toDisplayTensor(x_out, 1, 16), image.toDisplayTensor(xHat, 1, 16), 2)
    return recon
end

function utilsImg.evalImEn(x_in, x_out)
    local xHat, latentHat, latentHat_var = decoder:forward(encoder:forward(x_in:cuda()))
    xHat = imtools_LCH.mat2img(xHat)
    x_out = imtools_LCH.mat2img(x_out)
    recon = torch.cat(image.toDisplayTensor(x_out, 1, 16), image.toDisplayTensor(xHat, 1, 16), 2)
    return recon
end


function utilsImg.get_embeddings(encoder, dataProvider, train_or_test)
    
    local ndat = dataProvider[train_or_test].labels:size(1)
    local embeddings = torch.zeros(ndat, opts.nLatentDims)
    
    for j = 1, ndat do 
        print('Getting embedding for ' .. j .. '/' .. ndat)
        local img = dataProvider:getImages(torch.LongTensor{j,j}, train_or_test):cuda()
        codes = encoder:forward(img)
        
        embeddings[{j}]:copy(codes[#codes][1])
    end
    
    return embeddings
end


-- Generates a simple plot of the images
function utilsImg.showImg(x_in, titleIn, sizeSide)
    sizeSide = sizeSide or 10
    titleIn = titleIn or 'Displaying Images (no Title Provided)'
    local sampleView_recon = image.toDisplayTensor(x_in, 0, sizeSide)
    gnuplot.figure(8889);
    gnuplot.title(titleIn);
    gnuplot.imagesc(sampleView_recon[1])    
end