-- Adapted from the Allen Institute for Cell Science
-- https://github.com/AllenCellModeling/torch_integrated_cell/ 

local nn = require 'nn'
require 'dpnn'

local Model = {
  zSize = 10 --  -- Size of isotropic multivariate Gaussian Z
}

local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m:noBias()
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end

function Model:create(opts)
    self.scale_factor = 8/(512/opts.imsize)
    self.nLatentDims = opts.nLatentDims
    self.nChIn = opts.nChIn
    self.nChOut = opts.nChOut
    self.nOther = opts.nOther
    self.dropoutRate = opts.dropoutRate
    self.opts = opts
    
    Model:createAutoencoder()
    Model:createAdversary()
    Model:assemble()
end


function Model:assemble()
    self.autoencoder = nn.Sequential()
    self.autoencoder:add(self.encoder)
    self.autoencoder:add(self.decoder)    
end


function Model:createAutoencoder(X)
    -- local featureSize = X:size(2) * X:size(3)
    scale = self.scale_factor
    -- Create encoder (generator)
    -- Create encoder
    self.encoder = nn.Sequential()
    
    self.encoder:add(nn.SpatialConvolution(1, 64, 4, 4, 2, 2, 1, 1))
    -- self.encoder:add(nn.SpatialConvolution(self.nChIn, 64, 4, 4, 2, 2, 1, 1))
    self.encoder:add(nn.SpatialBatchNormalization(64))
    
    self.encoder:add(nn.PReLU())
    self.encoder:add(nn.SpatialConvolution(64, 128, 4, 4, 2, 2, 1, 1))
    self.encoder:add(nn.SpatialBatchNormalization(128))

    self.encoder:add(nn.PReLU())
    self.encoder:add(nn.SpatialConvolution(128, 256, 4, 4, 2, 2, 1, 1))
    self.encoder:add(nn.SpatialBatchNormalization(256))

    self.encoder:add(nn.PReLU())
    self.encoder:add(nn.SpatialConvolution(256, 512, 4, 4, 2, 2, 1, 1))
    self.encoder:add(nn.SpatialBatchNormalization(512))
    
    self.encoder:add(nn.PReLU())
    self.encoder:add(nn.SpatialConvolution(512, 1024, 4, 4, 2, 2, 1, 1))
    self.encoder:add(nn.SpatialBatchNormalization(1024))
    
    self.encoder:add(nn.PReLU())
    self.encoder:add(nn.SpatialConvolution(1024, 1024, 4, 4, 2, 2, 1, 1))
    self.encoder:add(nn.SpatialBatchNormalization(1024))
    
    self.encoder:add(nn.View(1024*scale*scale))  
    self.encoder:add(nn.PReLU())
    
    -- Latent embedding (z)
    self.encoder:add(nn.Linear(1024*scale*scale, self.nLatentDims))
    self.encoder:add(nn.BatchNormalization(self.nLatentDims)) -- maybe add back??
  
  -------------------------------
  -- Create decoder
    self.decoder = nn.Sequential()
    self.decoder:add(nn.Linear(self.nLatentDims, 1024*scale*scale))
    self.decoder:add(nn.View(1024, scale, scale))
    --
    self.decoder:add(nn.PReLU())    
    self.decoder:add(nn.SpatialFullConvolution(1024, 1024, 4, 4, 2, 2, 1, 1))
    self.decoder:add(nn.SpatialBatchNormalization(1024))
    
    self.decoder:add(nn.PReLU())    
    self.decoder:add(nn.SpatialFullConvolution(1024, 512, 4, 4, 2, 2, 1, 1))
    self.decoder:add(nn.SpatialBatchNormalization(512))
    
    self.decoder:add(nn.PReLU())    
    self.decoder:add(nn.SpatialFullConvolution(512, 256, 4, 4, 2, 2, 1, 1))
    self.decoder:add(nn.SpatialBatchNormalization(256))
    
    self.decoder:add(nn.PReLU())      
    self.decoder:add(nn.SpatialFullConvolution(256, 128, 4, 4, 2, 2, 1, 1))
    self.decoder:add(nn.SpatialBatchNormalization(128))
    
    self.decoder:add(nn.PReLU())      
    self.decoder:add(nn.SpatialFullConvolution(128, 64, 4, 4, 2, 2, 1, 1))
    self.decoder:add(nn.SpatialBatchNormalization(64))
    
    self.decoder:add(nn.PReLU())      
    self.decoder:add(nn.SpatialFullConvolution(64, 1, 4, 4, 2, 2, 1, 1))
    -- self.decoder:add(nn.SpatialFullConvolution(64, self.nChOut, 4, 4, 2, 2, 1, 1))
    self.decoder:add(nn.Sigmoid())

  
    self.encoder:apply(weights_init)
    self.decoder:apply(weights_init)
end

function Model:createAdversary()
  -- Create adversary (discriminator)
    noise = 0.1
    
    self.adversary = nn.Sequential()
    self.adversary:add(nn.Linear(self.nLatentDims, 1024))
    self.adversary:add(nn.LeakyReLU(0.2, true))
    self.adversary:add(nn.Linear(1024, 1024))
    self.adversary:add(nn.BatchNormalization(1024))
    self.adversary:add(nn.LeakyReLU(0.2, true))
    self.adversary:add(nn.Linear(1024, 512)) 
    self.adversary:add(nn.BatchNormalization(512))
    self.adversary:add(nn.LeakyReLU(0.2, true))
    self.adversary:add(nn.Linear(512, 1))
    self.adversary:add(nn.Sigmoid(true))

    self.adversary:apply(weights_init)
end
return Model
