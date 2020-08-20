require 'debugRepl'
local path = require 'pl.path'
setup = {}

function setup.getModel(model_opts_in)

    local model_opts = model_opts_in
    print('Preparring model type: ' .. model_opts.model_name)

    package.loaded['models/' .. model_opts.model_name] = nil
    Model = require ('models/' .. model_opts.model_name)
    autoencoder = nil
    encoder = nil
    decoder = nil
    adversary = nil
    collectgarbage()

        if paths.filep(model_opts.save_dir .. '/autoencoder_eval.t7') then

        print('---------[LOADING PREVIOUS MODEL]------------')
        print('Loading model from ' .. model_opts.save_dir)
        
        print('Loading adversary')
        adversary = torch.load(model_opts.save_dir .. '/adversary.t7')
        adversary:float()
        adversary:clearState()
        collectgarbage()

        print('Loading autoencoder')

        autoencoder = torch.load(model_opts.save_dir .. '/autoencoder_eval.t7')    
        autoencoder:float()
        autoencoder:clearState()
        collectgarbage()
        
        if paths.filep(model_opts.save_dir .. '/epoch.t7') then
            print('Loading epoch count')
            epoch_start = torch.load(model_opts.save_dir .. '/epoch.t7')    
            print('Last saved state at epoch '.. epoch_start)
            collectgarbage()
        end

        if paths.filep(model_opts.save_dir .. '/rng.t7') then
            print('Loading RNG')
            torch.setRNGState(torch.load(model_opts.save_dir .. '/rng.t7'))
            cutorch.setRNGState(torch.load(model_opts.save_dir .. '/rng_cuda.t7'))
        end        
        print('Done loading model')
    else
        print('---------[CREATING NEW MODEL]------------')
        print('Creating new model')
        print(model_opts)
        Model:create(model_opts)
        Model:createAutoencoder()
        paths.mkdir(model_opts.save_dir)
        print('Done creating model')
        
        adversary = Model.adversary
        -- if model_opts.load_autoencoder ~= 'none' then
        if #model_opts.load_autoencoder > 0 then -- and paths.filep then
            print('>>>NOTICE: LOADING SPECIFIC autoencoder<<< ')
            print(model_opts.load_autoencoder)
            autoencoder = torch.load(model_opts.load_autoencoder)
        else
            autoencoder = Model.autoencoder
        end
        Model = nil
    end
    
    -- Create loss
    criterion = nn.BCECriterion()
    softmax = nn.SoftMax()

    if cuda then
        print('Converting to cuda')
        
        adversary:cuda()
        autoencoder:cuda()
   
        criterion:cuda()
        softmax:cuda()        

        print('Done converting to cuda')
    end
    
    if model_opts.test_model then

        print('===============================')
        print('Testing ---- model')
        print(autoencoder)
        print('[PRE-TRAIN INITIALIZE] Testing encoder')

        adversary:evaluate()
        autoencoder:evaluate()

        print('Testing ---- model')
        local im_in, im_out = dataProvider:getImages(torch.LongTensor{1,1}, 'test')
        print('Testing encoder')
        local code = autoencoder.modules[1]:forward(im_in:cuda());
    
        print(im_in:type())
        print(im_in:size())

        print('Code size:')
        print(code)
        
        print(code:size(2))

        print('Testing decoder')
        local im_out_hat = autoencoder.modules[2]:forward(code)

        print('Out size:')
        print(im_out_hat:size())
        
        print(criterion:forward(im_out_hat, im_out:cuda()))

        print('Testing adversary')
        print(adversary:forward(code))
        print('[PRE-TRAIN INITIALIZE] SUCCESS!')
        print('===============================')

        autoencoder:training()
        adversary:training()

    end    
    
    cudnn.benchmark = false
    cudnn.fastest = false
    
    cudnn.convert(autoencoder, cudnn)
    cudnn.convert(adversary, cudnn)
   
    print('Done getting parameters')
end

function setup.getLearnOpts(model_opts)

    opt_path = model_opts.save_dir .. '/opt.t7'
    optEnc_path = model_opts.save_dir .. '/optEnc.t7'
    optDec_path = model_opts.save_dir .. '/optDec.t7'
    optAdv_path = model_opts.save_dir .. '/optD.t7'
    
    stateEnc_path = model_opts.save_dir .. '/stateEnc.t7'
    stateDec_path = model_opts.save_dir .. '/stateDec.t7'
    stateAdv_path = model_opts.save_dir .. '/stateD.t7'

    plots_path = model_opts.save_dir .. '/plots.t7'
    
    if paths.filep(opt_path) then
        print('Loading previous optimizer state')

        opt = torch.load(opt_path)

        optEnc = torch.load(optEnc_path)
        optDec = torch.load(optDec_path)
        optAdv = torch.load(optAdv_path)
        
        stateEnc = utils.table2cuda(torch.load(stateEnc_path))
        stateDec = utils.table2cuda(torch.load(stateDec_path))    
        stateAdv = utils.table2cuda(torch.load(stateAdv_path))
        
        plots = torch.load(plots_path)
        
        stateAdvGen = {}       
        losses = plots[1][3]:totable()
        latentlosses = plots[2][3]:totable()
        advlosses = plots[3][3]:totable()
        advMinimaxLoss = plots[4][3]:totable()
        reencodelosses = plots[5][3]:totable()

    else
        opt = {}

        opt.epoch = 0
        opt.nepochs = 2000
        opt.adversarial = true

        opt.learningRateA = 0.01
        opt.learningRateAdv = 0.01

        opt.min_rateA =  0.0000001
        opt.min_rateD =  0.0001
        opt.learningRateDecay = 0.999
        opt.optimizer = 'adam'
        opt.batchSize = 64
        opt.verbose = model_opts.verbose

        opt.update_thresh = 0.58
        opt.saveProgressIter = 5
        opt.saveStateIter = 50

        optEnc = {}
        optEnc.optimizer = opt.optimizer
        optEnc.learningRate = opt.learningRateA
        optEnc.min_rateA =  0.0000001
        optEnc.beta1 = 0.5        

        optDec = {}
        optDec.optimizer = opt.optimizer
        optDec.learningRate = opt.learningRateA
        optDec.min_rateA =  0.0000001
        optDec.beta1 = 0.5
        
        optAdv = {}
        optAdv.optimizer = opt.optimizer
        optAdv.learningRate = opt.learningRateAdv
        optAdv.min_rateA =  0.0000001
        optAdv.beta1 = 0.5        
        
        stateEnc = {}
        stateDec = {}
        stateAdv = {}
        
        losses, latentlosses, advlosses, advMinimaxLoss, advlossesGen, reencodelosses = {}, {}, {}, {}, {}, {}, {}
    end

    plots = {}
    loss, advloss = {}, {}
end

return setup

