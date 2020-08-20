# Instructions for downloading and running openLCH.
# Code will train autoencoder for embedding imaging in low-dimensional latent space.
# Also, example scripts for interpolation in the latent space and reconstructing synthetic images

# See associated preprint here: https://doi.org/10.1101/2020.05.15.096628 
# Github: https://github.com/andrewjUTSW/openLCH 


# Steps:
# 1) Download small data sample (2000) random images (about 50MB):
#  Compressed file link: https://cloud.biohpc.swmed.edu/index.php/s/FqZSqoKfHii6ony/download

curl https://cloud.biohpc.swmed.edu/index.php/s/FqZSqoKfHii6ony/download --output sample2000.tar.gz

# unzip data
tar xvzf ./sample2000.tar.gz

# Should now see directory called "data2"

# 2) generate image file path list to be used by the training script.

ls `pwd`/data2/*.png > imagePathList.txt

# "imagePathList.txt" should look something like this

head -n 3 ./imagePathList.txt 
/your/working/path/data2/140604_mv3_s02_t60_x389_y1840_t290_t7Ready_f00136.png
/your/working/path/data2/140604_mv3_s03_t60_x482_y1585_t184_t7Ready_f00075.png
/your/working/path/data2/140604_mv3_s07_t184_x713_y440_t308_t7Ready_f00008.png

# 3) Download sample code repo from github
git clone https://github.com/andrewjUTSW/openLCH


# 4) Set up working computing/GPU CUDA environment using Singularity (https://sylabs.io/singularity/)
# also see: https://portal.biohpc.swmed.edu/content/guides/singularity-containers-biohpc/

# Tested with Singularity 3.5.3 on RedHat Linux 7 (module add singularity/3.5.3)
# NOTE, requires CUDA 8.0+ compatible GPU (e.g., P100)
# pull singularity container
singularity pull shub://andrewjUTSW/openLCH:latest
# container binary also available here: https://cloud.biohpc.swmed.edu/index.php/s/a88iQABCbg7SWwi/download

# verify containter GPU access works
singularity exec --nv ./openLCH_latest.sif nvidia-smi # test GPU


# 5) Download our previously trained autoencoder use this link
# (stored as a torch .t7 file)
curl https://cloud.biohpc.swmed.edu/index.php/s/YAQQtpwTX2NKS89/download --output autoencoder_eval_56zTRAINED.t7

# (optional) Set CODE PATH for convenience in scripts below
export LCH_PATH=YOUR_CODE_PATH_HERE


# 6) Quick test of torch code with previously trained autoencoder [Interpolation example]
# Run example image interpolation in latent space to generate synthetic images between two sample images  
singularity exec --nv openLCH_latest.sif /bin/bash -c 'cd ./code; \
th -i ./interp_LatentSpace_LCH_MD_single_2.lua \
-imPathFile $LCH_PATH/imagePathList.txt \
-autoencoder $LCH_PATH/autoencoder_eval_56zTRAINED.t7 \
-outDir $LCH_PATH/output/interpOut/ \
-img1 501 \
-img2 801 \
-gpu 1'


# 7) Train new autoencoder on images, save output to "outputNew" directory
singularity exec --nv openLCH_latest.sif /bin/bash -c 'cd ./code; \
th ./run_mainLCH_AAE_Train_2.lua \
-nLatentDims 56 \
-imsize 256 \
-savedir $LCH_PATH/outputNew/ \
-imPathFile $LCH_PATH/imagePathList.txt \
-modelname AAEconv_CLEAN \
-epochs 100 \
-saveProgressIter 1 \
-saveStateIter 1 \
-batchSize 50 \
-batchSizeLoad 20000 \
-miniBatchSizeLoad 1000 \
-gpu 1 \
-useParallel 1 \
-epochs 25'


# 8) Extract new latent embedding vectors from newly trained Autoencoder
singularity exec --nv openLCH_latest.sif /bin/bash -c 'cd ./code; \
th ./call_DynComputeEmbeddingsRobust_2.lua \
-autoencoder $LCH_PATH/outputNew/autoencoder_eval.t7 \
-imsize 256 \
-dataProvider DynDataProviderRobust_2 \
-imPathFile $LCH_PATH/imagePathList.txt \
-batchSize 100 \
-batchSizeLoad 20000 \
-miniBatchSizeLoad 2500 \
-gpu 2 \
-useParallel 1 \
-numThreads 3 \
-embeddingFile $LCH_PATH/outputNew/embeddings_sampleTest.csv'

# 9) Reconstruction of Images from latent space.
# using pre-trained autoencoder and computed embeddings from previous step
# first 
singularity exec --nv openLCH_latest.sif /bin/bash -c 'cd ./code; \
th -i ./zLatent2ReconBatchLCH_2.lua \
-autoencoder $LCH_PATH/autoencoder_eval_56zTRAINED.t7 \
-zLatentFile $LCH_PATH/outputNew/embeddings_sampleTest.csv \
-reconPath $LCH_PATH/outputNew/zRecon/ \
-nLatentDims 56'

# Can download embddings vectors.


#10) Illustration of Latent Space Exploration by shifting embedding values ("zShift")

## Example using TRAINED autoencoder
singularity exec --nv openLCH_latest.sif /bin/bash -c 'cd ./code; \
th -i ./exploreZ_LatentSpace_LCH_single_2.lua \
-imPathFile $LCH_PATH/imagePathList.txt \
-autoencoder $LCH_PATH/autoencoder_eval_56zTRAINED.t7 \
-outDir $LCH_PATH/outputOrig/zExploreOut \
-img1 10 \
-uR 1 \
-numSteps 6'

# Example with newly trained autoencoder
singularity exec --nv openLCH_latest.sif /bin/bash -c 'cd ./code; \
th -i ./exploreZ_LatentSpace_LCH_single_2.lua \
-imPathFile $LCH_PATH/imagePathList.txt \
-autoencoder $LCH_PATH/outputNew/autoencoder_eval.t7 \
-outDir $LCH_PATH/outputNew/zExploreOut \
-img1 10 \
-uR .5 \
-numSteps 20 \
-nLatentDims 56'



