# [Live Cell Histology](https://www.biorxiv.org/content/10.1101/2020.05.15.096628v1)

Extracting latent features from label-free live cell images using [Adversarial Autoencoders](https://arxiv.org/abs/1511.05644)

#### Manuscript pre-print: https://www.biorxiv.org/content/10.1101/2020.05.15.096628v1 

![fig1](/img/LCH_smaller3_fig.png)
![interp](/img/VideoS3_PairInterpolationExample_1244485_465651.gif)

## Setup & Running Source Code 

Developed and tested on Red Hat Linux 7.

### Installation Steps

#### Containers

- Set-up compute environment with containers:
    - Install [Singularity](https://sylabs.io/docs/)
	- Tested with Singularity 3.5.3 
	- Need CUDA 8.0+ compatible GPU and drivers (e.g. P100)
	- Pull Singularity container .sif image file from [Singularity Hub](https://singularity-hub.org/)
		- alternatively, a copy can be found [here](https://cloud.biohpc.swmed.edu/index.php/s/a88iQABCbg7SWwi/download) 
	- `singularity pull shub://andrewjUTSW/openLCH:latest`
	- Test GPU `singularity exec --nv ./openLCH_latest.sif nvidia-smi`

#### Download and Prepare Example Data

- Download code and sample data 
	- unzip code and data
	- Included are a random sample set of 256x256 cell images 
	- create image list file with <bash script example>
	 
#### Run Example Scripts 

- Train AAE (run_mainLCH_AAE_Train_CLEAN.lua[link])
	- INPUT: image file list
	- OUTPUT: trained AAE
- Extract latent embeddings (call_DynComputeEmbeddingsRobust_CLEAN.lua)
	- ![dr](img/extractLatent.png)
- Interpolate between reconstructed cell images (interp_LatentSpace_LCH_MD_single_CLEAN.lua [link])
	- ![interp2](img/InterpExample.png)
- Explore Latent Space (exploreZ_LatentSpace_LCH_single_CLEAN.lua)
	- (snapshot?)
- Reconstruct images from latent codes (zLatent2ReconBatchLCH_CLEAN.lua)
	- ![recon](img/reconLatent.png)

## Citation
```bibtex
@article {Zaritsky2020.05.15.096628,
	author = {Zaritsky, Assaf and Jamieson, Andrew R. and Welf, Erik S. and Nevarez, Andres and Cillay, Justin and Eskiocak, Ugur and Cantarel, Brandi L. and Danuser, Gaudenz},
	title = {Interpretable deep learning of label-free live cell images uncovers functional hallmarks of highly-metastatic melanoma},
	elocation-id = {2020.05.15.096628},
	year = {2020},
	doi = {10.1101/2020.05.15.096628},
	URL = {https://www.biorxiv.org/content/early/2020/05/15/2020.05.15.096628},
	eprint = {https://www.biorxiv.org/content/early/2020/05/15/2020.05.15.096628.full.pdf},
	journal = {bioRxiv}
}
```

Inspiration: https://github.com/AllenCellModeling/torch_integrated_cell/
