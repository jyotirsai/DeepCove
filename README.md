# DeepCove
This repo
uses a diffusion-based data augmentation strategy to produce test samples that aim to increase the
coverage of DNNs. The results show that diffusion models such as Denoising Diffusion
Probabilistic Models (DDPMs) and Denoising Diffusion Implicit Models (DDIMs) are capable of
producing many coverage increasing samples. However, their ability to produce quality samples
is largely dependent on their training time and underlying neural network. Therefore, DDPMs and
DDIMs can sometimes produce erroneous data that either do not belong to any label or the wrong
label. To address this issue, a Top-K selection discriminator network is used to filter out erroneous
data to obtain a more accurate picture of the coverage. Diffusion models as well as other generative
models are effective at augmenting data and improving the robustness of neural networks.

| Model 	| NBC % Increase 	| SNAC % Increase 	| % Error 	| FID Score 	|
|-------	|----------------	|-----------------	|---------	|-----------	|
| DDPM  	| 2.91           	| 3.85            	| 2.26    	| 20.09     	|
| DDIM  	| 5.82           	| 5.13            	| 4.78    	| 49.51     	|

