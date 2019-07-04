# SCOPS: Self-Supervised Co-Part Segmentation (CVPR 2019) 
[project_page](https://varunjampani.github.io/scops/)

PyTorch implementation for self-supervised co-part segmentation.

![](https://varunjampani.github.io/images/projectpic/scops_results.png)

## License

Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

## Paper
[paper](https://varunjampani.github.io/papers/hung19_SCOPS.pdf)

[supplementary](https://varunjampani.github.io/papers/hung19_SCOPS_supp.pdf)

## Installation

The code is developed based on Pytorch v0.4 with TensorboardX as visualization tools. We recommend to use virtualenv to run our code:

```
$ virtualenv -p python3 scops_env
$ source scops_env/bin/activate
(scops_env)$ pip install -r requirements.txt
```

To deactivate the virtual environment, run `$ deactivate`. To activate the environment again, run `$ source scops_env/bin/activate`.

## SCOPS on Unaligned CelebA

Download data (Saliency, labels, pretrained model) 

```$ ./download_CelebA.sh```

Download CelebA unaligned from [here](https://drive.google.com/open?id=0B7EVK8r0v71peklHb0pGdDl6R28).

## Test the pretrained model

```$ ./evaluate_celebAWild.sh``` and accept all default options. The results are stored in a single webpage at ```results_CelebA/SCOPS_K8/ITER_100000/web_html/index.html```.

## Train the model

```$ CUDA_VISIBLE_DEVICES={GPU} python train.py -f exps/SCOPS_K8_retrain.json``` where `{GPU}` is the GPU device number.

## Citation

Please consider citing our paper if you find this code useful for your research.

```
@inproceedings{hung:CVPR:2019,
	title = {SCOPS: Self-Supervised Co-Part Segmentation},
	author = {Hung, Wei-Chih and Jampani, Varun and Liu, Sifei and Molchanov, Pavlo and Yang, Ming-Hsuan and Kautz, Jan},
	booktitle = {IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
	month = june,
	year = {2019}
}
```
