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

## SCOPS on [Caltech-UCSD Birds](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)

## Test the pretrained model
**Note: The model is trained with two main differences in the master branch: 1) it is trained with ground truth silhouettes rather than saliency maps. 2) it crops birds w.r.t bounding boxes rather than using the original image.**

First set [image](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) and [annotation](https://github.com/akanazawa/cmr/issues/3#issuecomment-451757610) path in [line 35](https://github.com/sunshineatnoon/SCOPS/blob/master/dataset/cub.py#L35) and [line 37](https://github.com/sunshineatnoon/SCOPS/blob/master/dataset/cub.py#L37) in `dataset/cub.py`. Then run:

```sh eval_cub.sh```

Results as well as visualizations could be found in the `results/cub/ITER_60000/train/` folder.


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
