# Conditional DDPM

## Introduction

We implement a simple conditional form of *Diffusion Model* described in [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239), in PyTorch. Preparing this repository, we inspired by the course [How Diffusion Models Work](https://www.deeplearning.ai/short-courses/how-diffusion-models-work) and the repository [minDiffusion](https://github.com/cloneofsimo/minDiffusion). While training, we use [MNIST](http://yann.lecun.com/exdb/mnist/), [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist), and Sprite (see [FrootsnVeggies](https://zrghr.itch.io/froots-and-veggies-culinary-pixels) and [kyrise](https://kyrise.itch.io/)) datasets.

## Setting Up the Environment

1. Install [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html), if not already installed.
2. Clone the repository
    ~~~
    git clone https://github.com/byrkbrk/diffusion-model.git
    ~~~
3. In the directory `diffusion-model`, for macos, run:
    ~~~
    conda env create -f diffusion-env_macos.yaml
    ~~~
    For linux or windows, run:
    ~~~
    conda env create -f diffusion-env_linux_or_windows.yaml
    ~~~
4. Activate the environment:
    ~~~
    conda activate diffusion-env
    ~~~

## Training and Sampling

### MNIST
To train the model on MNIST dataset from scratch,
~~~
python3 train.py --dataset-name mnist
~~~

In order to sample from our (pretrained) checkpoint:
~~~
python3 sample.py pretrained_mnist_checkpoint_49.pth --n-samples 400 --n-images-per-row 20
~~~

Results (jpeg and gif files) will be saved into `generated-images` directory, and are seen below where each two rows represents a class label (in total 20 rows and 10 classes).

<p align="center">
  <img src="files-for-readme/mnist_ddpm_images.jpeg" width="45%" />
  <img src="files-for-readme/mnist_ani.gif" width="45%" />
</p>

### Fashion-MNIST

To train the model from scratch on Fashion-MNIST dataset,
~~~
python3 train.py --dataset-name fashion_mnist
~~~

In order to sample from our (pretrained) checkpoint, run:
~~~
python3 sample.py pretrained_fashion_mnist_checkpoint_49.pth --n-samples 400 --n-images-per-row 20
~~~

Results (jpeg and gif files) will be saved into `generated-images` directory, and are seen below where each two rows represents a class label (in total 20 rows and 10 classes).

<p align="center">
  <img src="files-for-readme/fashion_mnist_ddpm_images.jpeg" width="45%" />
  <img src="files-for-readme/fashion_mnist_ani.gif" width="45%" />
</p>

### Sprite
To train the model from scratch on Sprite dataset:
~~~
python3 train.py --dataset-name sprite
~~~

In order to sample from our (pretrained) checkpoint, run:
~~~
python3 sample.py pretrained_sprite_checkpoint_49.pth --n-samples 225 --n-images-per-row 15
~~~

Results (jpeg and gif files) will be saved into `generated-images` directory, and are seen below where each three rows represents a class label (in total 15 rows and 5 classes).

<p align="center">
  <img src="files-for-readme/sprite_ddpm_images.jpeg" width="45%" />
  <img src="files-for-readme/sprite_ani.gif" width="45%" />
</p>

### Paired Stain Dataset
To train on a paired dataset located at `E:/patches` containing `stained` and `unstained` folders (pairs are returned as `(unstained, stained)`):
~~~
python3 train.py --dataset-name paired --dataset-path E:/patches
~~~

This will learn to generate stained images conditioned on their unstained counterparts.

To sample a stained image from an unstained slide using a trained model:
~~~
python3 sample.py pretrained_paired_checkpoint_XX.pth --context-image path/to/unstained.jpeg --n-samples 1
~~~

