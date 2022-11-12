# Install [Nvidia Driver, CUDA Toolkit, cuDNN] and [PyTorch, TensorFlow, JAX] with pip on Ubuntu 20.04, and train a demo CNN

This tutorial 

0. starts with a fresh Ubuntu 20.04 install, 
1. goes through the installation of the Nvidia Driver (520), 
2. CUDA Toolkit, which includes nvcc (11.8), 
3. cuDNN library (8.5.0), 
4. python3.9 and python3.9-venv, 
5. and PyTorch (1.13.0+cu117), TensorFlow (2.10.0), and JAX (cuda11_cudnn82).
6. And ends with training a CIFAR 10 classifier (PyTorch, TF) and MNIST (JAX). 

The versions of TyPorch, TensorFlow, and JAX are the latest as of 12th November 2022 and you should replace them by the latest at installation time.

> Note: **Pytorch does not need CUDA Toolkit or cuDNN** if it is installed via Conda or Pip. See the PyTorch forum [here](https://discuss.pytorch.org/t/please-help-me-understand-installation-for-cuda-on-linux/14217/4), [here](https://discuss.pytorch.org/t/please-help-me-understand-installation-for-cuda-on-linux/14217) and this [issue](https://github.com/pytorch/pytorch/issues/17445#issuecomment-466838819) (The forum links seem to work in Chrome but not in Firefox!).

## 0. Ubuntu 20.04 (LTS)
Here, we start with newly installed Ubuntu 20.04, which comes with Python3.8. The installation procedure is described [here](https://ubuntu.com/tutorials/install-ubuntu-desktop#1-overview) and we assume that you selected "Install third-party software for graphics [...]" at [this step of the installation](https://ubuntu.com/tutorials/install-ubuntu-desktop#5-installation-setup). This additional software includes the Nvidia Driver.

## 1. Nvidia Driver
If you didn't install the third-party software (including Nvidia Driver) during Ubuntu install, you could follow one of the many online tutorials like [this one](https://www.cyberciti.biz/faq/ubuntu-linux-install-nvidia-driver-latest-proprietary-driver/), or install the driver during the CUDA Toolkit installation described below. Best case, the driver is the latest possible from `Software & Updates -> Additional Drivers`, and is proprietary and tested, e.g. `Using NVIDIA driver metapackage from nvidia-driver-520 (proprietry)`. You know that you are done when 

```
nvidia-smi
```
returns something like
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 520.56.06    Driver Version: 520.56.06    CUDA Version: 11.8     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0  On |                  N/A |
| N/A   43C    P8     7W /  N/A |    329MiB /  8192MiB |     12%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      1759      G   /usr/lib/xorg/Xorg                 20MiB |
|    0   N/A  N/A      2688      G   /usr/lib/xorg/Xorg                 79MiB |
|    0   N/A  N/A      4204    C+G   ...156811687088679943,131072      227MiB |
+-----------------------------------------------------------------------------+
```

Important: the CUDA version here (top right) should be as high as possible and at least the same as the CUDA Toolkit version.

## 2. [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
The toolkit contains (almost) all we need for ML applications, including the `nvcc` compiler driver. We need to select the proper Toolkit version. Here, we demonstrate how to set up the latest CUDA 11.8 from the [Nvidia archive](https://developer.nvidia.com/cuda-toolkit-archive). After specifying the system, choose the `runfile (local)` installation. In our case:

```
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
```

When prompted by the *CUDA Installer*, if you don't already have Nvidia Driver, you should install it now. From the other suggested options only *CUDA Toolkit 11.8* is necessary.

#### Post-Installation
We need to tell the command line where to find the toolkit installation. This is done by adding two lines to the `~/.bashrc` file. In our case:

```
export PATH=$PATH:/usr/local/cuda-11.8/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/lib64
```

You know that the installation is done when you open a new terminal and 

```
nvcc -V
```
returns something like

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Sep_21_10:33:58_PDT_2022
Cuda compilation tools, release 11.8, V11.8.89
Build cuda_11.8.r11.8/compiler.31833905_0
```

## 3. [cuDNN](https://developer.nvidia.com/cudnn)
This library efficiently implements some deep learning primitives like CNN layer. Which [cuDNN version](https://developer.nvidia.com/rdp/cudnn-archive) to select? As long as the CUDA version is the same as of the Toolkit version, the higher cuDNN version the better. Then, download *cuDNN Library for Linux (x86_64)* which should be a `.tgz` or `.tar.xz` file. 


Once downloaded, unzip the contents and copy them to the corresponding `include` and `lib64` CUDA install folders in `/usr/local/cuda-11.8` (see steps inlcuding `chmod` [here](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installlinux-tar)). In our case:

```
tar xvf cudnn-linux-x86_64-8.5.0.96_cuda11-archive.tar.xz
cd cudnn-linux-x86_64-8.5.0.96_cuda11-archive
sudo cp include/cudnn*.h /usr/local/cuda-11.8/include 
sudo cp -P lib/libcudnn* /usr/local/cuda-11.8/lib64 
sudo chmod a+r /usr/local/cuda-11.8/include/cudnn*.h /usr/local/cuda-11.8/lib64/libcudnn*
```


## 4. Python 3.9 and Virtual Environments
Here, we 

1. demonstrate how to install an arbitrary python 3.x version
```
sudo apt install python3.9
```
2. and how to create virtual environmens using that version. 
```
sudo apt install python3.9-venv  
python3.9 -m venv ./venv  # create empty virtual env
source venv/bin/activate  # begin working inside the environment
pip install --upgrade pip
```

In the following, we assume that you have created and activated a new environment. 

## 5. PyTorch, TensorFlow, JAX
Here, you see how to install the latest versions of the three libraries as of 12th November 2022.

1. PyTorch [installation](https://pytorch.org/get-started/locally/). The only requirement is the Nvidia Driver (no CUDA Toolkit or CuDNN required as they come with PyTorch), whose version has to be >= the CUDA version below.

```
pip3 install torch==1.13.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
```

2. TODO: TensorFlow [installation](https://www.tensorflow.org/install/gpu?hl=en). Depends on CUDA Toolkit and cuDNN.
```
pip install tensorflow==2.10.0
```
Also, for TF we need to add the following line to `~/.bashrc`:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/extras/CUPTI/lib64 
```

3. JAX [installation](https://github.com/google/jax). JAX requires CUDA Toolkit and CuDNN, but doesn't care much about the versions. The Toolkit has to support CUDA >= 11.1, and cuDNN has to be >=8.0.5. For deep learning we need additional libraries, which are in the second line here.

```
pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -U dm-haiku  # additional dependencie for ML
```


## 6. Run tests

We recommend installing Torch, TF, and JAX in an *venv*, next to which we clone the repo with the scripts:

```
git clone https://github.com/arturtoshev/test_torch_cuda.git
cd test_torch_cuda
```

Now, test the CUDA installation by running the training scripts as shown below. Each script contains training a neural network for 2 epochs using a GPU and 2 epochs with CPU. PyTorch code adapted from [this tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html), TensorFlow code from [here](https://www.tensorflow.org/tutorials/images/cnn), and JAX code from [here](https://coderzcolumn.com/tutorials/artifical-intelligence/haiku-cnn). Total runtime should take around one minute. Note that the datasets will be downloaded in `.data/` (PyTorch and JAX) and `~/.keras/datasets/` (TensorFlow) and can be deleted afterwards.


```
python main_torch.py
python main_tf.py
python main_jax.py
```

Everything worked fine if the runs finished in less than two minutes and didn't give any errors. Also, GPU runs have to be at least 2x faster.
