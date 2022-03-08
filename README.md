# Install [Nvidia Driver, CUDA Toolkit, cuDNN] and [PyTorch, TensorFlow] with pip on Ubuntu 20.04, and train a demo CNN

As the title suggests, this tutorial 

0. starts with a fresh Ubuntu 20.04 install, 
1. goes through the installation of the Nvidia Driver, 
2. CUDA Toolkit (including nvcc), 
3. cuDNN library, 
4. python3.9 and python3.9-venv, 
5. and PyTorch (does not need CUDA Toolkit or cuDNN) and TensorFlow.
6. And ends with training a CIFAR 10 classifier using both Torch and TF. 



## 0. Ubuntu 20.04 (LTS)
Here, we start with newly installed Ubuntu 20.04.4, which comes with Python3.8. The installation procedure is described [here](https://ubuntu.com/tutorials/install-ubuntu-desktop#1-overview) and we assume that you selected "Install third-party software for graphics [...]" mentioned at [this step of the installation](https://ubuntu.com/tutorials/install-ubuntu-desktop#5-installation-setup). This additional software includes the Nvidia Driver.

## 1. Nvidia Driver
If you didn't install third-party software (e.g. Nvidia Driver) during Ubuntu install, you could follow one of the many online tutorials like [this one](https://www.cyberciti.biz/faq/ubuntu-linux-install-nvidia-driver-latest-proprietary-driver/), or install the driver via the toolkit installation below. Best case, the driver is the latest possible from `Software & Updates -> Additional Drivers`, and is proprietary and tested. You know that you are done when 

```
nvidia-smi
```
returns something like
```
Tue Mar  8 02:05:51 2022       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 510.47.03    Driver Version: 510.47.03    CUDA Version: 11.6     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0  On |                  N/A |
| N/A   42C    P8    11W /  N/A |     45MiB /  8192MiB |     14%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
```

Important: the CUDA version here (top right) should be as high as possible and at least the same as the CUDA Toolkit version.

## 2. [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
The toolkit contains (almost) all we need for ML applications, including the `nvcc` compiler. We need to select the proper version. Here, we want to use TensorFlow, so we look at the latest supported CUDA [here](https://www.tensorflow.org/install/gpu?hl=en). On 8th March 2022, it is CUDA 11.2. We select the latest CUDA Toolkit 11.2.x from the [Nvidia archive](https://developer.nvidia.com/cuda-toolkit-archive) and after specifying the system, apply the `runfile (local)` installation. In our case:

```
wget https://developer.download.nvidia.com/compute/cuda/11.2.2/local_installers/cuda_11.2.2_460.32.03_linux.run
sudo sh cuda_11.2.2_460.32.03_linux.run
```

When prompted by the *CUDA Installer*, if you don't already have Nvidia Driver, you can install it here. From the other options only *CUDA Toolkit 11.2* in necessary.

#### Post-Installation
We need to tell the command line where to find the toolkit installation. This is done by adding two lines to the `~/.bashrc` file. In our case:

```
export PATH=$PATH:/usr/local/cuda-11.2/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.2/lib64
```

## 3. [cuDNN](https://developer.nvidia.com/cudnn)
This library implements efficiently some deep learning primitives like CNN layer. Which [cuDNN version](https://developer.nvidia.com/rdp/cudnn-archive) to select? As long as the CUDA version is the same as of the Toolkit, the higher cuDNN version the better. Then, download *cuDNN Library for Linux (x86_64)* which should be a .tgz file. 


Once downloaded, unzip the contents and copy them to the corresponding `include` and `lib64` CUDA install folders in `/usr/local/cuda-11.2` (see steps inlcuding **chmod** [here](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installlinux-tar)). In our case:

```
tar -xvf cudnn-11.2-linux-x64-v8.1.0.77.tgz
sudo cp cuda/include/cudnn*.h /usr/local/cuda/include   # or to the cuda-11.2 folder
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64 
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```


## 4. Python 3.9 and Virtual Environments
Here, we 

1. demonstrate how to work with an arbitrary python 3.x version
```
sudo apt install python3.9
```
2. and how to create virtual environmens using that version. 
```
sudo apt install python3.9-venv  
python3.9 -m venv ./venv  # create empty virtual env
source venv/bin/activate  # begin working inside the environment
```


## 5. PyTorch, TensorFlow
Install latest versions of the libraries. We give an example of the latest versions as of 8th March 2022.

1. PyTorch [installation](https://pytorch.org/get-started/locally/).

```
pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

2. TensorFlow (JAX should behave similarly)
```
pip install tensorflow-gpu==2.8
```
Also, for TF we need to add the following line to `~/.bashrc` (see [TF installation](https://www.tensorflow.org/install/gpu?hl=en)):
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64 
```


## 6. Run tests

Get code from github.

```
git clone https://github.com/arturtoshev/test_torch_cuda.git
cd test_torch_cuda
pip3 install -r requirements.txt
```

Then, test the CUDA installation by running the two training scripts containing a GPU and a CPU training loops for 2 epochs. PyTorch code adapted from [this tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) and TensorFlow code from [here](https://www.tensorflow.org/tutorials/images/cnn). Total runtime should be less than a minute.


```
python main_torch.py
python main_tf.py

```


