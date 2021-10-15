# Test if Torch+CUDA works

Test if the CIFAR10 classifier from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

## What to do:

```
git clone https://github.com/arturtoshev/test_torch_cuda.git
cd test_torch_cuda

python3 -m venv ./venv
source venv/bin/activate
pip install $latesttorch+CUDAversion
```

e.g. `pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html`

```
pip3 install -r requirements.txt
python3 main.py
```


