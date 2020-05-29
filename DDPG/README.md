# Deep Deterministic Policy Gradient

## Descriptions
This is a Pytorch implementation of [**Deep Deterministic Policy Gradient**](https://arxiv.org/pdf/1509.02971.pdf)

## Create the environment
* Create a virtual environment with python 3+
On macOS and Linux:
```bath
python3 -m venv env
```
On Windows:
```bath
py -m venv env
```
* activate the virtual environment by
```bath
source env/bin/activate
```
* run
```bath
pip install -r requirements.txt
```

## To train
```bath
python ddpg.py
```
The result can be visualized on tensorboard using
```bath
tensorboard --logdir=runs
```
This also saves model paramaters inside model/ directory
## To test
```bath
python ddpg.py --mode=test
```
The result can be visualized on tensorboard using
```bath
tensorboard --logdir=runs
```

## Result
According to [**here**](https://github.com/openai/gym/wiki/MountainCarContinuous-v0), getting a reward over 90 solves the MountainCar environment