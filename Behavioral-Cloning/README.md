# Behavioral Cloning

## Descriptions
This is a pytorch implementation of Behavioral Cloning.

Behavioral Cloning is nothing but a supervised problem. In this particular implementation, human expert data can be first recorded through expert_recorder.py (referred from https://github.com/MadcowD/tensorgym/blob/master/tf_demo/expert_recorder.py). In environment MountainCar, human expert can push the car to the left by pressing 'a', to the right by pressing 'd'. The goal is to reach the right side where the flag is. After the data are collected, the neural network is given a classification problem: to predict an action given a state, and also to improve the prediction by the incurred softmax loss between the performed action and the expert action.

## To train
First, record your own expert trajectories
```bath
python expert_recorder.py PATH_NAME/
```
Then do
```bath
python BC.py PATH_NAME/
```

